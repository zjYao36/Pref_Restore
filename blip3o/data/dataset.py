import copy
import glob
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from blip3o.utils import rank0_print
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


## target transform for sana
target_transform = v2.Compose(
    [
        v2.Resize(512),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

# 把gpt信息中的<image>换成<im_start><image><im_end> 也就是说这是一段conversations中最后一个<im_start>和<im_end>
def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN # "<image>"
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    #tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>") # 最后一个id 217210
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end] # [198, 151644, 151645]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])


        # target += [IGNORE_INDEX] * len(input_id)
        target += input_id

        for conv in source:
            # Make sure blip3o data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}] # qwen的格式
            encode_id = tokenizer.apply_chat_template(conv)

            # import ipdb; ipdb.set_trace()
            # 解码查看实际的文本内容
            # '<|im_start|>user\n<image>\nPlease reconstruct the given image based on the image content: a photography of a woman with blonde hair and a bow on her head<|im_end|>\n'
            # '<|im_start|>assistant\n<im_start><image><im_end><|im_end|>\n'
            # <im_start>151669 <im_end>151670 != <|im_start|>151644 <|im_end|>151645
            # decoded_text = tokenizer.decode(encode_id, skip_special_tokens=False)

            input_id += encode_id
            if role in ["user", "system"]:
                # target += [IGNORE_INDEX] * len(encode_id)
                target += encode_id

            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx: # ["<|im_start|>", "<|im_start|>", "\n"] 不起作用
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX # 把所有<image>217210换成了 -200
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  
        labels=targets,  
    )



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_args
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        if os.path.isdir(data_path):
            # 如果 data_path 是一个目录, 递归查找其中的所有 .tar 文件
            files = glob.glob(os.path.join(data_path, "**", "*.tar"), recursive=True)
        else:
            # 否则, 假设它是一个文件路径或 glob 模式 (也支持递归)
            files = glob.glob(data_path, recursive=True)

        train_dataset = load_dataset("webdataset", data_files=files, split="train", num_proc=1, cache_dir='/data/zgq/yaozhengjian/Datasets/UniWorld-V1/data/BLIP3o-60k/webdataset')
        # train_dataset = load_dataset("webdataset", data_files=data_path, split="train", num_proc=1, cache_dir='/fsx/sfr/data/jiuhai/webdataset')
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["image", "txt", "type"])])
        print(f"finish loading image {len(train_dataset)}")
        list_data_dict.append(train_dataset)



        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)


        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.modality = torch.tensor(0) # 0 is for und task, 1 is for gen task


    def __len__(self):
        return len(self.list_data_dict)


    def process_image(self, image):
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality


    def process_target_image(self, image):
        image = target_transform(image)
        return image


    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]


            if sources["type"] == "T2I":

                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]


            elif sources["type"] == "I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]

            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]

                if not isinstance(image_files, list):
                    image_files = [image_files]

                images = []

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "I2I":
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error


                ## test if can apply img_process 
                if not images is None:
                    try:
                        process_images = [self.process_image(f) for f in images]
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None


                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue

                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])

            data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


            # image exist in the data
            if "image" in self.list_data_dict[i]:
                data_dict["image"] = process_images
                data_dict["target_image"] = [self.process_target_image(f) for f in images]

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict

class LazySupervisedRestoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_args
    ):
        super(LazySupervisedRestoreDataset, self).__init__()

        self.data_args = data_args

        datasets_to_merge = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path and os.path.isdir(path):
                    # 在每个路径下递归查找 parquet 和 tar 文件
                    parquet_files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
                    tar_files = glob.glob(os.path.join(path, "**", "*.tar"), recursive=True)
                    
                    # 处理 parquet 文件
                    if parquet_files:
                        # 原始加载方式
                        dataset = load_dataset("parquet", data_files=parquet_files, split="train", num_proc=1, cache_dir='/data/zgq/yaozhengjian/Datasets/FFHQ/cache')
                        # 重命名text列为txt
                        if "text" in dataset.column_names:
                            dataset = dataset.rename_column("text", "txt")
                        # 只保留需要的列
                        required_columns = ["image", "txt"]
                        columns_to_keep = [col for col in dataset.column_names if col in required_columns]
                        dataset = dataset.select_columns(columns_to_keep)
                        # 添加类型列
                        dataset = dataset.add_column('type', len(dataset) * ['I2I'])
                        datasets_to_merge.append(dataset)
                    
                    # 处理 tar 文件
                    if tar_files:
                        dataset = load_dataset("webdataset", data_files=tar_files, split="train", num_proc=1, cache_dir='/data/zgq/yaozhengjian/Datasets/UniWorld-V1/data/BLIP3o-60k/webdataset')
                        dataset = dataset.rename_column("jpg", "image")
                        dataset = dataset.add_column('type', len(dataset) * ['I2I'])
                        dataset = dataset.remove_columns([col for col in dataset.column_names if not col in (
                            ["image", "txt", "type"])])
                        datasets_to_merge.append(dataset)
                    '''
                    # 在每个路径下递归查找 parquet 文件
                    parquet_files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
                    if parquet_files:
                        # 为每个路径单独加载数据集
                        dataset = load_dataset("parquet", data_files=parquet_files, split="train", num_proc=1, cache_dir='/data/zgq/yaozhengjian/Datasets/FFHQ')
                        # 重命名text列为txt
                        if "text" in dataset.column_names:
                            dataset = dataset.rename_column("text", "txt")
                        # 只保留需要的列，过滤掉其他字段
                        required_columns = ["image", "txt"]
                        columns_to_keep = [col for col in dataset.column_names if col in required_columns]
                        dataset = dataset.select_columns(columns_to_keep)
                        # 添加类型列
                        dataset = dataset.add_column('type', len(dataset) * ['I2I'])
                        datasets_to_merge.append(dataset)
                    '''
        # 合并所有数据集
        if datasets_to_merge:
            train_dataset = concatenate_datasets(datasets_to_merge)
        else:
            raise ValueError(f"No valid parquet files found in paths from {data_path}")

        train_dataset = train_dataset.shuffle(seed=42)
        rank0_print(f"Total number of training instance: {len(train_dataset)}") # 110848
        self.tokenizer = tokenizer
        self.list_data_dict = train_dataset
        self.modality = torch.tensor(1) # 0 is for und task, 1 is for gen task
        
        # 图像退化配置
        self.degradation_params = {
            'gt_size': 512,
            'in_size': 512,
            'use_motion_kernel': False,
            'blur_kernel_size': 41,
            'blur_sigma': [1, 15],
            'downsample_range': [1, 30],
            # 'downsample_range': [1, 15],
            'noise_range': [0, 20],
            'jpeg_range': [30, 90]
        }
        # self.degradation_params = {
        #     'gt_size': 512,
        #     'in_size': 512,
        #     'use_motion_kernel': False,
        #     'blur_kernel_size': 41,
        #     'blur_sigma': [0.2, 10],
        #     'downsample_range': [1, 8],
        #     'noise_range': [0, 15],
        #     'jpeg_range': [60, 100]
        # }


    def __len__(self):
        return len(self.list_data_dict)


    def process_image(self, image):
        # === Image Processor Configuration ===
        # {'_processor_class': None, 'do_resize': True, 'size': (384, 384), 'resample': <Resampling.BICUBIC: 3>, 'do_rescale': True, 'rescale_factor': 0.00392156862745098, 'do_normalize': True, 'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.5, 0.5, 0.5], 'do_convert_rgb': None, 'crop_size': {'height': 384, 'width': 384}, 'image_processor_type': 'SiglipImageProcessor'}
        # key 384*384
        # =====================================
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality


    def process_target_image(self, image):
        image = target_transform(image)
        return image


    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]


            if sources["type"] == "T2I":

                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]


            elif sources["type"] == "I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image based on the image content: {sources['txt']}" if random.random() < 0.9 else "<image>\nPlease reconstruct the given image.",
                    }, # 90% 带caption 10%不带caption
                    {"from": "gpt", "value": "<image>"},
                ]
                # sources["conversations"] = [
                #     {
                #         "from": "human",
                #         "value": f"<image>\nPlease reconstruct the given image.",
                #     }, # 不带caption
                #     {"from": "gpt", "value": "<image>"},
                # ]

            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]

                if not isinstance(image_files, list):
                    image_files = [image_files]

                images = []

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "I2I":
                            img = img.convert("RGB") # PIL.Image.Image
                        else:
                            raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error


                ## test if can apply img_process 
                if not images is None:
                    try:
                        # 使用图像退化功能对images列表中的每个图像进行退化处理
                        from .image_degradation import degrade_image
                        # import ipdb; ipdb.set_trace()
                        # 对每个图像应用退化处理，使用类中配置的参数
                        # 以10%的概率直接使用原图（重建任务），90%的概率使用退化图像（复原任务）
                        if random.random() < 0.2:
                            degrade_images = images  # 直接使用原图，作为重建任务
                            sources["conversations"] = [
                                {
                                    "from": "human",
                                    "value": f"<image>\nThis is a side mission. Please complete the auxiliary reconstruction task by duplicating the given high-definition image.",
                                },
                                {"from": "gpt", "value": "<image>"},
                            ]
                        else:
                            degrade_images = [degrade_image(img, **self.degradation_params) for img in images]
                        images_for_llm = degrade_images + images
                        process_images = [self.process_image(f) for f in images_for_llm] # 过siglip的processor的resize, normalize等
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None


                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue

                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])
            # 文本变id然后<image>变-200
            data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


            # image exist in the data
            if "image" in self.list_data_dict[i]: # modality好像没用
                data_dict["image"] = process_images # Siglip images 一些3元组的列别[(torch.Size([3, 384, 384]), (512, 512), tensor(1)), .....]
                data_dict["target_image"] = [self.process_target_image(f) for f in images] # 默认改成了512x512, 要生成的目标图像 [torch.Size([3, 512, 512])]
                data_dict["detailed_condition"] = [self.process_target_image(f) for f in degrade_images] # 退化图像的tensor [torch.Size([3, 512, 512])]
            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None: # "<|endoftext|>"151643
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances] # 列表中的列表
            # 一维扁平化操作 List[List[Tuple]]-> List[Any], 如果只有一张图就是长度为batch size的列表
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images # 列表嵌套列表

            target_images = [instance["target_image"][0] for instance in instances] # target_image 只有一个
            target_images = torch.stack(target_images, dim=0) if target_images else None # [B, 3, 512, 512]
            batch["target_images"] = target_images

            if "detailed_condition" in instances[0]: # 退化图像
                detailed_conditions = [instance["detailed_condition"][0] for instance in instances] # target_image 只有一个
                detailed_conditions = torch.stack(detailed_conditions, dim=0) if detailed_conditions else None # [B, 3, 512, 512]
                batch["detailed_conditions"] = detailed_conditions


        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        return batch

def get_dataset_cls(name):

    if name == 'mix':
        dataset_cls = LazySupervisedMixDataset
    elif name == 'restore':
        dataset_cls = LazySupervisedRestoreDataset
    else:
        raise ValueError(f'Unknown dataset class {name}')
    return dataset_cls

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = get_dataset_cls(data_args.dataset_cls)
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)