import cv2
import math
import random
import numpy as np
import torch
from PIL import Image
import io
from . import gaussian_kernels


def degrade_image(img_gt, 
                  gt_size=512,
                  in_size=512,
                  use_motion_kernel=False,
                  motion_kernels=None,
                  motion_kernel_prob=0.001,
                  kernel_list=['iso', 'aniso'],
                  kernel_prob=[0.5, 0.5],
                  blur_kernel_size=41,
                  blur_sigma=[1, 15],
                  downsample_range=[4, 30],
                  noise_range=[0, 20],
                  jpeg_range=[30, 80]):
    """
    对输入图像进行退化处理，包括运动模糊、高斯模糊、下采样、噪声和JPEG压缩
    
    Args:
        img_gt (PIL.Image.Image): 输入的高质量PIL图像
        gt_size (int): 原始图像尺寸，默认512
        in_size (int): 输出图像尺寸，默认512
        use_motion_kernel (bool): 是否使用运动模糊，默认False
        motion_kernels (dict): 运动模糊核字典，默认None
        motion_kernel_prob (float): 运动模糊概率，默认0.001
        kernel_list (list): 高斯核类型列表，默认['iso', 'aniso']
        kernel_prob (list): 高斯核概率列表，默认[0.5, 0.5]
        blur_kernel_size (int): 模糊核大小，默认21
        blur_sigma (list): 模糊标准差范围，默认[0.2, 3]
        downsample_range (list): 下采样范围，默认[1, 4]
        noise_range (list): 噪声范围，默认[2, 25]
        jpeg_range (list): JPEG质量范围，默认[30, 95]
    
    Returns:
        PIL.Image.Image: 退化后的PIL图像
    """
    
    # 将PIL图像转换为numpy数组 (RGB -> BGR for OpenCV)
    img_array = np.array(img_gt)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_in = img_array.astype(np.float32) / 255.0
    
    # 运动模糊
    if use_motion_kernel and motion_kernels is not None and random.random() < motion_kernel_prob:
        m_i = random.randint(0, 31)
        k = motion_kernels[f'{m_i:02d}']
        img_in = cv2.filter2D(img_in, -1, k)
    
    # 高斯模糊
    kernel = gaussian_kernels.random_mixed_kernels(
        kernel_list,
        kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma, 
        [-math.pi, math.pi],
        noise_range=None)
    img_in = cv2.filter2D(img_in, -1, kernel)

    # 下采样
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    img_in = cv2.resize(img_in, (int(gt_size // scale), int(gt_size // scale)), interpolation=cv2.INTER_LINEAR)

    # 添加噪声
    if noise_range is not None:
        noise_sigma = np.random.uniform(noise_range[0] / 255., noise_range[1] / 255.)
        noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
        img_in = img_in + noise
        img_in = np.clip(img_in, 0, 1)

    # JPEG压缩
    if jpeg_range is not None:
        jpeg_p = np.random.uniform(jpeg_range[0], jpeg_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_p)]
        _, encimg = cv2.imencode('.jpg', img_in * 255., encode_param)
        img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.

    # 调整到目标尺寸
    img_in = cv2.resize(img_in, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
    
    # 转换回PIL图像格式 (BGR -> RGB)
    img_in = np.clip(img_in * 255.0, 0, 255).astype(np.uint8)
    if len(img_in.shape) == 3 and img_in.shape[2] == 3:
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_in)


def load_motion_kernels(motion_kernel_path='basicsr/data/motion-blur-kernels-32.pth'):
    """
    加载运动模糊核
    
    Args:
        motion_kernel_path (str): 运动模糊核文件路径
    
    Returns:
        dict: 运动模糊核字典
    """
    return torch.load(motion_kernel_path)


class ImageDegradationConfig:
    """
    图像退化配置类，用于方便地管理退化参数
    """
    def __init__(self,
                 gt_size=512,
                 in_size=512,
                 use_motion_kernel=False,
                 motion_kernel_path='basicsr/data/motion-blur-kernels-32.pth',
                 motion_kernel_prob=0.001,
                 kernel_list=['iso', 'aniso'],
                 kernel_prob=[0.5, 0.5],
                 blur_kernel_size=41,
                 blur_sigma=[1, 15],
                 downsample_range=[4, 30],
                 noise_range=[0, 20],
                 jpeg_range=[30, 80]):
        
        self.gt_size = gt_size
        self.in_size = in_size
        self.use_motion_kernel = use_motion_kernel
        self.motion_kernel_prob = motion_kernel_prob
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        
        # 加载运动模糊核
        self.motion_kernels = None
        if self.use_motion_kernel:
            self.motion_kernels = load_motion_kernels(motion_kernel_path)
    
    def degrade(self, img_gt):
        """
        使用配置参数对图像进行退化处理
        
        Args:
            img_gt (PIL.Image.Image): 输入的高质量PIL图像
        
        Returns:
            PIL.Image.Image: 退化后的PIL图像
        """
        return degrade_image(
            img_gt=img_gt,
            gt_size=self.gt_size,
            in_size=self.in_size,
            use_motion_kernel=self.use_motion_kernel,
            motion_kernels=self.motion_kernels,
            motion_kernel_prob=self.motion_kernel_prob,
            kernel_list=self.kernel_list,
            kernel_prob=self.kernel_prob,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
            downsample_range=self.downsample_range,
            noise_range=self.noise_range,
            jpeg_range=self.jpeg_range
        )


# 使用示例
if __name__ == "__main__":
    # 方法1: 直接使用函数
    # from PIL import Image
    # img_gt = Image.open("your_image.jpg")
    # img_degraded = degrade_image(img_gt)
    # img_degraded.save("degraded.jpg")
    
    # 方法2: 使用配置类
    config = ImageDegradationConfig(
        gt_size=512,
        in_size=512,
        use_motion_kernel=False,
        motion_kernels=None,
        kernel_list=['iso', 'aniso'],
        kernel_prob=[0.5, 0.5],
        blur_kernel_size=41,
        blur_sigma=[1, 15],
        downsample_range=[4, 30],
        noise_range=[0, 20],
        jpeg_range=[30, 80],
    )
    img_degraded = config.degrade(img_gt)
    # img_degraded.save("degraded.jpg")
    pass