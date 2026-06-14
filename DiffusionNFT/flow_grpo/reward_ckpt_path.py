"""Resolves where to find the reward-model checkpoints.

Default layout (recommended): all reward weights live under
    <repo>/DiffusionNFT/reward_ckpts/
which is what the project README documents.

If you want to keep the weights elsewhere (e.g. on a shared scratch volume),
set the environment variable PREF_RESTORE_REWARD_CKPT_DIR=/your/path
and the loaders will pick that up automatically.
"""
import os

CKPT_PATH = os.environ.get(
    "PREF_RESTORE_REWARD_CKPT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reward_ckpts")),
)
