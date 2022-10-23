from typing import Tuple

from transformers import PretrainedConfig


class SwinIRConfig(PretrainedConfig):
    def __init__(
        self,
        img_size: Tuple[int] = (64, 64),
        patch_size: Tuple[int] = (1, 1),
        in_chans: int = 3,
        embed_dim: int = 240,
        depths: Tuple[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6),
        num_heads: Tuple[int] = (8, 8, 8, 8, 8, 8, 8, 8, 8),
        window_size: int = 8,
        mlp_ratio: int = 2,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "nearest+conv",
        resi_connection: str = "3conv",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.upscale = upscale
        self.img_range = img_range
        self.upsampler = upsampler
        self.resi_connection = resi_connection
