from functools import partial
from selectors import EpollSelector
from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np

import jax
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax import random


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# TODOs:
# - Add config file
# - Test the attention layer
# - Wrap model with transformers
# - Make forward pass work.
# - Port the weights and test equivalence.


def window_partition(tensor, window_size):
    batch, height, width, channels = tensor.shape
    tensor = tensor.reshape(
        batch, height // window_size[0], window_size[0], width // window_size[1], window_size[1], channels
    )
    # (num_windows*batch, window_size, window_size, channels)
    windows = jnp.transpose(tensor, (0, 1, 3, 2, 4, 5)).reshape(-1, window_size[0], window_size[1], channels)
    return windows


def window_reverse(windows, window_size, height, width):
    # windows: (num_windows*batch, window_size, window_size, channels)
    batch = int(windows.shape[0] / (height * width / window_size[0] / window_size[1]))
    tensor = windows.reshape(
        batch, height // window_size[0], width // window_size[1], window_size[0], window_size[1], -1
    )
    # (batch, height, width, channels)
    tensor = jnp.transpose(tensor, (0, 1, 3, 2, 4, 5)).reshape(batch, height, width, -1)
    return tensor


def to_patches(image, patch_size):
    batch, height, _, channels = image.shape
    num_patches = (height // patch_size) ** 2
    patch_height = patch_width = height // patch_size

    patches = image.reshape(batch, patch_height, patch_size, patch_width, patch_size, channels)
    # (batch, patch_height, patch_width, patch_size, patch_size, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    # (batch, patch_height*patch_width, patch_size * patch_size * channels)
    patches = patches.reshape(batch, num_patches, -1)
    return patches



class PatchEmbed(nn.Module):
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    patch_norm: bool = True

    def setup(self):
        patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        if self.patch_norm:
            self.norm = nn.LayerNorm(epsilon=1e-05)

    @nn.compact
    def __call__(self, pixel_values):
        hidden_states = to_patches(pixel_values, self.patch_size[0])
        
        if self.patch_norm:
            hidden_states = self.norm(hidden_states)
        
        return hidden_states


class PatchUnEmbed(nn.Module):
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    embed_dim = 96
    patch_norm: bool = True

    def setup(self):
        patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        if self.patch_norm:
            self.norm = nn.LayerNorm(epsilon=1e-05)

    @nn.compact
    def __call__(self, hidden_states):
        pass


class MLP(nn.Module):
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, hidden_states, deterministic=True):
        actual_out_dim = hidden_states.shape[-1] if self.out_dim is None else self.out_dim

        Dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

        hidden_states = Dense(features=self.hidden_dim)(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        hidden_states = Dense(features=actual_out_dim)(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        return hidden_states


class WindowAttention(nn.Module):
    dim: int
    window_size: Tuple[int]
    num_heads: int
    qkv_bias: Optional[bool] = True
    qk_scale: Optional[float] = None
    attn_drop_rate: Optional[float] = 0.0
    proj_drop_rate: Optional[float] = 0.0

    def setup(self):
        self.scale = self.qk_scale or self.head_dim**-0.5

        bias_shape = (
            (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
            self.num_heads,
        )  # # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.param("relative_position_bias_table", nn.zeros, bias_shape)

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        x, y = np.meshgrid(coords_h, coords_w)
        coords = np.stack([y, x])  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Wwbn

        self.qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias)

        self.attn_drop = nn.Dropout(rate=self.attn_drop_rate)
        self.proj = nn.Dense(features=self.dim)
        self.proj_drop = nn.Dropout(rate=self.proj_drop_rate)

    @nn.compact
    def __call__(self, hidden_states, mask=None, deterministic=True):
        B_, N, C = x.shape

        qkv = jnp.transpose(
            self.qkv(hidden_states).reshape(B_, N, 3, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + jnp.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + jnp.expand_dims(mask, axis=(0, 2))
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = nn.softmax(attn, axis=-1)
        else:
            attn = nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn, deterministic=deterministic)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)

        return x


def create_attn_mask(shift_size, input_resolution, window_size):
    if shift_size > 0:
        height, width = input_resolution
        img_mask = jnp.zeros((1, height, width, 1))
        h_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask = img_mask.at[:, h, w, :].set(cnt)
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size[0] * window_size[1])
        attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(mask_windows, axis=2)
        attn_mask = jnp.where(attn_mask == 0, x=float(0.0), y=float(-100.0))
    else:
        attn_mask = None

    return attn_mask


class SwinTransformerBlock(nn.Module):
    dim: int
    input_resolution: Tuple[int]
    num_heads: int
    window_size: Tuple[int]
    shift_size: int = 0
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0

    def setup(self):
        if min(self.input_resolution) <= max(self.window_size):
            self.shift_size2 = 0
            self.window_size2 = (min(self.input_resolution), min(self.input_resolution))
        else:
            self.shift_size2 = self.shift_size
            self.window_size2 = self.window_size

        assert 0 <= self.shift_size2 < min(self.window_size2)

        self.norm1 = nn.LayerNorm(epsilon=1e-05)
        self.attn = WindowAttention(
            self.dim,
            window_size=self.window_size2,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
        )

        self.norm2 = nn.LayerNorm(epsilon=1e-05)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, dropout_rate=self.drop_rate)

        self.attn_mask = create_attn_mask(self.shift_size2, self.input_resolution, self.window_size2)

    @nn.compact
    def __call__(self, hidden_states, deterministic=True):
        height, width = self.input_resolution
        batch, _, channels = hidden_states.shape

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels)

        # cyclic shift
        if self.shift_size2 > 0:
            hidden_states = jnp.roll(hidden_states, shift=(-self.shift_size2, -self.shift_size2), axis=(1, 2))

        # cyclic shift
        hidden_states = window_partition(hidden_states, self.window_size2)
        hidden_states = hidden_states.reshape(-1, self.window_size2[0] * self.window_size2[1], channels)

        # W-MSA/SW-MSA
        hidden_states = self.attn(hidden_states, mask=self.attn_mask, deterministic=deterministic)

        # merge windows
        hidden_states = hidden_states.reshape(-1, self.window_size2[0], self.window_size2[1], channels)
        hidden_states = window_reverse(hidden_states, self.window_size2, height, width)

        # reverse cyclic shift
        if self.shift_size2 > 0:
            hidden_states = jnp.roll(hidden_states, shift=(self.shift_size2, self.shift_size2), axis=(1, 2))

        hidden_states = hidden_states.reshape(batch, height * width, channels)

        # residual connection
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states

        return hidden_states


class BasicLayer(nn.Module):
    dim: int
    input_resolution: Tuple[int]
    depth: int
    num_heads: int
    window_size: Tuple[int]
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0

    def setup(self):
        self.blocks = [
            SwinTransformerBlock(
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % 2 == 0) else min(self.window_size) // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate[i]
                if isinstance(self.drop_path_rate, tuple)
                else self.drop_path_rate,
            )
            for i in range(self.depth)
        ]

    @nn.compact
    def __call__(self, hidden_states, deterministic=True):
        for block in self.blocks:
            hidden_states = block(hidden_states, deterministic=deterministic)


class Conv3Block(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.conv1 = nn.Conv(
            self.embed_dim // 4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )
        self.conv2 = nn.Conv(
            self.embed_dim // 4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        self.conv3 = nn.Conv(
            self.embed_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )
    
    def __call__(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class ResidualSwinBlock(nn.Module):
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: int = 6
    num_heads: int = 6
    window_size: Tuple[int] = (8, 8)
    mlp_ratio: int = 4
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    resi_connection: bool = "1conv"

    def setup(self):
        self.residual_group = BasicLayer(
            dim=self.embed_dim,
            input_resolution=self.img_size,
            depth=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
        )

        if self.resi_connection == "1conv":
            self.conv = nn.Conv(
                self.embed_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )
        elif self.resi_connection == "3conv":
            self.conv = Conv3Block(self.embed_dim)
        else:
            raise ValueError(f"Unknown resi_connection {self.resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            patch_norm=False
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            patch_norm=False
        )

    def __call__(self, hidden_states, deterministic=True):
        residual = hidden_states
        hidden_states = self.residual_group(hidden_states, deterministic=deterministic)
        hidden_states = self.patch_unembed(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SwinIR(nn.Module):
    img_size: Tuple[int] = (64, 64)
    patch_size: Tuple[int] = (1, 1)
    in_chans: int = 3
    embed_dim: int = 96
    depths: Tuple[int] = (6, 6, 6, 6)
    num_heads: Tuple[int] = (6, 6, 6, 6)
    window_size: Tuple[int] = (7, 7)
    mlp_ratio: int = 4
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    upsampler: str =''
    resi_connection: str = '1conv',

    def setup(self):
       pass

    def __call__(self, pixel_values, deterministic=True):
        pass
