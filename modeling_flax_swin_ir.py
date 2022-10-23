from functools import partial
from typing import Any, Optional, Tuple

import jax
import numpy as np
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import numpy as jnp
from transformers import FlaxPreTrainedModel

from configuration_swin_ir import SwinIRConfig

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
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    # (num_windows*batch, window_size, window_size, channels)
    windows = jnp.transpose(tensor, (0, 1, 3, 2, 4, 5)).reshape(-1, window_size, window_size, channels)
    return windows


def window_reverse(windows, window_size, height, width):
    # windows: (num_windows*batch, window_size, window_size, channels)
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    tensor = windows.reshape(batch, height // window_size, width // window_size, window_size, window_size, -1)
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
        hidden_states = to_patches(pixel_values, self.patch_size[0]) # (batch, num_patches, channels)

        if self.patch_norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class PatchUnEmbed(nn.Module):
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    embed_dim: int = 96

    def setup(self):
        patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

    @nn.compact
    def __call__(self, hidden_states, x_size):
        # (batch, num_patches, channels)
        batch, _, _ = hidden_states.shape 
        # (batch, height, width, channels)
        hidden_states = hidden_states.reshape(batch, x_size[0], x_size[1], self.embed_dim)
        return hidden_states


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
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

        hidden_states = Dense(features=self.hidden_dim, name="fc1")(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        hidden_states = Dense(features=actual_out_dim, name="fc2")(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(hidden_states, deterministic=deterministic)
        return hidden_states


class WindowAttention(nn.Module):
    dim: int
    window_size: int
    num_heads: int
    qkv_bias: Optional[bool] = True
    qk_scale: Optional[float] = None
    attn_drop_rate: Optional[float] = 0.0
    proj_drop_rate: Optional[float] = 0.0

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.qk_scale or self.head_dim**-0.5

        bias_shape = (
            (2 * self.window_size - 1) * (2 * self.window_size - 1),
            self.num_heads,
        )  # # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.param("relative_position_bias_table", nn.zeros, bias_shape)

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size)
        coords_w = np.arange(self.window_size)
        x, y = np.meshgrid(coords_h, coords_w)
        coords = np.stack([y, x])  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Wwbn

        self.qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias)

        self.attn_drop = nn.Dropout(rate=self.attn_drop_rate)
        self.proj = nn.Dense(features=self.dim)
        self.proj_drop = nn.Dropout(rate=self.proj_drop_rate)

    @nn.compact
    def __call__(self, hidden_states, mask=None, deterministic=True):
        B_, N, C = hidden_states.shape

        qkv = jnp.transpose(
            self.qkv(hidden_states).reshape(B_, N, 3, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = jnp.einsum("...hqd,...hkd->...hqk", q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape((-1,))]
        relative_position_bias = relative_position_bias.reshape(
            (self.window_size * self.window_size, self.window_size * self.window_size, -1)
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
        attn = attn + jnp.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + jnp.expand_dims(mask, axis=(0, 2))
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = nn.softmax(attn, axis=-1)
        else:
            attn = nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn, deterministic=deterministic)

        # x = attn @ v
        x = jnp.einsum("...hqk,...hkd->...hqd", attn, v)
        x = jnp.swapaxes(x, 1, 2)
        x = x.reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)

        return x


class SwinTransformerBlock(nn.Module):
    config: SwinIRConfig
    input_resolution: Tuple[int]
    num_heads: int
    shift_size: int = 0
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.window_size = self.config.window_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(epsilon=1e-05)
        self.attn = WindowAttention(
            self.config.embed_dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.config.qkv_bias,
            attn_drop_rate=self.config.attn_drop_rate,
            proj_drop_rate=self.config.drop_rate,
        )

        self.norm2 = nn.LayerNorm(epsilon=1e-05)
        mlp_hidden_dim = int(self.config.embed_dim * self.config.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, dropout_rate=self.config.drop_rate)

        if self.shift_size > 0:
            self.attn_mask = self.create_attn_mask(self.shift_size, self.input_resolution, self.window_size)
        else:
            self.attn_mask = None

    def create_attn_mask(self, shift_size, input_resolution, window_size):
        height, width = input_resolution
        img_mask = jnp.zeros((1, height, width, 1))

        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask = img_mask.at[:, h, w, :].set(cnt)
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size * window_size)

        attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(mask_windows, axis=2)

        attn_mask = jnp.where(attn_mask != 0.0, float(-100.0), attn_mask)
        attn_mask = jnp.where(attn_mask == 0.0, float(0.0), attn_mask)

        return attn_mask

    @nn.compact
    def __call__(self, hidden_states, x_size, deterministic=True):
        height, width = x_size
        batch, _, channels = hidden_states.shape

        # (batch, num_patches, channels)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels)

        # cyclic shift
        if self.shift_size > 0:
            hidden_states = jnp.roll(hidden_states, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))

        # cyclic shift
        # (num_windows*batch, window_size, window_size, channels)
        hidden_states = window_partition(hidden_states, self.window_size)
        # (num_windows*batch, window_size*window_size, channels)
        hidden_states = hidden_states.reshape(-1, self.window_size * self.window_size, channels)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            hidden_states = self.attn(hidden_states, mask=self.attn_mask, deterministic=deterministic)
        else:
            mask = self.create_attn_mask(self.shift_size, x_size, self.window_size)
            hidden_states = self.attn(hidden_states, mask=mask, deterministic=deterministic)

        # merge windows
        # (num_windows*batch, window_size, window_size, channels)
        hidden_states = hidden_states.reshape(-1, self.window_size, self.window_size, channels)
        # (batch, height, width, channels)
        hidden_states = window_reverse(hidden_states, self.window_size, height, width)

        # reverse cyclic shift
        if self.shift_size > 0:
            hidden_states = jnp.roll(hidden_states, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        # (batch, height*width, channels)
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
    config: SwinIRConfig
    input_resolution: Tuple[int]
    depth: int
    num_heads: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        depths = [i for i in range(self.depth)]
        self.blocks = [
            SwinTransformerBlock(
                self.config,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                shift_size=0 if (i % 2 == 0) else self.config.window_size // 2,
                dtype=self.dtype,
            )
            for i in depths
        ]

    def __call__(self, hidden_states, x_size, deterministic=True):
        for block in self.blocks:
            hidden_states = block(hidden_states, x_size, deterministic=deterministic)  # (batch, num_patches, channels)
        return hidden_states


class Conv3Block(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(
            self.dim // 4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            name="conv_0",
            dtype=self.dtype,
        )
        self.conv2 = nn.Conv(
            self.dim // 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            name="conv_2",
            dtype=self.dtype,
        )
        self.conv3 = nn.Conv(
            self.dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            name="conv_4",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class RSTB(nn.Module):
    config: SwinIRConfig
    input_resolution: Tuple[int]
    depths: int = 6
    num_heads: int = 6
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.embed_dim

        self.residual_group = BasicLayer(
            self.config,
            input_resolution=self.input_resolution,
            depth=self.depths,
            num_heads=self.num_heads,
            dtype=self.dtype,
        )

        if self.config.resi_connection == "1conv":
            self.conv = nn.Conv(
                self.embed_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
            )
        elif self.config.resi_connection == "3conv":
            self.conv = Conv3Block(self.embed_dim)
        else:
            raise ValueError(f"Unknown resi_connection {self.config.resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=self.config.img_size, patch_size=self.config.patch_size, patch_norm=False
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=self.config.img_size, patch_size=self.config.patch_size, embed_dim=self.embed_dim
        )

    def __call__(self, hidden_states, x_size, deterministic=True):
        # (batch, num_patches, channels)
        residual = hidden_states
        hidden_states = self.residual_group(hidden_states, x_size, deterministic=deterministic)
        # (batch, height, width, channels)
        hidden_states = self.patch_unembed(hidden_states, x_size)
        hidden_states = self.conv(hidden_states)
        # (batch, num_patches, channels)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SwinIRModule(nn.Module):
    config: SwinIRConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_feat = 64
        self.num_layers = len(self.config.depths)
        self.embed_dim = self.config.embed_dim
        self.num_features = self.config.embed_dim
        self.patch_size = self.config.patch_size
        self.img_size = self.config.img_size
        self.window_size = self.config.window_size

        conv_3x1X1 = partial(
            nn.Conv,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        if self.config.in_chans == 3:
            self.rgb_mean = jnp.array([0.4488, 0.4371, 0.4040]).reshape(1, 1, 1, 3)
        else:
            self.rgb_mean = jnp.zeros(1, 1, 1, 1)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = conv_3x1X1(self.embed_dim)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, patch_norm=self.config.patch_norm
        )
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim)

        # # absolute position embedding
        if self.config.ape:
            self.absolute_pos_embed = self.param(
                "absolute_pos_embed", jax.nn.initializers.zeros, (1, self.num_patches, self.embed_dim)
            )

        layers = []
        for i_layer in range(self.num_layers):
            layer = RSTB(
                config=self.config,
                input_resolution=self.patches_resolution,
                depths=self.config.depths[i_layer],
                num_heads=self.config.num_heads[i_layer],
                dtype=self.dtype,
            )
            layers.append(layer)
        self.layers = layers

        self.norm = nn.LayerNorm(epsilon=1e-5)

        # build the last conv layer in deep feature extraction
        if self.config.resi_connection == "1conv":
            self.conv_after_body = conv_3x1X1(self.embed_dim)
        elif self.config.resi_connection == "3conv":
            self.conv_after_body = Conv3Block(self.embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.config.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            self.conv_before_upsample = conv_3x1X1(self.num_feat, name="conv_before_upsample_0")
            self.conv_up1 = conv_3x1X1(self.num_feat)

            if self.config.upscale == 4:
                self.conv_up2 = conv_3x1X1(self.num_feat)

            self.conv_hr = conv_3x1X1(self.num_feat)
            self.conv_last = conv_3x1X1(self.config.in_chans)
        else:
            raise NotImplementedError

    def pad_image_if_needed(self, pixel_values):
        _, height, width, _ = pixel_values.shape
        mod_pad_h = (self.window_size - height % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - width % self.window_size) % self.window_size
        pixel_values = jnp.pad(pixel_values, ((0, 0), (0, mod_pad_h), (0, mod_pad_w), (0, 0)), "reflect")
        return pixel_values

    def __call__(self, pixel_values, deterministic=True):
        original_height, original_Width = pixel_values.shape[1], pixel_values.shape[2]

        pixel_values = self.pad_image_if_needed(pixel_values)
        pixel_values = (pixel_values - self.rgb_mean) * self.config.img_range

        # for real-world SR
        if self.config.upsampler == "nearest+conv":
            hidden_states = self.conv_first(pixel_values)
            residuel = hidden_states

            # body
            x_size = (hidden_states.shape[1], hidden_states.shape[2])
            hidden_states = self.patch_embed(hidden_states) # (batch, num_patches, channels)
            
            if self.config.ape:
                hidden_states = hidden_states + self.absolute_pos_embed

            for layer in self.layers:
                hidden_states = layer(hidden_states, x_size, deterministic=deterministic) # (batch, num_patches, channels)

            hidden_states = self.norm(hidden_states)
            hidden_states = self.patch_unembed(hidden_states, x_size) # (batch, height, width, channels)

            # end of body
            
            hidden_states = self.conv_after_body(hidden_states) + residuel
            
            hidden_states = self.conv_before_upsample(hidden_states)
            hidden_states = nn.leaky_relu(hidden_states)

            # upsample 2x
            batch, height, width, channel = hidden_states.shape
            hidden_states = jax.image.resize(
                hidden_states,
                shape=(batch, height * 2, width * 2, channel),
                method="nearest",
            )
            hidden_states = self.conv_up1(hidden_states)
            hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
            
            if self.config.upscale == 4:
                # upsample 2x
                batch, height, width, channel = hidden_states.shape
                hidden_states = jax.image.resize(
                    hidden_states,
                    shape=(batch, height * 2, width * 2, channel),
                    method="nearest",
                )
                hidden_states = self.conv_up2(hidden_states)
                hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
            
            hidden_states = self.conv_hr(hidden_states)
            hidden_states = nn.leaky_relu(hidden_states, negative_slope=0.2)
            hidden_states = self.conv_last(hidden_states)
        else:
            raise NotImplementedError

        hidden_states = hidden_states / self.config.img_range + self.rgb_mean

        return hidden_states[:, : original_height * self.config.upscale, : original_Width * self.config.upscale, :]


class SwinIRPretrainedModel(FlaxPreTrainedModel):
    config_class = SwinIRConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: SwinIRConfig,
        input_shape: Tuple = (1, 64, 64, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.module.init(rngs, pixel_values)["params"]

    def __call__(self, pixel_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values),
            not train,
            rngs=rngs,
        )


class SwinIR(SwinIRPretrainedModel):
    module_class = SwinIRModule
