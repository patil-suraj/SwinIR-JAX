from modeling_flax_swin_ir import *

model = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=(64, 64),
    window_size=8,
    img_range=1.0,
    depths=[1],
    embed_dim=240,
    num_heads=[8],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="3conv",
)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 68, 68, 3))

rngs = {"params": key}
params = model.init(rngs, x)["params"]
out = model.apply({"params": params}, x)
print(out.shape)
