# SwinIR-JAX

Unofficial JAX implementation of [SwinIR](https://github.com/JingyunLiang/SwinIR).

As of now, this is repo only supports the Real SR task from SwinIR, pool requests are more than welcome for other tasks. 

## How to use

[ <a href="https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/patil-suraj/SwinIR-JAX/blob/main/swin_ir_jax.ipynb)

## Citation
    @article{liang2021swinir,
      title={SwinIR: Image Restoration Using Swin Transformer},
      author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
      journal={arXiv preprint arXiv:2108.10257},
      year={2021}
    }

## Acknowledgement
Some of the JAX code is adapted from the amazing JAX implementation of `SwinTransformer` from [jax-models](https://github.com/DarshanDeshpande/jax-models) repo by @DarshanDeshpande