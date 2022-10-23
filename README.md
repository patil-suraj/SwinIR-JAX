# SwinIR-JAX

Unofficial JAX implementation of [SwinIR](https://github.com/JingyunLiang/SwinIR).

As of now, this repo only supports the Real SR task from SwinIR, pull requests are more than welcome for other tasks. 

## How to use

[ <a target="_blank" href="https://colab.research.google.com/github/patil-suraj/SwinIR-JAX/blob/main/swin_ir_jax.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/patil-suraj/SwinIR-JAX/blob/main/swin_ir_jax.ipynb)

The checkpoints are available on the [huggingface hub](https://huggingface.co/models?other=swin-ir).

## Citation
    @article{liang2021swinir,
      title={SwinIR: Image Restoration Using Swin Transformer},
      author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
      journal={arXiv preprint arXiv:2108.10257},
      year={2021}
    }

## Acknowledgement
Some of the JAX code is adapted from the amazing JAX implementation of `SwinTransformer` from [jax-models](https://github.com/DarshanDeshpande/jax-models) repo by @DarshanDeshpande
