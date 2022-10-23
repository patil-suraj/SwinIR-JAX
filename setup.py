import os
# To use a consistent encoding
from codecs import open

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="swin-ir-jax",
    version="0.0.1",
    description="JAX implementation of SwinIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["flax", "transformers"],
)