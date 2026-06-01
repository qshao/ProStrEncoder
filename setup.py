from setuptools import setup, find_packages

setup(
    name="prostrencoder",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "biopython>=1.81",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
)
