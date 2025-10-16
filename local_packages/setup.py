from setuptools import setup, find_packages

setup(
    name="lightrag-hku-modified",
    version="1.4.9.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "networkx>=3.0",
        "tenacity>=8.0.0",
        "pydantic>=2.0.0",
        "xxhash>=3.0.0",
        "aiofiles>=23.0.0",
        "nano-vectordb>=0.1.0",
    ],
    description="Modified LightRAG package for Hugging Face deployment",
)
