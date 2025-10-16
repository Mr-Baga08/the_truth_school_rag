from setuptools import setup, find_packages

setup(
    name="lightrag-hku-modified",
    version="1.4.9.1",
    packages=find_packages(),
    install_requires=[
        "aiohttp",
        "configparser",
        "dotenv",
        "future",
        "json_repair",
        "nano-vectordb",
        "networkx",
        "numpy",
        "pandas>=2.0.0",
        "pipmaster",
        "pydantic",
        "pypinyin",
        "python-dotenv",
        "setuptools",
        "tenacity",
        "tiktoken",
        "xlsxwriter>=3.1.0",
    ],
    description="Modified LightRAG package for Hugging Face deployment",
)
