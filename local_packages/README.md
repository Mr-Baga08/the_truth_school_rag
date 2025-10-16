# Modified LightRAG Package

This directory contains a modified version of the LightRAG package that is used in the Hugging Face deployment.

## Why This Exists

The standard `lightrag-hku` package from PyPI needed custom modifications for this project. Instead of maintaining a fork, we include the modified version directly in the repository for Hugging Face deployment.

## Structure

- `lightrag/` - The modified LightRAG package code
- `setup.py` - Setup configuration for installing the modified package
- `METADATA` - Original package metadata from lightrag-hku

## Installation

During Docker build, this package is installed with:
```bash
pip install -e /app/local_packages/
```

## Modifications

The key modifications from the standard lightrag-hku package are in:
- `lightrag/prompt.py` - Custom prompt modifications (timestamp: Oct 14 12:40)

## Version

Based on lightrag-hku version 1.4.9.1
