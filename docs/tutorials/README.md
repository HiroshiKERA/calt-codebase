# CALT Library Tutorials

CALT (Computer ALgebra with Transformer) is a Python library for learning arithmetic and symbolic computation using Transformer models.

## Overview

The CALT library provides the following features:

- **Basic Transformer Model**: Deep neural models to realize sequence-to-sequence functions
- **Efficient Dataset Construction**: Large-scale dataset construction through parallel processing
- **Customizable**: Users can implement custom instance generators for their own tasks
- **Mathematician-Friendly**: Focus on dataset construction without requiring deep learning expertise

## Tutorial Index

### 1. [Getting Started with CALT](./getting-started.md)
- CALT library installation
- Basic usage
- Environment setup

### 2. [Dataset Generation Tutorial](./dataset-generation.md)
- Polynomial addition task example
- Creating custom generators
- Best practices for dataset construction

### 3. [Model Training Tutorial](./model-training.md)
- Training Transformer models
- Hyperparameter tuning
- Training monitoring and evaluation

### 4. [Advanced Examples](./advanced-examples.md)
- Implementation of complex mathematical tasks
- Performance optimization
- Troubleshooting

## Research Examples

Research examples using the CALT library:

- ["Learning to Compute Gröbner Bases," Kera et al., 2024](https://arxiv.org/abs/2311.12904)
- ["Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms," Kera and Pelleriti et al., 2025](https://arxiv.org/abs/2505.23696)
- ["Geometric Generality of Transformer-Based Gröbner Basis Computation," Kambe et al., 2025](https://arxiv.org/abs/2504.12465)

For more details, see the paper ["CALT: A Library for Computer Algebra with Transformer," Kera et al., 2025](https://arxiv.org/abs/2506.08600).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kera2025calt,
  title={CALT: A Library for Computer Algebra with Transformer},
  author={Hiroshi Kera and Shun Arawaka and Yuta Sato},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.08600}
}
``` 