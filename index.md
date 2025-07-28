# CALT codebase

The CALT codebase provides a template for generating arithmetic and symbolic computation instances and training Transformer models using the [CALT (Computer ALgebra with Transformer) library](https://github.com/HiroshiKERA/calt).

While CALT can be installed via `pip`, the following is the simplest setup your experiment with all dependencies:

```
git clone https://github.com/HiroshiKERA/calt-codebase.git
cd calt-codebase
conda env create -f environment.yml 

```

The documentation of the CALT codebase provides a quickstart guide and tips for organizing your own projects. For detailed usage of the CALT library, please refer to the [CALT documentation](https://hiroshikera.github.io/calt/).

## Citation

If you use this code in your research, please cite our paper:

```
@misc{kera2025calt,
  title={CALT: A Library for Computer Algebra with Transformer},
  author={Hiroshi Kera and Shun Arawaka and Yuta Sato},
  year={2025},
  archivePrefix={arXiv},
  eprint={2506.08600}
}

```

The following is a small list of such studies from our group.

- ["Learning to Compute Gröbner Bases," Kera et al., 2024](https://arxiv.org/abs/2311.12904)
- ["Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms," Kera and Pelleriti et al., 2025](https://arxiv.org/abs/2505.23696)
- ["Geometric Generality of Transformer-Based Gröbner Basis Computation," Kambe et al., 2025](https://arxiv.org/abs/2504.12465)

Refer to our paper ["CALT: A Library for Computer Algebra with Transformer," Kera et al., 2025](https://arxiv.org/abs/2506.08600) for a comprehensive overview.
