# calt-codebase

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://maximefaure.github.io/CALT/)
[![Built on calt-x](https://img.shields.io/badge/built%20on-calt--x-blue)](https://github.com/HiroshiKERA/calt)
[![Paper](https://img.shields.io/badge/arXiv-2506.08600-b31b1b)](https://arxiv.org/abs/2506.08600)

- [Documentation](https://maximefaure.github.io/CALT/)
- [CALT library](https://github.com/HiroshiKERA/calt) ([documentation](https://hiroshikera.github.io/calt/))
- [Demo Notebook (Google Colab)](https://colab.research.google.com/github/HiroshiKERA/calt/blob/main/examples/demos/minimal_demo.ipynb)
- [Task Builder (web app)](https://calt.mai2.org/)

Experiments for learning **algebraic computations** with Transformer models,
built on top of the [CALT library](https://github.com/HiroshiKERA/calt) 
(`calt-x`, available from [conda-forge](https://github.com/conda-forge/calt-x-feedstock)).
With the [Task Builder](https://calt.mai2.org/) you can set up a task in the
browser and download a ready-to-run project with the same layout as this
repository.

The idea: frame an algebraic problem as a *translation task* (input expression →
output expression) and let a Transformer learn the rule from many examples.

```
Input  : x^2 + 2*x | 3*x - 1        (two polynomials)
Output : 1 | x^2 - 1                 (their Gröbner basis, for instance)
```

> New here? Read the **[full documentation](https://maximefaure.github.io/CALT/)**
> (also available as [DOCUMENTATION.md](DOCUMENTATION.md)) — a walkthrough that
> assumes no prior knowledge of transformers or deep learning.
> Building an AI tool on top of this repo? See **[AI_CONTEXT.md](AI_CONTEXT.md)**.

---

## Structure

Each top-level directory is an independent **task** (a math problem the model learns to solve):

```
calt-codebase/
├── shared/          # Utilities shared across all tasks (seeds, configs, paths, plotting)
├── parity/          # Permutation parity: predict sign(σ) ∈ {+1, -1}
├── groebner_basis/  # Gröbner basis of ⟨f1, f2⟩
├── border_basis/    # Border basis of a zero-dimensional ideal
└── templates/       # Copy task_template/ to create a new task
```

Every task follows the same layout:

```
<task>/
├── README.md           # What is the math problem? How to run?
├── core/               # Reusable logic (importable from scripts)
│   ├── generator.py    # Produces (problem, answer) pairs — class with __call__(seed)
│   ├── formatter.py    # Math objects → strings
│   ├── parser.py       # Optional load-time preprocessor (strings → tokenizable text)
│   ├── metrics.py      # Per-sample stats + exact-match success rate
│   └── train.py        # Training runner (calls the CALT pipelines)
└── experiments/
    └── <name>/         # toy / scaling / ablation / finite_field …
        ├── configs/    # data.yaml, lexer.yaml, train.yaml
        └── scripts/    # generate.py, train.py, evaluate.py, run.sh
```

---

## Installation

We recommend installing everything in one dedicated conda environment, with
PyTorch from conda-forge in the same command:

```bash
# The quotes around the package specs are required
# With an NVIDIA GPU
conda create -n calt-env -c conda-forge python=3.12 "calt-x>=1.6.1" "pytorch>=2.6=cuda*" sage matplotlib click sortedcontainers
# Without an NVIDIA GPU (including macOS)
conda create -n calt-env -c conda-forge python=3.12 "calt-x>=1.6.1" "pytorch>=2.6" sage matplotlib click sortedcontainers
conda activate calt-env
# Prints e.g. "12.9 True" with a GPU, "None False" without one
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

SageMath (`sage`) is needed for the polynomial tasks (Gröbner basis, border
basis); `parity` runs without it. The GPU command needs NVIDIA driver 525 or
newer. If conda cannot see the GPU (e.g. on an HPC login node), prefix the
command with `CONDA_OVERRIDE_CUDA=12.9`. Do not install PyTorch with pip into
this environment: PyPI's CUDA build of torch can make conda-forge packages such
as SciPy and SageMath fail to import (``version `CXXABI_1.3.15' not found``).
See [CALT's installation notes](https://github.com/HiroshiKERA/calt#installation)
for details.

The commands pin Python 3.12 because this repository declares
`requires-python = ">=3.11,<3.13"` in `pyproject.toml`; `calt-x` itself supports
Python `>=3.11`.

You can check the available versions of `calt-x` with:

```bash
conda search calt-x --channel conda-forge
```

The conda-forge feedstock is available here: [conda-forge/calt-x-feedstock](https://github.com/conda-forge/calt-x-feedstock).

---

## Quick start

First, activate the conda environment that has the required dependencies installed:

```bash
conda activate calt-env
```

Then run the 3-step workflow:

```bash
cd groebner_basis/experiments/toy/scripts
python generate.py    # 1. generate data   → ../data/
python train.py       # 2. train model     → ../outputs/results_degrevlex/
python evaluate.py    # 3. report success rate + sample predictions
```

Or run all three at once:

```bash
bash run.sh
```

---

## Tasks

| Task | Problem | Field(s) |
|---|---|---|
| `parity` | Predict the sign of a permutation σ | — (integers) |
| `groebner_basis` | Predict the Gröbner basis of ⟨f1, f2⟩ | ℚ, GF(p), ℤ |
| `border_basis` | Predict the border basis of a 0-dimensional ideal | GF(p) |

---

## Adding a new task

```bash
cp -r templates/task_template my_new_task
# Edit my_new_task/core/generator.py  → implement TaskGenerator.__call__(seed)
# Edit my_new_task/core/formatter.py  → define the string format
# Edit my_new_task/core/parser.py     → only if data is stored as pickle/JSON
# Update configs in my_new_task/experiments/toy/configs/
```

See the checklist in `templates/task_template/README.md`.

---

## Dependencies

- Python `>=3.11,<3.13`
- [`calt-x`](https://github.com/HiroshiKERA/calt) `>=1.6.1` (from [conda-forge](https://github.com/conda-forge/calt-x-feedstock)) and PyTorch `>=2.6`, also from conda-forge
- SageMath (`sage`), required for polynomial tasks such as Gröbner basis and border basis experiments
- Experiment utilities: `matplotlib`, `click`, and `sortedcontainers`
- `wandb` is optional — used by `shared/logging.py`; set `no_wandb: true` in a `train.yaml` or set `WANDB_MODE=disabled` to skip it.

---

## Links

- CALT library: <https://github.com/HiroshiKERA/calt> · docs: <https://hiroshikera.github.io/calt/>
- Task Builder (web app): <https://calt.mai2.org/>
- Demo Notebook (Google Colab): <https://colab.research.google.com/github/HiroshiKERA/calt/blob/main/examples/demos/minimal_demo.ipynb>
- Paper: *CALT: A Library for Computer Algebra with Transformer*, Kera et al. — [arXiv:2506.08600](https://arxiv.org/abs/2506.08600)
- Border basis algorithm: *Computational Algebra with Attention*, Kera et al. — [arXiv:2505.23696](https://arxiv.org/abs/2505.23696)
