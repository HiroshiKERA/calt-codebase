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

You can install the dependencies with conda or with pip. We recommend conda: it
installs everything in one dedicated environment, with PyTorch from conda-forge
in the same command, and it also provides SageMath (`sage`), which the
polynomial tasks (Gröbner basis, border basis) need. pip works for the tasks
that do not need SageMath, such as `parity`.

### With conda (recommended)

1. Create the environment. Keep the quotes around the package specs.

   With an NVIDIA GPU:

   ```bash
   conda create -n calt-env -c conda-forge "calt-x>=1.6.1" "pytorch>=2.6=cuda*" sage matplotlib click sortedcontainers
   ```

   Without an NVIDIA GPU (including macOS):

   ```bash
   conda create -n calt-env -c conda-forge "calt-x>=1.6.1" "pytorch>=2.6" sage matplotlib click sortedcontainers
   ```

2. Activate the environment:

   ```bash
   conda activate calt-env
   ```

3. Check PyTorch. It prints e.g. `12.9 True` with a GPU and `None False`
   without one.

   ```bash
   python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
   ```

This repository supports Python `>=3.11`, like `calt-x`; conda picks a Python
version that all the packages support. The available versions of `calt-x` on
conda-forge are listed by `conda search calt-x --channel conda-forge`, and the
feedstock is
[conda-forge/calt-x-feedstock](https://github.com/conda-forge/calt-x-feedstock).

### With pip (tasks without SageMath)

1. Create a virtual environment with Python 3.11 or newer, and activate it.
   Check the version with `python3 --version`; if it is older, use e.g.
   `python3.12` instead of `python3`. On Windows, run `py -3 -m venv calt-env`
   and `calt-env\Scripts\activate` instead.

   ```bash
   python3 -m venv calt-env
   source calt-env/bin/activate
   ```

2. Install the dependencies. pip installs PyTorch from PyPI along with `calt-x`.

   ```bash
   pip install calt-x matplotlib click
   ```

3. Check PyTorch. On a Linux machine with an NVIDIA GPU, PyPI's torch 2.11 and
   later needs driver 580 or newer; if this prints `False` there, run
   `pip install "torch<2.11"`.

   ```bash
   python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
   ```

### Notes on GPUs and conda

- **NVIDIA driver:** the conda GPU command (`"pytorch>=2.6=cuda*"`) needs driver
  525 or newer.
- **GPU not visible to conda** (e.g. on an HPC login node): prefix the command
  with `CONDA_OVERRIDE_CUDA=12.9`.
- **No pip torch in the conda environment:** PyPI's CUDA build of torch can make
  conda-forge packages such as SciPy and SageMath fail to import
  (``version `CXXABI_1.3.15' not found``).

See [CALT's installation notes](https://github.com/HiroshiKERA/calt#installation)
for details.

---

## Quick start

First, activate the conda environment that has the required dependencies installed:

```bash
conda activate calt-env
```

If you installed with pip, activate that environment instead
(`source calt-env/bin/activate`) and use `parity/experiments/toy/scripts`, since
`groebner_basis` needs SageMath.

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
# Edit my_new_task/core/parser.py     → only for JSON Lines data or load-time transformations
# Update configs in my_new_task/experiments/toy/configs/
```

See the checklist in `templates/task_template/README.md`.

---

## Dependencies

- Python `>=3.11`
- [`calt-x`](https://github.com/HiroshiKERA/calt) `>=1.6.1` (from [conda-forge](https://github.com/conda-forge/calt-x-feedstock), or PyPI with pip) and PyTorch `>=2.6`, from the same source
- SageMath (`sage`), required for polynomial tasks such as Gröbner basis and border basis experiments
- Experiment utilities: `matplotlib`, `click`, and `sortedcontainers`
- `wandb` is optional — used by `shared/logging.py`; set `no_wandb: true` in a `train.yaml` or set `WANDB_MODE=disabled` to skip it.

---

## Links

- CALT library: <https://github.com/HiroshiKERA/calt> · docs: <https://hiroshikera.github.io/calt/>
- Task Builder (web app): <https://calt.mai2.org/>
- Demo Notebook (Google Colab): <https://colab.research.google.com/github/HiroshiKERA/calt/blob/main/examples/demos/minimal_demo.ipynb>
- *CALT: A Library for Computer Algebra with Transformer*, Kera et al. — [arXiv:2506.08600](https://arxiv.org/abs/2506.08600)
- *Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms*, Kera et al. — [arXiv:2505.23696](https://arxiv.org/abs/2505.23696)
