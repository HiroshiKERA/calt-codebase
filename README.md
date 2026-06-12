# calt-codebase

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://maximefaure.github.io/CALT/)
[![Built on calt-x](https://img.shields.io/badge/built%20on-calt--x-blue)](https://github.com/HiroshiKERA/calt)
[![Paper](https://img.shields.io/badge/arXiv-2506.08600-b31b1b)](https://arxiv.org/abs/2506.08600)

> 📖 **[Read the full documentation online](https://maximefaure.github.io/CALT/)**

Experiments for learning **algebraic computations** with Transformer models,
built on top of the [CALT library](https://github.com/HiroshiKERA/calt)
(`pip install calt-x`).

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

## Quick start

First, activate the conda environment that has `calt-x` installed:

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

```bash
pip install calt-x omegaconf matplotlib click   # or: conda install calt-x
```

- Python `>=3.10,<3.13`
- `calt-x` is on both [PyPI](https://pypi.org/project/calt-x/) and
  [conda-forge](https://github.com/conda-forge/calt-x-feedstock). It provides the
  four pipelines (`DatasetPipeline`, `IOPipeline`, `ModelPipeline`,
  `TrainerPipeline`) and pulls in PyTorch, HuggingFace Transformers and (for
  polynomial tasks) SageMath.
- `wandb` is optional — used by `shared/logging.py`; set `no_wandb: true` in a
  `train.yaml` (or `WANDB_MODE=disabled`) to skip it.

---

## Links

- CALT library: <https://github.com/HiroshiKERA/calt> · docs: <https://hiroshikera.github.io/calt/>
- Paper: *CALT: A Library for Computer Algebra with Transformer*, Kera et al. — [arXiv:2506.08600](https://arxiv.org/abs/2506.08600)
- Border basis algorithm: *Computational Algebra with Attention*, Kera et al. — [arXiv:2505.23696](https://arxiv.org/abs/2505.23696)
