# CALT codebase — Complete Documentation

> A full guide for anyone who wants to understand, use, or extend this repository —
> **no prior knowledge of transformers or deep learning required.**
>
> This document describes **calt-codebase** (the experiment repository in this folder).
> The underlying library it builds on is **calt-x** (`pip install calt-x`); for a
> source-level reference aimed at AI assistants, see [AI_CONTEXT.md](AI_CONTEXT.md).

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [Two layers: calt-x and calt-codebase](#2-two-layers-calt-x-and-calt-codebase)
3. [Installation](#3-installation)
4. [Core concepts](#4-core-concepts)
5. [Repository structure](#5-repository-structure)
6. [The 3-step workflow](#6-the-3-step-workflow)
7. [Configuration files](#7-configuration-files)
8. [The three tasks](#8-the-three-tasks)
9. [Creating your own task](#9-creating-your-own-task)
10. [The `shared/` utilities](#10-the-shared-utilities)
11. [Advanced usage](#11-advanced-usage)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)

---

## 1. What is this project?

This repository trains Transformer models to solve **symbolic mathematics**
problems. It explores one central question:

> Can a neural network learn to perform an algebraic computation just from examples?

Some computer-algebra algorithms (Gröbner bases, border bases) are extremely
expensive — doubly exponential in the number of variables in the worst case. The
idea here is to **learn the input → output mapping from data**, the same way a
language model learns translation.

Every task is framed as a **translation task**:

```
Input  : f1 | f2          (a problem, written as text)
Output : g1 | … | gk       (its answer, written as text)
```

The model sees thousands of such pairs and learns the rule that maps one side to
the other — without that rule being programmed explicitly.

### Example tasks in this repo

| Input | Output | Task |
|-------|--------|------|
| `2 0 1` | `+1` | Parity of a permutation |
| `x^2 + 2*x \| 3*x - 1` | `1 \| x^2 - 1` | Gröbner basis of ⟨f1, f2⟩ |
| `g1 \| g2 \| g3` (over GF(7)) | border basis | Border basis of a 0-dim ideal |

---

## 2. Two layers: calt-x and calt-codebase

```
┌─────────────────────────────────────────────────────────┐
│                    calt-codebase                          │
│  (THIS REPO — experiment template)                        │
│                                                           │
│  • parity / groebner_basis / border_basis tasks           │
│  • shared/ utilities (seeds, configs, paths, plotting)    │
│  • templates/ to scaffold new tasks                       │
└──────────────────────────┬────────────────────────────────┘
                           │ uses (pip install calt-x)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                       calt-x                              │
│  (Python library — the engine)                            │
│                                                           │
│  • DatasetPipeline  → generate (problem, answer) pairs    │
│  • IOPipeline       → load, preprocess, tokenize          │
│  • ModelPipeline    → build the Transformer (BART-based)  │
│  • TrainerPipeline  → train + evaluate                    │
└─────────────────────────────────────────────────────────┘
```

**Analogy**: `calt-x` is the engine; `calt-codebase` is the vehicle built around
it. You write task-specific math in `calt-codebase`; `calt-x` does the
data-generation, tokenization, training and evaluation heavy lifting.

---

## 3. Installation

`calt-x` is available on both **PyPI** and **conda-forge**:

```bash
# via pip
pip install calt-x omegaconf matplotlib click

# or via conda-forge
conda install calt-x
```

- Python `>=3.10,<3.13`.
- `calt-x` pulls in PyTorch, HuggingFace Transformers, and (for the polynomial
  tasks) **SageMath**. SageMath is large; the easiest way to get it is a conda
  environment with a SageMath build, then install `calt-x` inside it.
- A GPU is strongly recommended for training (CPU works but is very slow).
- `wandb` is optional (used for live training curves). Disable it with
  `no_wandb: true` in `train.yaml` or `WANDB_MODE=disabled`.

> **conda-forge packaging** — `calt-x` is maintained on conda-forge at
> [conda-forge/calt-x-feedstock](https://github.com/conda-forge/calt-x-feedstock).
> When a new version is released on PyPI, a bot automatically opens a PR on the
> feedstock; merging it is enough when dependencies are unchanged. If dependencies
> change (or a CI build fails), the feedstock maintainer updates `meta.yaml` by hand.

### Recommended: use a dedicated conda environment

The safest setup is a dedicated environment named `calt-env` that bundles
SageMath, PyTorch, and `calt-x` together:

```bash
conda create -n calt-env python=3.12
conda activate calt-env
conda install calt-x omegaconf matplotlib click
```

> **Important** — always activate this environment before running any script in
> this repo, otherwise Python will not find `calt-x` and you will get a
> `ModuleNotFoundError`:
>
> ```bash
> conda activate calt-env
> ```

Verify:

```bash
conda activate calt-env
python -c "import calt; print('calt-x OK')"
cd groebner_basis/experiments/toy/scripts
python generate.py            # generate a small dataset
python train.py --dryrun      # quick smoke test (reduced epochs)
```

---

## 4. Core concepts

*No AI background required.*

### 4.1 The Transformer: a universal translator

A **Transformer** converts one sequence of symbols into another. It has two parts:

```
Input (problem) → ENCODER → DECODER → Output (solution)
```

The decoder generates the output **one token at a time**, using what it has
already produced. This step-by-step generation is what lets the model "reason"
through multi-step computations.

### 4.2 Training: learning from examples

The model starts random. We show it examples, measure how wrong it is (the
**loss**), nudge its parameters to reduce the error, and repeat. This is
**gradient descent**. In these experiments: thousands to 100k examples, tens of
epochs, the AdamW optimizer.

### 4.3 Tokenization: turning math into machine-readable text

The Transformer sees **lists of integers**, not math. Every expression is first
turned into a sequence of **tokens** (atomic symbols), and every token must be
declared in `lexer.yaml`.

This repo uses the **native string form** of the objects (what SageMath prints),
tokenized with a small vocabulary:

```
Polynomial:  3*x^2*y - 2*x + 1
Tokens:      3 * x ^ 2 * y - 2 * x + 1      (with numbers split per digit)
```

Two tokenization styles are used, depending on the task:

- **Polynomial tasks** (`groebner_basis`, `border_basis`): vocabulary is
  `numbers` + the operators `+ - * ^ x y | /`, with `digit_group: 1`
  (**each digit is its own token**). Splitting numbers digit-by-digit means the
  model can represent arbitrarily large coefficients with just the symbols 0–9.
- **Parity**: permutation entries are whole-number tokens (`numbers: ["", 0, 20]`,
  `digit_group: 0`), and the two answers `+1` / `-1` are single tokens.

> **Note**: this is *not* the older "C/E" (Coefficient/Exponent) encoding. Tasks
> here feed the polynomial's printed form directly and rely on per-digit
> tokenization.

### 4.4 Evaluation metric: exact match

In symbolic math a partially-correct answer is useless: `1 | x^2 - 1` vs
`1 | x^2 + 1` is simply wrong. So the metric is the **exact-match success rate** —
the fraction of predictions that are character-for-character identical to the
expected answer (after stripping whitespace). See `core/metrics.py::success_rate`
in any task.

---

## 5. Repository structure

```
calt-codebase/
├── shared/             # utilities imported by every task
│   ├── seed.py         # set_seed()  — fix Python/NumPy/PyTorch/HF seeds
│   ├── config.py       # load_config(), save_config()
│   ├── paths.py        # data_dir(), config_dir(), output_dir() …
│   ├── calt_adapter.py # re-export the 4 CALT pipelines + run_standard_training()
│   ├── logging.py      # CustomLoggingCallback (grad norms, GPU memory → wandb)
│   └── plotting.py     # load_eval_results(), show_examples(), plot_success_rate()
│
├── parity/             # task: sign of a permutation
├── groebner_basis/     # task: Gröbner basis of ⟨f1, f2⟩
├── border_basis/       # task: border basis of a 0-dim ideal
│
├── templates/
│   └── task_template/  # copy this to start a new task
│
├── pyproject.toml
└── README.md
```

Each task is **self-contained** and always follows this layout:

```
<task>/
├── README.md
├── core/
│   ├── generator.py    # class with __call__(seed) → (problem, answer)
│   ├── formatter.py    # object → string helpers (for tests/REPL)
│   ├── parser.py       # optional load-time preprocessor (often empty for text data)
│   ├── metrics.py      # instance_stats() + success_rate()
│   └── train.py        # run_training(cfg, …) — drives the CALT pipelines
└── experiments/
    └── <name>/         # toy, scaling, ablation, finite_field, …
        ├── configs/    # data.yaml, lexer.yaml, train.yaml
        └── scripts/    # generate.py, train.py, evaluate.py, run.sh
```

The `core/` modules hold reusable logic; the `scripts/` are thin entry points
that load a config, set the seed, and call into `core/`.

---

## 6. The 3-step workflow

Using `groebner_basis/experiments/toy` as the example.

> **Before anything**: make sure the `calt-env` conda environment is active.
> ```bash
> conda activate calt-env
> ```

### Step 1 — Generate the data

```bash
cd groebner_basis/experiments/toy/scripts
python generate.py
```

Internally `generate.py`:
1. loads `../configs/data.yaml`,
2. builds a `PolynomialSampler` and a `GroebnerGenerator(sampler, …)`,
3. hands the generator to `DatasetPipeline`, which calls it for seeds
   `0 … N-1` **in parallel** and writes:

```
../data/QQ/train_raw.txt    # "f1 | f2 # g1 | … | gk"  per line
../data/QQ/test_raw.txt
../data/QQ/train_stats.yaml  # aggregate generation statistics
../data/QQ/test_stats.yaml
```

**File format**: one example per line, `input # output`. `|` separates list
elements; `#` separates input from output.

### Step 2 — Train the model

```bash
python train.py                       # full run
python train.py --dryrun              # quick smoke test
python train.py --training_order lex  # Gröbner-specific variant (see §11)
```

Internally `train.py` → `core/train.py::run_training`:

```
train_raw.txt
   │  IOPipeline.from_config(cfg.data).build()
   │     read lines → split on " # " → tokenize with the lexer
   ▼  ModelPipeline.from_io_dict(cfg.model, io_dict).build()
   │     build a BART encoder-decoder from cfg.model
   ▼  TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
   │     train (cross-entropy + AdamW), evaluate periodically, log to wandb
   ▼
../outputs/results_degrevlex/
   ├── model.safetensors
   ├── tokenizer.json
   ├── train.yaml             # snapshot of the config actually used
   ├── eval_results.json      # final predictions vs references
   └── eval_results/step_*.json
```

### Step 3 — Evaluate

```bash
python evaluate.py
```

`evaluate.py` loads `eval_results.json` via `shared.plotting.load_eval_results`,
prints the exact-match success rate, and shows a few correct and incorrect
predictions with `show_examples`.

---

## 7. Configuration files

Each experiment has three YAML files in `configs/`.

### 7.1 `data.yaml` — dataset generation

```yaml
# Parameters forwarded to PolynomialSampler (polynomial tasks only)
sampler:
  symbols: "x,y"
  field_str: "QQ"          # "QQ", "ZZ", "GF7", "GF11", …
  order: "degrevlex"
  max_num_terms: 5
  max_degree: 4
  min_degree: 1
  max_coeff: 5
  degree_sampling: "uniform"
  term_sampling: "uniform"

# Parameters forwarded to the task's generator class
problem_generator:
  num_polynomials: 2

# Parameters consumed by DatasetPipeline
dataset:
  save_dir: "../data/QQ"
  num_train_samples: 1000
  num_test_samples: 200
  batch_size: 100
  n_jobs: 4                 # CPU cores for parallel generation
  root_seed: 42             # same seed ⇒ same data
  verbose: true
  backend: "sagemath"       # "sagemath" or "sympy"
  save_text: true           # write *_raw.txt (required for training)
  save_json: false
```

The `parity` task has no `sampler:` block (it needs no polynomials) — just
`problem_generator: {n: …}` and `dataset:`.

### 7.2 `lexer.yaml` — tokenizer vocabulary

Polynomial task example (`groebner_basis`, `border_basis`):

```yaml
vocab:
  range:
    numbers: ["", -100, 100]   # [prefix, min, max] → seeds number tokens
  misc: ["+", "-", "*", "^", "x", "y", "|", "/"]
  special_tokens: {}
  flags:
    include_base_vocab: true
    include_base_special_tokens: true   # <s>, </s>, [PAD], [UNK], …

number_policy:
  attach_sign: true     # "-3" as one signed token rather than "- 3"
  digit_group: 1        # tokenize each digit separately → any coefficient size
  allow_float: false

strict: true            # raise on an unknown token (recommended)
include_base_vocab: true
```

Parity example:

```yaml
vocab:
  range:
    numbers: ["", 0, 20]   # permutation entries 0..19 (n ≤ 20)
    signed:  ["", -1, -1]  # the target token "-1"
  misc: ["+1"]             # the target token "+1"
  flags:
    include_base_vocab: true
    include_base_special_tokens: true
number_policy:
  digit_group: 0           # whole-number tokens
strict: false
```

> **Critical rule**: the vocabulary must cover **every** symbol that appears in
> the data, *including the outputs*. With `digit_group: 1`, the digits 0–9 cover
> any integer; with `digit_group: 0` you must declare a wide enough `numbers`
> range. Set `validate_*_tokens: true` (see below) to catch gaps before training.

### 7.3 `train.yaml` — model + training + data paths

```yaml
model:
  model_type: generic        # BART encoder-decoder
  num_encoder_layers: 3
  num_encoder_heads: 4
  num_decoder_layers: 3
  num_decoder_heads: 4
  d_model: 256
  encoder_ffn_dim: 1024
  decoder_ffn_dim: 1024
  max_sequence_length: 2048

train:
  save_dir: ../outputs/results
  num_train_epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0
  warmup_ratio: 0.05
  batch_size: 16
  test_batch_size: 16
  lr_scheduler_type: linear
  max_grad_norm: 1.0
  optimizer: adamw_torch
  num_workers: 2
  seed: 42
  wandb:
    project: calt-experiments
    group: groebner_basis
    name: toy_QQ
    no_wandb: false          # true → disable wandb entirely

data:
  train_dataset_path: ../data/QQ/train_raw.txt
  test_dataset_path: ../data/QQ/test_raw.txt
  lexer_config: ../configs/lexer.yaml
  validate_train_tokens: false   # set true to verify vocab before training
  validate_test_tokens: false
```

Paths are written **relative to the `scripts/` directory** (that's where you run
the scripts from). The `shared.paths` helpers resolve them regardless of cwd.

---

## 8. The three tasks

### 8.1 `parity` — sign of a permutation ⭐ start here

- **Input**: one-line notation of a permutation σ, e.g. `2 0 1`.
- **Output**: `+1` (even) or `-1` (odd), where `sign(σ) = (−1)^(#inversions)`.
- **Storage**: plain text, no preprocessor, `sympy` backend (no SageMath needed).
- **Experiments**: `toy/` (n=5) and `scaling/` (n ∈ {5, 7, 10}, via `--n`).
- **Why**: parity is a *global* property — the model must compare all pairs, not
  just look at one element. A clean probe of attention over the full sequence.

### 8.2 `groebner_basis` — Gröbner basis of ⟨f1, f2⟩

- **Input**: two random polynomials `f1 | f2` over k[x, y], k ∈ {ℚ, GF(p), ℤ}.
- **Output**: the Gröbner basis `g1 | … | gk` (Buchberger's algorithm via SageMath).
- **Storage**: plain text. Over ℚ, samples with large rational coefficients are
  rejected to avoid coefficient swell.
- **Experiments**: `toy/`, `scaling/`, `ablation/` (degrevlex vs. lex).
- **Variant**: `--training_order lex` recomputes the basis in lex order at load
  time (see §11).

### 8.3 `border_basis` — border basis of a 0-dimensional ideal

- **Input**: generators `g1 | g2 | g3` over GF(p) (so the ideal is 0-dimensional).
- **Output**: the border basis, computed by a verbatim port of Algorithm 4.1
  (BBasis) from *Computational Algebra with Attention* (arXiv:2505.23696), in
  `core/algorithm.py` (`BorderBasisCalculator`).
- **Experiments**: `toy/` (GF(7)), `finite_field/` (GF(5/7/11/17/31)), `ablation/`.

---

## 9. Creating your own task

```bash
cp -r templates/task_template my_new_task
cd my_new_task
```

Then work through the checklist in `templates/task_template/README.md`:

1. **`core/generator.py`** — implement `TaskGenerator.__call__(seed) → (problem, answer)`.
   - Always seed your RNG inside `__call__` (e.g. `random.seed(seed)`, and for
     SageMath `sage.misc.randstate.set_random_seed(seed)`).
   - `problem`/`answer` may be a string, a list (joined with ` | `), or a
     SageMath object (`str()` is called automatically).
2. **`core/formatter.py`** — define how objects become strings (handy for tests).
3. **`core/parser.py`** — only needed if you store data as pickle/JSON; plain
   text needs no parser.
4. **`core/metrics.py`** — `instance_stats(problem, answer)` and `success_rate`.
5. **`experiments/toy/configs/`** — set `data.yaml` (sizes, sampler/generator
   params), `lexer.yaml` (vocabulary for *your* symbols), `train.yaml` (model + paths).
6. **Run**: `python generate.py && python train.py --dryrun && python evaluate.py`.

**Generator contract**: one argument `seed: int`; deterministic given the seed;
returns `(problem, answer)`.

---

## 10. The `shared/` utilities

Import these instead of duplicating code (see `shared/README.md`):

| Import | What it does |
|---|---|
| `from shared.seed import set_seed` | Fix all RNG seeds (Python, NumPy, PyTorch, HF) + deterministic CuBLAS |
| `from shared.config import load_config, save_config` | Load YAML with optional `key=value` overrides |
| `from shared.paths import data_dir, config_dir, output_dir, experiment_dir` | Paths relative to a script file |
| `from shared.calt_adapter import DatasetPipeline, IOPipeline, ModelPipeline, TrainerPipeline, apply_dryrun_settings` | One import for all four pipelines |
| `from shared.calt_adapter import run_standard_training` | One-call train→eval for tasks needing no custom logic |
| `from shared.logging import CustomLoggingCallback` | Log grad norms + GPU memory to wandb |
| `from shared.plotting import load_eval_results, show_examples, plot_success_rate` | Inspect eval results in scripts/notebooks |

Every script begins by putting the repo root on `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
```

(`parents[4]` because scripts live at `<task>/experiments/<exp>/scripts/<file>.py`.)

---

## 11. Advanced usage

### 11.1 Gröbner lex-order variant

```bash
python train.py --training_order lex
```

This wires a `ChainLoadPreprocessor(TextToSageLoadPreprocessor, GroebnerLexOrderPreprocessor)`
into the IOPipeline. At load time it parses `f1 | f2` back into SageMath
polynomials, moves them into a **lex**-ordered ring, and recomputes the lex
Gröbner basis (`libsingular:std`). Output dirs and wandb names get a `_lex`
suffix. Requires `data.yaml` (passed automatically) to reconstruct the source ring.

### 11.2 Extra training metrics

```python
from shared.logging import CustomLoggingCallback
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
trainer_pipeline.trainer.add_callback(CustomLoggingCallback())
```

Logs average parameter norm and GPU memory at each logging step.

### 11.3 Reproducibility

```python
from shared.seed import set_seed
set_seed(42)   # also sets CUBLAS_WORKSPACE_CONFIG and deterministic algorithms
```

Same `root_seed` in `data.yaml` ⇒ identical datasets.

### 11.4 Disabling wandb

```bash
WANDB_MODE=disabled python train.py
# or set  train.wandb.no_wandb: true  in train.yaml
```

### 11.5 A custom Trainer

There is no `src/custom_trainer.py` in this repo. For a custom loss or metric,
subclass `calt.trainer.trainer.Trainer`, then build the trainer pipeline and
swap in your class (see the CALT library docs / [AI_CONTEXT.md](AI_CONTEXT.md)).
The lightweight alternative — extra logging without changing the loss — is the
`CustomLoggingCallback` above.

---


## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'calt'`
`calt-x` isn't installed in the active environment: `pip install calt-x`
(inside the env that has SageMath, for polynomial tasks).

### `ModuleNotFoundError: No module named 'shared'` (or your task)
Run scripts from inside their `scripts/` directory, or ensure the
`sys.path.insert(... parents[4])` line is present. The repo root must be importable.

### `[UNK]` tokens / validation error
A symbol in the data isn't in `lexer.yaml`. Widen the `numbers` range or add the
missing operator to `misc`, and set `validate_train_tokens: true` to catch it early.

### `CUDA out of memory`
Lower `batch_size` in `train.yaml` (try 8, 4, 2). Note polynomial tasks use a
large `max_sequence_length` (2048).

### Loss not decreasing
Check `learning_rate` (try 1e-4…5e-4), confirm `max_sequence_length` isn't
truncating long bases, and inspect a few raw lines of `train_raw.txt`.

### `wandb` errors
`pip install --upgrade wandb` and `wandb login`, or just set
`WANDB_MODE=disabled` / `no_wandb: true`.

### Same seed gives the same data every run
Expected. Change `root_seed` in `data.yaml` for a different dataset.

---

## 13. Glossary

| Term | Plain-language definition |
|------|--------------------------|
| **Transformer** | Neural network that converts one sequence of symbols into another |
| **Token** | An atomic symbol in the model's vocabulary (`x`, `^`, `+`, `+1`, a digit) |
| **Tokenization** | Turning an expression into a list of tokens |
| **Vocabulary** | The full set of allowed tokens, declared in `lexer.yaml` |
| **`digit_group`** | How many digits form one number token (`1` = per digit) |
| **Loss** | A measure of the model's error; lower is better |
| **Epoch** | One full pass over the training data |
| **Batch** | A group of examples processed together |
| **Learning rate** | How fast the model updates its parameters |
| **AdamW** | The optimizer used to update parameters |
| **Exact match** | Is the prediction character-for-character identical to the answer? |
| **Success rate** | Fraction of test examples that match exactly |
| **Gröbner basis** | A canonical generating set of a polynomial ideal (Buchberger's algorithm) |
| **Border basis** | An order-free alternative to a Gröbner basis for 0-dimensional ideals |
| **Permutation parity** | The sign (+1 / −1) of a permutation |
| **Finite field GF(p)** | Arithmetic modulo a prime p |
| **SageMath** | Computer-algebra system used to compute the answers |
| **DatasetPipeline / IOPipeline / ModelPipeline / TrainerPipeline** | The four calt-x stages: generate, load, build, train |
| **Seed** | Reproducibility integer: same seed ⇒ same random data |

---

## Useful links

- This repo's README: [README.md](README.md)
- AI-assistant reference: [AI_CONTEXT.md](AI_CONTEXT.md)
- calt-x library: <https://github.com/HiroshiKERA/calt> · docs: <https://hiroshikera.github.io/calt/>
- Paper (CALT): [arXiv:2506.08600](https://arxiv.org/abs/2506.08600)
- Paper (border basis): [arXiv:2505.23696](https://arxiv.org/abs/2505.23696)
