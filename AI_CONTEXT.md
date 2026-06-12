# calt-codebase — AI Context Document

> This document is for AI assistants answering questions about **this repository**
> (`calt-codebase`, an experiment repo built on the `calt-x` library). It contains
> the real directory tree, the house API (`shared/`), the per-task `core/`
> contract, the exact way `calt-x` is used here, the config schemas, and known
> gotchas. Read this file to answer "how does this code work" questions.
>
> For a beginner-friendly walkthrough see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## 0. Quick facts

| Property | Value |
|----------|-------|
| Repo role | Experiments for learning algebraic computations with Transformers |
| Built on | `calt-x` library — provides the 4 pipelines |
| calt-x install | pip: `pip install calt-x` · conda: `conda install calt-x` (conda-forge) |
| conda-forge feedstock | <https://github.com/conda-forge/calt-x-feedstock> (PyPI release → bot PR → merge if deps unchanged) |
| Package name | `calt-experiments` (see `pyproject.toml`), version `0.1.0` |
| Python | `>=3.10,<3.13` |
| Declared deps | `calt-x>=0.1.0`, `omegaconf>=2.3.0`, `matplotlib>=3.7.0`, `click>=8.0.0` |
| Optional dep | `wandb` (used by `shared/logging.py`; disable via `no_wandb`/`WANDB_MODE=disabled`) |
| Heavy transitive deps | PyTorch, HuggingFace Transformers, **SageMath** (polynomial tasks) |
| Tasks | `parity`, `groebner_basis`, `border_basis` |
| calt-x library source | <https://github.com/HiroshiKERA/calt> |
| Papers | CALT: arXiv:2506.08600 · Border basis: arXiv:2505.23696 |
| Experiments mirror | `issac2026_experiments/*` on `HiroshiKERA/calt@experiment/issac2026` |

**Organizing principle**: each top-level directory is **one task** (one math
problem). A task has `core/` (reusable logic) and `experiments/<name>/` (configs +
scripts). Cross-task utilities live in `shared/`.

> Note for older docs: this repo does **not** use the "C/E" (Coefficient/Exponent)
> token format, an `examples/` or `src/` directory, a `CustomTrainer` file, an
> `environment.yml`, or Kaggle remote execution. Those belonged to a previous
> layout. Tokenization here is the polynomial's printed string + per-digit numbers.

---

## 1. Directory tree (real)

```
calt-codebase/
├── pyproject.toml              # name=calt-experiments, ruff line-length=100
├── README.md
├── DOCUMENTATION.md
├── AI_CONTEXT.md
├── .gitignore                  # ignores **/outputs/ **/data/ **/results/ **/wandb/
│
├── shared/                     # utilities imported by every task
│   ├── __init__.py             # exports set_seed, load_config, paths.*, plotting.*
│   ├── seed.py                 # set_seed()
│   ├── config.py               # load_config(), save_config()
│   ├── paths.py                # experiment_dir/data_dir/output_dir/config_dir/codebase_root
│   ├── calt_adapter.py         # re-exports 4 pipelines + run_standard_training()
│   ├── logging.py              # CustomLoggingCallback
│   ├── plotting.py             # load_eval_results, show_examples, showcase, plot_success_rate
│   └── README.md
│
├── parity/                     # task: sign of a permutation
│   ├── core/{generator,formatter,parser,metrics,train}.py
│   ├── experiments/toy/{configs,scripts}
│   ├── experiments/scaling/{configs,scripts,data_n5,data_n7,data_n10}
│   ├── experiments/ablation/{configs,scripts}
│   └── notebooks/analysis.ipynb
│
├── groebner_basis/             # task: Gröbner basis of ⟨f1,f2⟩
│   ├── core/{generator,formatter,parser,metrics,train}.py
│   ├── experiments/{toy,scaling,ablation}/{configs,scripts}
│   └── notebooks/analysis.ipynb
│
├── border_basis/               # task: border basis of a 0-dim ideal
│   ├── core/{generator,formatter,parser,metrics,train,algorithm}.py   # +algorithm.py
│   ├── experiments/{toy,finite_field,ablation}/{configs,scripts}
│   └── notebooks/analysis.ipynb
│
└── templates/task_template/    # scaffold for a new task (same layout, TODOs)
```

`experiments/scaling/data_n*/` contain committed `train_raw.txt`/`test_raw.txt`
and `*_stats.yaml` for the parity scaling runs; most other `data/`, `outputs/`,
`results/` dirs are git-ignored.

---

## 2. The per-task `core/` contract

Every task exposes the same five modules. Signatures below are the real ones.

### 2.1 `core/generator.py` — `class …Generator` with `__call__(seed)`

A **callable class**, not a function. Constructed with task params, then passed to
`DatasetPipeline(instance_generator=...)`, which calls it once per seed.

```python
class GroebnerGenerator:
    def __init__(self, sampler: PolynomialSampler, num_polynomials: int = 2): ...
    def __call__(self, seed: int) -> tuple[list, list]:
        randstate.set_random_seed(seed)
        ...
        return F, G          # (problem, answer); lists are joined with ' | ' by CALT
```

```python
class ParityGenerator:
    def __init__(self, n: int = 5): ...
    def __call__(self, seed: int) -> tuple[str, str]:
        random.seed(seed)
        ...
        return "2 0 1", "+1"   # plain strings
```

```python
class BorderBasisGenerator:
    def __init__(self, sampler, num_polynomials=3,
                 use_fast_elimination=True, lstabilization_only=False): ...
    def __call__(self, seed: int) -> tuple[list, list]: ...
```

**Contract**: one arg `seed: int`; deterministic; returns `(problem, answer)`.
Each side may be `str`, a `list` (CALT joins via `str()` + `' | '`), or a SageMath
object (`str()` called automatically). Generators retry up to 100 seeds to skip
degenerate samples (large-rational rejection for QQ Gröbner; non-0-dim ideals for
border basis).

### 2.2 `core/formatter.py`

`format_input(problem) -> str` and `format_target(answer) -> str`. Used for
tests/REPL/parsers — **not** by `DatasetPipeline` (which `str()`s objects itself).
SEP is `" | "`.

### 2.3 `core/parser.py` — optional load-time preprocessor

For plain-text tasks (`parity`, default `groebner`/`border`) this is empty/reserved.
The one real implementation is the Gröbner lex variant:

```python
class GroebnerLexOrderPreprocessor:
    def __init__(self, ring_src, delimiter: str = "|"): ...
    def process_sample(self, source: dict[str, Any]) -> tuple[str, str]:
        # rebuild F in a lex ring, recompute the lex Gröbner basis with
        # I_lex.groebner_basis("libsingular:std"); return (input_text, target_text)
```

A load preprocessor must implement `process_sample(source) -> (input_text, target_text)`
and is attached via `io_pipeline.dataset_load_preprocessor = <obj>`.

### 2.4 `core/metrics.py`

```python
def success_rate(predictions: list[str], targets: list[str]) -> float:
    # exact match after .strip(); fraction correct
```

The template also defines `instance_stats(problem, answer) -> dict` (per-sample
stats for `DatasetPipeline`'s `statistics_calculator`); the concrete tasks rely on
CALT's default stats instead.

### 2.5 `core/train.py` — `run_training(...)`

The function the `scripts/train.py` entry point calls. Drives the 4 CALT pipelines.

```python
# parity / border (simple text):
def run_training(cfg: DictConfig, dryrun: bool = False) -> float

# groebner (supports the lex variant):
def run_training(cfg, data_cfg=None, training_order="degrevlex", dryrun=False) -> float
```

Body (all tasks share this shape):

```python
if dryrun: apply_dryrun_settings(cfg)
save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
os.makedirs(save_dir, exist_ok=True)
OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

io_pipeline = IOPipeline.from_config(cfg.data)
# (groebner lex only) io_pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(...)
io_dict = io_pipeline.build()
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
trainer_pipeline.train()
trainer_pipeline.save_model()
return trainer_pipeline.evaluate_and_save_generation()   # float success rate
```

Gröbner's `run_training` additionally suffixes `save_dir` and `wandb.name` with
`_{training_order}` and, for `lex`, builds the source ring from `data_cfg.sampler`
and wires `ChainLoadPreprocessor(TextToSageLoadPreprocessor, GroebnerLexOrderPreprocessor)`.

---

## 3. `shared/` — house API (exact)

### 3.1 `shared/seed.py`

```python
def set_seed(seed: int = 42) -> None
# sets PYTHONHASHSEED, random, numpy, torch (+cuda), cudnn deterministic,
# transformers.set_seed, CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True)
```

### 3.2 `shared/config.py`

```python
def load_config(path, overrides: list[str] | None = None) -> DictConfig
# overrides are OmegaConf dotlist, e.g. ["train.batch_size=32"]
def save_config(cfg: DictConfig, path) -> None
```

### 3.3 `shared/paths.py`

All take `__file__` of a script at `<task>/experiments/<exp>/scripts/<file>.py`.

```python
experiment_dir(f) -> Path   # <task>/experiments/<exp>/
data_dir(f)       -> .../data
output_dir(f)     -> .../outputs
config_dir(f)     -> .../configs
codebase_root(f)  -> Path(f).resolve().parents[4]   # repo root
```

### 3.4 `shared/calt_adapter.py`

Re-exports `DatasetPipeline`, `IOPipeline`, `ModelPipeline`, `TrainerPipeline`,
`apply_dryrun_settings`. Plus:

```python
def run_standard_training(cfg, load_preprocessor=None, dryrun=False) -> float
# one-call train→eval for tasks needing no custom Trainer; mirrors core/train.py.
```

### 3.5 `shared/logging.py`

```python
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, **kwargs)
    # logs train/avg_param_norm, train/gpu_memory_used_MB, _reserved_MB to wandb
# attach: trainer_pipeline.trainer.add_callback(CustomLoggingCallback())
```

### 3.6 `shared/plotting.py`

```python
def load_eval_results(results_dir) -> tuple[list[str], list[str]]
# resolves <dir>/eval_results.json, else latest eval_results/step_*.json;
# delegates to calt.io.visualization.comparison_vis.load_eval_results

def show_examples(generated, references, n=5, successes=True) -> None
# prints up to n matching (successes=True) or mismatching cases — used by every
# experiments/*/scripts/evaluate.py

def showcase(dataset, success_cases=True, num_show=5, eval_results_path=None,
             results_dir="results") -> None
# decodes via dataset.preprocessor.decode(); port of the official demo helper

def plot_success_rate(results_dir, ax=None) -> plt.Axes
# success rate vs step from eval_results/step_*.json
```

`shared/__init__.py` exports: `set_seed, load_config, experiment_dir, data_dir,
output_dir, config_dir, showcase, show_examples, load_eval_results, plot_success_rate`.

> Historical bug (now fixed): `evaluate.py` files import `show_examples`, which
> previously did not exist in `plotting.py` (only `showcase`). `show_examples` is
> now defined and exported.

---

## 4. How `calt-x` is used here (touchpoints)

This repo treats `calt-x` as a black box behind four pipelines. The exact imports
and call sites used:

```python
# dataset
from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler

sampler = PolynomialSampler(**sampler_cfg)        # see §6 for sampler_cfg keys
sampler.get_ring()                                 # → SageMath PolynomialRing
sampler.sample(num_samples=k)                      # → list of polynomials
DatasetPipeline.from_config(cfg.dataset, instance_generator=generator).run()

# io
from calt.io import IOPipeline, ChainLoadPreprocessor, TextToSageLoadPreprocessor
io = IOPipeline.from_config(cfg.data)
io.dataset_load_preprocessor = <obj with process_sample>   # optional
io_dict = io.build()        # keys: tokenizer, train_dataset, test_dataset, data_collator

# models
from calt.models import ModelPipeline
model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()   # BART enc-dec

# trainer
from calt.trainer import TrainerPipeline, apply_dryrun_settings
tp = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
tp.train(); tp.save_model()
rate = tp.evaluate_and_save_generation()           # float exact-match rate
tp.trainer                                          # underlying HF Trainer (add_callback, …)

# eval result loading
from calt.io.visualization.comparison_vis import load_eval_results
```

`apply_dryrun_settings(cfg)` mutates the config in place to reduce epochs/data.
Exact internal signatures of these classes live in the installed `calt` package —
verify against `pip show calt-x` / the library source rather than assuming, as the
version is only pinned `>=0.1.0`.

---

## 5. `lexer.yaml` schema (real)

```yaml
vocab:
  range:
    numbers: ["", min, max]    # [prefix, min, max] → number tokens; prefix usually ""
    # extra named ranges allowed, e.g. signed: ["", -1, -1]
  misc: ["+", "-", "*", "^", "x", "y", "|", "/"]   # literal operator/variable tokens
  special_tokens: {}
  flags:
    include_base_vocab: true
    include_base_special_tokens: true    # adds <s>, </s>, [PAD], [UNK], [CLS], [SEP]

number_policy:
  attach_sign: true     # "-3" one token vs "- 3" two tokens
  digit_group: 1        # 1 = each digit its own token; 0 = whole numbers
  allow_float: false

strict: true            # raise on unknown token
include_base_vocab: true
```

**Polynomial tasks** (`groebner`, `border`): `digit_group: 1`, misc includes
`x y ^ * + - |` (and `/` for QQ). Digit-level tokenization → any integer size.
**Parity**: `digit_group: 0`, `numbers: ["", 0, 20]`, misc `["+1"]`,
`signed: ["", -1, -1]`, `strict: false`.

---

## 6. Config schemas (real)

### `data.yaml`

```yaml
sampler:                 # polynomial tasks only (forwarded to PolynomialSampler)
  symbols: "x,y"
  field_str: "QQ"        # QQ | ZZ | RR | GF<p> (e.g. GF7, GF11)
  order: "degrevlex"
  max_num_terms: 5
  max_degree: 4
  min_degree: 1
  max_coeff: 5
  degree_sampling: "uniform"
  term_sampling: "uniform"

problem_generator:       # forwarded to the task's generator class
  num_polynomials: 2     # parity uses: n: 7

dataset:                 # forwarded to DatasetPipeline
  save_dir: "../data/QQ"
  num_train_samples: 1000
  num_test_samples: 200
  batch_size: 100
  n_jobs: 4
  root_seed: 42
  verbose: true
  backend: "sagemath"    # "sagemath" | "sympy" (parity uses sympy)
  save_text: true
  save_json: false
```

### `train.yaml`

```yaml
model:
  model_type: generic
  num_encoder_layers: 3
  num_encoder_heads: 4
  num_decoder_layers: 3
  num_decoder_heads: 4
  d_model: 256
  encoder_ffn_dim: 1024
  decoder_ffn_dim: 1024
  max_sequence_length: 2048      # polynomial tasks need long sequences

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
  wandb: {project: calt-experiments, group: groebner_basis, name: toy_QQ, no_wandb: false}

data:
  train_dataset_path: ../data/QQ/train_raw.txt
  test_dataset_path: ../data/QQ/test_raw.txt
  lexer_config: ../configs/lexer.yaml
  validate_train_tokens: false
  validate_test_tokens: false
```

All paths are relative to the `scripts/` directory (the run location).

---

## 7. Data format and on-disk outputs

**Dataset text** (`train_raw.txt` / `test_raw.txt`): one example per line,

```
<input> # <output>
```

- `#` separates input from output (with surrounding spaces: `" # "`).
- `|` separates list elements within a side (` | `).
- Examples:
  - parity: `0 2 3 1 4 # +1`
  - groebner: `f1 | f2 # g1 | g2 | … | gk`

**Generation stats**: `train_stats.yaml` / `test_stats.yaml` (YAML, not JSON):
`total_time`, `num_samples`, `samples_per_second`, `generation_time:{mean,std,min,max}`.

**Training outputs** (`<save_dir>/`): `model.safetensors`, `tokenizer.json`,
`train.yaml` (snapshot of the config used), `eval_results.json`
(`[{"generated": ..., "reference": ...}, …]`), and `eval_results/step_*.json`.

---

## 8. The three tasks (specifics)

| | parity | groebner_basis | border_basis |
|---|---|---|---|
| Input | one-line perm `2 0 1` | `f1 \| f2` over k[x,y] | `g1 \| g2 \| g3` over GF(p) |
| Output | `+1` / `-1` | Gröbner basis `g1 \| …` | border basis |
| Backend | sympy | sagemath | sagemath |
| Storage | plain text | plain text | plain text |
| Preprocessor | none | none (lex variant: Chain) | none |
| Generator class | `ParityGenerator(n)` | `GroebnerGenerator(sampler, num_polynomials)` | `BorderBasisGenerator(sampler, num_polynomials, …)` |
| Experiments | toy, scaling (n∈{5,7,10}) | toy, scaling, ablation (degrevlex/lex) | toy, finite_field (GF 5/7/11/17/31), ablation |
| Algorithm | inversions count | SageMath `ideal.groebner_basis()` | `core/algorithm.py::BorderBasisCalculator` (port of arXiv:2505.23696 Alg 4.1) |

- **parity**: `sign(σ) = (−1)^(#inversions)`; `count_inversions`, `permutation_parity`
  helpers in `core/generator.py`.
- **groebner**: QQ samples with ≥3-digit numerator/denominator rejected
  (`_has_large_rational_coefficients`, threshold 100). `--training_order lex`
  recomputes the basis in a lex ring at load time.
- **border**: requires a 0-dimensional ideal (`ideal.dimension() == 0`), retried up
  to 100 seeds; `BorderBasisCalculator` does L-stable span → optimal order ideal via
  `scipy.optimize.milp` → basis transformation; uses `sortedcontainers.SortedList`.

The `scripts/train.py` of each experiment is a thin `click` wrapper:
`parity/scaling` exposes `--n` (rewrites data paths + save_dir + wandb name);
`groebner/toy` exposes `--training_order {degrevlex,lex}`, `--data_config_path`,
`--config_path`, `--dryrun`.

---

## 9. `templates/task_template/`

Same layout as a real task, with `TODO`s. The runnable, consistent pattern:

- `core/generator.py::TaskGenerator(**params)` with `__call__(seed)` (raises until filled).
- `core/parser.py::TaskParser.process_sample(source)` — only if storing pickle/JSON.
- `core/train.py::run_training(cfg, dryrun=False)` — sets `dataset_load_preprocessor = TaskParser()`
  (comment says remove for plain text).
- `experiments/toy/scripts/generate.py` — builds `TaskGenerator(**cfg.problem_generator)`
  and runs `DatasetPipeline.from_config(cfg.dataset, instance_generator=…)`.
- `experiments/toy/configs/data.yaml` — has a `problem_generator: {}` block + `dataset:`.
- README has the fill-in checklist.

To create a task: `cp -r templates/task_template my_new_task`, fill the `core/`
modules, edit the three configs, then `generate.py → train.py --dryrun → evaluate.py`.

---

## 10. Known issues, gotchas, and conventions

- **Run scripts from their `scripts/` dir.** Config paths are relative to it, and
  `sys.path.insert(0, parents[4])` makes the repo root importable. Running from
  elsewhere breaks both.
- **`[UNK]` / token validation**: a symbol in the data isn't in `lexer.yaml`. Widen
  `numbers` or add to `misc`; set `validate_train_tokens: true` to fail fast.
- **`max_sequence_length`**: polynomial bases get long; toy configs use 2048.
  Too small ⇒ truncation ⇒ stuck loss.
- **Determinism**: `set_seed` enables deterministic algorithms + sets
  `CUBLAS_WORKSPACE_CONFIG`. Same `root_seed` ⇒ identical datasets.
- **SageMath RNG is global**: generators call `randstate.set_random_seed(seed)` before sampling.
- **wandb**: optional; `no_wandb: true` or `WANDB_MODE=disabled` to skip.
- **No `src/`, no `examples/`, no `CustomTrainer.py`, no `environment.yml`, no
  Kaggle remote** in this repo (unlike the previous CALT layout). Custom training
  hooks go through `shared/logging.py` or by subclassing `calt.trainer.trainer.Trainer`.
- **Internal cross-references** in code/docstrings to `issac2026_experiments/*`
  point at the upstream `HiroshiKERA/calt@experiment/issac2026` branch the tasks
  were ported from — not files in this repo.

---

## 11. End-to-end data flow

```
1. GENERATE
   scripts/generate.py
     load data.yaml → set_seed(root_seed)
     (poly tasks) PolynomialSampler(**sampler) ; <Task>Generator(sampler, **problem_generator)
     DatasetPipeline.from_config(cfg.dataset, instance_generator=gen).run()
       for seed in 0..N-1 (parallel, n_jobs): problem,answer = gen(seed)
       serialize → "<input> # <output>" → train_raw.txt / test_raw.txt (+ *_stats.yaml)

2. TRAIN
   scripts/train.py → core/train.py::run_training(cfg, …)
     IOPipeline.from_config(cfg.data)[.dataset_load_preprocessor = …].build() → io_dict
     ModelPipeline.from_io_dict(cfg.model, io_dict).build()                   → BART model
     TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()
     .train() → .save_model() → .evaluate_and_save_generation() → float rate
     writes <save_dir>/{model.safetensors, tokenizer.json, train.yaml, eval_results*.json}

3. EVALUATE
   scripts/evaluate.py
     load_eval_results(output_dir/"results")  → (generated, references)
     print exact-match rate ; show_examples(... successes=True/False)
```
