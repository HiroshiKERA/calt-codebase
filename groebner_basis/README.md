# groebner_basis

**Task**: Given two polynomials f1, f2, predict the Gröbner basis of the ideal
⟨f1, f2⟩.

## The math

Let R = k[x, y] and I = ⟨f1, f2⟩ ⊂ R.

1. Sample f1, f2 ∈ R at random.
2. Compute G = the degrevlex Gröbner basis of I (SageMath).

The model receives (f1, f2) as input and must predict G, element by element.

**Why is this hard?** Computing G takes many reduction steps whose number and
size are not visible from the input, and over ℚ the coefficients of G can grow
far beyond those of f1, f2.

## Data

- Field: ℚ in the shipped configs (`field_str: "QQ"`); `core/generator.py` also
  supports GF(p) and ℤ through `field_str`
- Variables: x, y
- Polynomials (toy): 1–5 terms, degree 1–4; over ℚ the coefficients are
  fractions whose numerator and denominator are at most 10 in absolute value
  (`PolynomialSampler` uses `num_bound`, default 10, for ℚ; `max_coeff` in
  `data.yaml` applies to ℤ)
- Over ℚ, samples where a coefficient of F or G has a numerator or denominator
  of 100 or more are redrawn (up to 100 draws per seed)
- Stored as text: `../data/QQ/train_raw.txt` and `test_raw.txt`, one
  `f1 | f2 # g1 | … | gk` per line

## Experiments

| Experiment | What varies |
|---|---|
| `toy/` | Small model, 1000 train samples — verify the pipeline runs |
| `scaling/` | Larger models and datasets — measure scaling behavior |
| `ablation/` | Training order: degrevlex (the stored basis) vs. lex (recomputed at load time) |

## Quick start

```bash
cd groebner_basis/experiments/toy/scripts
python generate.py          # generate data  (~1 min)
python train.py             # train model    (~10 min on GPU)
python evaluate.py          # evaluate       (prints success rate)
```

Or run everything:

```bash
bash run.sh
```

## Offline preprocessing (optional, recommended for `lex` order)

`train.py` applies the `dataset_load_preprocessor` (TextToSage → lex Gröbner basis for lex,
TextToSage → ExpandedForm for C/E format) **every time it starts**. For large
datasets and the `lex` chain that means several minutes of SageMath
recomputation before each run.

`preprocess.py` runs that chain **once** offline and writes a JSONL cache. By
default it also pre-tokenizes all the way to `input_ids` so that `train.py`
performs **zero tokenization at runtime** — only padding + GPU forward.

```bash
cd groebner_basis/experiments/toy/scripts
python preprocess.py                          # cache (pretokenized) for degrevlex
python preprocess.py --training_order lex     # cache (pretokenized) for lex order
python preprocess.py --no-pretokenize         # cache stops at the preprocessed strings (legacy mode)
python train.py --training_order lex          # auto-detects cache → fast startup
python evaluate.py --training_order lex       # reads ../outputs/results_lex/
```

Two cache flavors live side-by-side; `train.py` prefers the pre-tokenized one:

| Cache directory | Contains | Skipped at training time |
|---|---|---|
| `processed_<order>_<format>_ids/` | `{"input_ids": [...], "target_ids": [...]}` | SageMath + UnifiedLexer + HF tokenizer |
| `processed_<order>_<format>/`     | `{"problem": str, "answer": str}`         | SageMath only (lexer/tokenizer still online) |
| (none)                            | raw `.txt`                                | nothing — legacy path |

The caches live next to your raw `.txt`:

```
data/QQ/
├── train_raw.txt
├── test_raw.txt
├── processed_lex_raw_ids/         ← pre-tokenized (default --pretokenize)
│   ├── train_ids.jsonl            ← {"input_ids":[...], "target_ids":[...]}
│   ├── test_ids.jsonl
│   ├── _hash.txt                  ← includes tokenizer vocab in the hash
│   └── _meta.yaml
└── processed_lex_raw/             ← strings cache (--no-pretokenize)
    ├── train_processed.jsonl
    ├── test_processed.jsonl
    ├── _hash.txt
    └── _meta.yaml
```

**When to re-run `preprocess.py`** — any change to `lexer.yaml`, `data.yaml`
sampler config, or the `--training_order` produces a new hash. `train.py`
detects the mismatch automatically and falls back to the slow path, printing
a `cache is STALE` warning. Re-run `preprocess.py` (or `--force`) to refresh.

**When NOT to bother** — for `--training_order degrevlex` with the default raw
lexer there is no load-time work to skip (the chain is empty), so the cache
just copies the raw text. Still safe to use, but the speedup is zero.

## User post-processing hooks (optional)

Two opt-in slots let you inject custom transformations of `(input_text, target_text)`
after the load chain (lex recomputation, expanded form, etc.) has run, before tokenization:

| Slot | File (create to enable) | Scope |
|---|---|---|
| Base hook   | `shared/base_postprocessor.py`         | Applied to every task (groebner, border, parity) |
| Per-task    | `groebner_basis/core/postprocessor.py` | Applied to groebner only, AFTER the base hook |

Each file must expose:

```python
def postprocess(input_text: str, target_text: str) -> tuple[str, str]:
    return input_text, target_text  # default: identity
```

To enable, copy the corresponding `.py.example` file:

```bash
cp shared/base_postprocessor.py.example shared/base_postprocessor.py
# then edit shared/base_postprocessor.py
```

The hook source code is folded into the preprocessing cache hash, so editing
`postprocess()` automatically invalidates stale caches — your next
`preprocess.py` run rebuilds them. If you delete the file the system reverts
to identity (no hook).

Examples and caveats are in [shared/base_postprocessor.py.example](../shared/base_postprocessor.py.example).

## Variants

- `--order lex` (run.sh) / `--training_order lex` (train.py, evaluate.py) :
  train on the lex Gröbner basis, recomputed from (f1, f2) at load time with
  SageMath's `libsingular:std` (see `core/parser.py`)

## Tokenization format (raw vs C/E expanded)

Two tokenization schemes are available. The training pipeline auto-detects which one
to use from the `data.lexer_config` field in your `train.yaml`.

| Format | Lexer config | How a term is encoded | Vocab size |
|---|---|---|---|
| **raw** (default) | `configs/lexer.yaml` | `"x ^ 2 + y"` → `["x", "^", "2", "+", "y"]` | larger (one token per digit) |
| **expanded** (C/E) | `configs/lexer_expanded.yaml` | `"2*x^2*y"` → `["C2", "E2", "E1"]` | smaller, fixed |

**Switching is one line**: in `experiments/<exp>/configs/train.yaml`, change

```yaml
data:
  lexer_config: ../configs/lexer.yaml           # raw  (default)
  # lexer_config: ../configs/lexer_expanded.yaml  # C/E expanded
```

That's it. `core/train.py` calls `detect_lexer_format()` on the file and wires
`ExpandedFormLoadPreprocessor` into the load chain automatically when needed.

**When to use which**:
- **raw** — default, matches ISSAC '26 §5 reproduction. Handles coefficient swell on ℚ
  via `digit_group: 1` (each digit is its own token). Use this to reproduce paper
  results.
- **expanded** — fixed-size vocab (only ranges `[Cmin, Cmax]` and `[Emin, Emax]`).
  Useful if you want a smaller embedding table or to align with the
  `polynomial_reduction` task in the ISSAC '26 paper. ⚠ Coefficients outside the
  declared range produce OOV tokens — widen `coefficients` in `lexer_expanded.yaml`
  if you change the field or sampler.

**Caveat — combining with `--order lex`**: not supported. `--training_order lex`
with the expanded lexer raises a `ValueError` in `build_load_preprocessor`
([core/train.py](core/train.py)); use the raw lexer for lex order.
