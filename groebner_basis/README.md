# groebner_basis

**Task**: Given a polynomial f and generators g1, g2, g3, predict the remainder
of f on division by the Gröbner basis of the ideal ⟨g1, g2, g3⟩.

## The math

Let R = ℤ[x, y, z] and I = ⟨g1, g2, g3⟩ ⊂ R.

1. Compute G = grevlex Gröbner basis of I  (using Buchberger's algorithm).
2. Compute r = I.reduce(f)  (remainder of the division of f by G).

The model receives (f, G) as input and must predict r.

**Why is this hard?** Even knowing G, computing r requires a multi-step
polynomial division algorithm. The model must learn this algorithm implicitly.

## Data

- Field: ℤ (integers)
- Variables: x, y, z
- Polynomials: 2–5 terms, degree ≤ 4, coefficients in [-10, 10]
- Stored as: pickle (SageMath objects)

## Experiments

| Experiment | What varies |
|---|---|
| `toy/` | Small model, 1000 train samples — verify the pipeline runs |
| `scaling/` | Larger models and datasets — measure scaling behavior |
| `ablation/` | Order (grevlex vs. lex), pattern (remainder vs. quotients+remainder) |

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

`train.py` applies the `dataset_load_preprocessor` (TextToSage → FGLM for lex,
TextToSage → ExpandedForm for C/E format) **every time it starts**. For large
datasets and the `lex` chain that means several minutes of SageMath/FGLM
recomputation before each run.

`preprocess.py` runs that chain **once** offline and writes a JSONL cache. By
default it also pre-tokenizes all the way to `input_ids` so that `train.py`
performs **zero tokenization at runtime** — only padding + GPU forward.

```bash
cd groebner_basis/experiments/toy/scripts
python preprocess.py                          # cache (pretokenized) for degrevlex
python preprocess.py --training_order lex     # cache (pretokenized) for lex order
python preprocess.py --no-pretokenize         # cache stops at post-FGLM strings (legacy mode)
python train.py --training_order lex          # auto-detects cache → fast startup
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
after the load chain (FGLM/Expanded/etc.) has run, before tokenization:

| Slot | File (create to enable) | Scope |
|---|---|---|
| Base hook   | `shared/base_postprocessor.py`         | Applied to every task that wires user hooks (groebner + border) |
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

- `--order lex` : convert to lex order via FGLM before training
- `--pattern 2` : predict quotients + remainder instead of remainder only

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

**Caveat — combining with `--order lex`**: both modes work together. The lex
preprocessor runs first (recomputes G in lex order via FGLM), then expanded form
runs on the lex-ordered output. See [core/train.py](core/train.py) `run_training()`.
