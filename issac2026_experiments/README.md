# ISSAC 2026 experiments

Every experiment behind Section 5 of the paper: the scripts that generate the
data, the configs that were trained, and the evaluation that produces the
tables. Built on [CALT](https://github.com/HiroshiKERA/calt) (`calt-x`).

Each task is a directory holding `generate_dataset.py`, `train.py`, `configs/`
and `sh/`, and is self-contained: nothing outside the task directory is needed
to reproduce its rows.

This layout deliberately differs from the `core/` + `experiments/` convention
the other tasks in this repository follow. These are the scripts exactly as they
were run to produce the published numbers, down to the config files; rewriting
them into the house layout would make the correspondence between a table in the
paper and the code that produced it something a reader has to take on trust.

```
issac2026_experiments/
├── arithmetic_addition/         A-Addition            (Tables 1, 2, 3)
├── arithmetic_factorization/    A-Factorization       (Table 1)
├── polynomial_multiplication/   P-Multiplication      (Tables 1, 2, 3, 8)
├── polynomial_reduction/        P-Reduction           (Tables 1, 2, 8)
├── digit_product/               A-Multiplication-DL   (Table 4)
├── relu_recurrence/             A-ReLU Recurrence     (Table 4)
├── groebner/                    Groebner bases        (Tables 5, 6, 7, 8)
├── evaluate/                    collects every run into one success-rate table
└── sh/                          generate_all_datasets.sh, train_all.sh
```

## Requirements

- SageMath (the dataset generators sample polynomials and compute Groebner
  bases through it), tested on 10.7
- `calt-x`, and the usual PyTorch / transformers stack it pulls in
- A CUDA GPU per concurrent run

The generators import `sage.all` before anything else; without it the deeper
Sage imports fail with `ImportError: PolynomialRing_generic`.

## Reproducing a table

Data first, then training, then evaluation. From a task directory:

```bash
bash sh/generate_dataset.sh     # writes data/,     gitignored
bash sh/train.sh                # writes results/,  gitignored
```

`sh/train.sh` launches its runs with `nohup ... &` and pins each to a GPU index
written into the script — edit those assignments to match your machine. To run
everything, `bash sh/generate_all_datasets.sh` then `bash sh/train_all.sh` from
this directory; that is 50 runs launched at once.

Generation is deterministic: each sample is seeded from the `root_seed` in the
task's data config, so regenerating gives the same dataset. The datasets
themselves are not shipped — they are ~500 MB and one command away.

Collect every finished run into a single table with:

```bash
python3 evaluate/run_all_eval.py     # -> evaluate/success_rate_table.{csv,md}
```

## What varies within a task

- **Coefficient field** — `configs/ZZ`, `configs/GF7`, `configs/GF31`,
  `configs/GF97`, one subdirectory each, same structure.
- **Chain of thought** — the `+` variants of Table 3 come from the `_dg1` /
  `_dg3` configs (intermediate steps written out) rather than a separate script.
- **Order of intermediate steps** — `train.py --target_reversed` swaps in
  `ReversedOrderLoadPreprocessor`, which is the only difference between the
  forward and reverse rows of Table 4.
- **Monomial order** — `groebner/train.py --training_order lex` moves the
  sampled system into a lex ring and recomputes the basis at load time; the
  dataset on disk is degrevlex either way.
- **Problem size** — the Table 5 rows are 2 variables at degree 4. The other
  scales of Table 6 (degree 16, degree 32, and a 5x5 square system) have their
  own configs; `groebner/sh/generate_scale_datasets.sh` then
  `groebner/sh/train_scale.sh` runs them, both orders each.
- **Input representation** — Table 8 compares `model_type: generic` against
  `model_type: monomial` on the same expanded-form data, at a budget of 32
  epochs fixed across all tasks. These are the `train_repr_standard.yaml` /
  `train_repr_monomial.yaml` configs, launched by the `sh/train_representation.sh`
  of each task directory.

  Two things to know before running them. `groebner/train.py` needs
  `--expanded_form`, because the Table 5 runs read raw polynomial strings and
  the monomial embedding cannot. And the monomial configs have to spell out
  `monomial_separators: ["+", "|"]`: the model defaults to `["+", "||"]`, while
  the expanded form here is built with `delimiter=" | "`, so without that line
  every sample fails the model's alignment check.

  Because the budget is fixed at 32 epochs and, for P-Groebner, the input
  representation changes, these success rates are not comparable with the ones
  in Tables 1 to 5. The paper says so in Section 5.5.

## Two additions that are not in the paper's tables

**`groebner/measure_gb_basis_stats.py`** — the degree and the term count of the
basis under each order, which is Table 7. The timing script above answers how
long the computer algebra system takes; this one answers how much the model has
to generate. `bash sh/measure_gb_basis_stats.sh` runs the paper's setting and
the same scale points as the timing sweep.

**`groebner/measure_gb_timing.py`** — the classical side of the Table 5
comparison. Times a Groebner basis in lex and in degrevlex on the same systems,
with `libsingular:slimgb`, so the computer-algebra cost gap can be put next to
the learning gap. stdfglm is deliberately not used: it is the degrevlex+FGLM
route, so it would measure the conversion strategy rather than a direct lex
computation. `bash sh/measure_gb_timing.sh` runs the paper's setting, a sweep in
the number of variables, and a degree control.

**`compare_order_curves.py`** — reads the `trainer_state.json` of finished runs
and prints how the two orders of Table 4 got to their final numbers: loss by
epoch, and the epoch at which each first reaches a given accuracy. When both
orders saturate, the final success rate no longer separates them and the curves
are what carry the comparison.

```bash
python3 compare_order_curves.py \
    --run "forward:relu_recurrence/results" \
    --run "reverse:relu_recurrence/results/reversed"
```

## Model

All the paper's configs use `model_type: generic`, CALT's encoder-decoder
Transformer, at 6+6 layers and `d_model` 512.

`model_type: decoder_only` is also available and is the standard choice in the
arithmetic-reasoning literature: one causal stack over the concatenation of the
problem and its solution, with next-token prediction starting at the first
solution token — the problem part is context and carries no loss. The
`configs/train_decoder.yaml` of `relu_recurrence` and `digit_product` are the
Table 4 tasks under that model, unchanged in every other respect.
