# Gröbner basis — scaling experiment

**Goal**: study how exact-match accuracy scales with model size and dataset size.

| Knob | Choices |
|------|---------|
| `--model_size` | `small`, `medium`, `large` |
| `--training_order` | `degrevlex` (default), `lex` (FGLM at load time) |

## What this experiment varies vs `toy/`
- Larger dataset: 50k train / 1k test (vs 1k / 200)
- Harder polynomials: `max_degree=5`, `max_num_terms=6`
- Configurable model size via CLI

## How to run
```bash
cd groebner_basis/experiments/scaling/scripts
python generate.py
python train.py --model_size small
python train.py --model_size large --training_order lex
```
