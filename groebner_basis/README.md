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

## Variants

- `--order lex` : convert to lex order via FGLM before training
- `--pattern 2` : predict quotients + remainder instead of remainder only
