# Parity — scaling experiment

**Goal**: study how exact-match accuracy scales with the permutation length `n`
(harder for larger `n` because the input space is `n!`).

## What it varies vs `toy/`
- Larger dataset (50k train / 1k test)
- `n` is selectable via CLI flag: `--n {5,7,10}`
- Larger model

## How to run
```bash
cd parity/experiments/scaling/scripts
python generate.py --n 7         # generates ../data_n7/
python train.py --n 7            # trains on it
```

The default toy run lives in `../toy/` and uses `n=5`.
