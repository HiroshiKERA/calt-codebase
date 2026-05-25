# parity

**Task**: Given a permutation σ of {0, 1, …, n−1}, predict its sign: +1 (even) or −1 (odd).

## The math

A permutation σ is **even** if it can be written as a product of an even number
of transpositions, and **odd** otherwise.

Equivalently:

    sign(σ) = (−1)^(number of inversions)

where an **inversion** is a pair (i, j) with i < j but σ(i) > σ(j).

**Example**:

    σ = [2, 0, 1]   (σ(0)=2, σ(1)=0, σ(2)=1)
    Inversions: (0,1) since σ(0)=2 > σ(1)=0
                (0,2) since σ(0)=2 > σ(2)=1
    2 inversions → even → sign(σ) = +1

## Why this task?

Parity is a benchmark for learning **global** properties of sequences:
the model cannot look at any single element — it must compare all pairs.
This is known to require attention across the full sequence length.

## Data format

Input  : space-separated integers — the one-line notation of σ
Output : "+1" or "−1"

    "2 0 1"  →  "+1"
    "3 0 2 1"  →  "−1"

## Experiments

| Experiment | What varies |
|---|---|
| `toy/` | n=5 permutations, 1000 samples — verify the pipeline |
| `scaling/` | n ∈ {5, 8, 12} — does performance degrade with longer sequences? |

## Quick start

```bash
cd parity/experiments/toy/scripts
python generate.py
python train.py
python evaluate.py
```
