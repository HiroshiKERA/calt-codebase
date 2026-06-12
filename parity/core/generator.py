"""
Generate (problem, answer) pairs for the permutation parity task.

Math
----
Input  : a random permutation σ of {0, 1, …, n−1}
Output : sign(σ) ∈ {+1, −1}

sign(σ) = (−1)^k   where k = number of inversions in σ

An inversion is a pair (i, j) with i < j and σ(i) > σ(j).

Output format
-------------
The generator returns plain strings. The DatasetPipeline writes them to disk:
  - input  : "2 0 1"        (space-separated one-line notation)
  - output : "+1" or "-1"

Reference
---------
Class-based generator pattern matching
issac2026_experiments/digit_product/generate_dataset.py
(HiroshiKERA/calt@experiment/issac2026).
"""

import random
from itertools import combinations


def count_inversions(perm: list[int]) -> int:
    """Number of pairs (i, j) with i < j and perm[i] > perm[j]."""
    return sum(
        1 for i, j in combinations(range(len(perm)), 2)
        if perm[i] > perm[j]
    )


def permutation_parity(perm: list[int]) -> int:
    """+1 (even) or −1 (odd) — equals (−1)^(number of inversions)."""
    return 1 if count_inversions(perm) % 2 == 0 else -1


class ParityGenerator:
    """
    Problem generator: random permutation σ of {0,…,n−1}, answer is sign(σ).

    Parameters
    ----------
    n : int
        Permutation length (default 5).
    """

    def __init__(self, n: int = 5):
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")
        self.n = n

    def __call__(self, seed: int) -> tuple[str, str]:
        random.seed(seed)
        perm = list(range(self.n))
        random.shuffle(perm)
        parity = permutation_parity(perm)
        input_str = " ".join(str(x) for x in perm)
        target_str = "+1" if parity == 1 else "-1"
        return input_str, target_str
