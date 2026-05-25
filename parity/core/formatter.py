"""
Convert permutation objects to their string representations.

Format
------
Input  : space-separated integers — the one-line notation of σ
Output : "+1" (even) or "-1" (odd)

Example:
    format_input([2, 0, 1])   →  "2 0 1"
    format_target(+1)         →  "+1"
    format_target(-1)         →  "-1"

Why these tokens?
-----------------
"+1" and "-1" are single tokens in the CALT vocabulary (numbers with sign).
The model output space is effectively binary, which makes evaluation simple.
"""


def format_input(perm: list[int]) -> str:
    """
    Convert a permutation list to a space-separated string.

    Example:  [2, 0, 1]  →  "2 0 1"
    """
    return " ".join(str(x) for x in perm)


def format_target(parity: int) -> str:
    """
    Convert +1 / −1 parity to its string token.

    Example:  +1  →  "+1"    −1  →  "-1"
    """
    return "+1" if parity == 1 else "-1"
