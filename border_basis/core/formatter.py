"""
Convert SageMath polynomial objects to tokenizable strings for the border basis task.

Format
------
Input  : "g1 | g2 | g3"        (generators separated by ' | ')
Target : "b1 | b2 | ... | bk"  (border basis elements)

Example
-------
generators:   g1 = x^2 + 6,  g2 = y^2 + x,  g3 = x*y + 1  (over GF(7))
border basis: B = [x + 1, y + 2, ...]
→ input  = "x^2 + 6 | y^2 + x | x*y + 1"
→ target = "x + 1 | y + 2 | ..."
"""

SEP = " | "


def poly_to_str(p) -> str:
    """Return the default SageMath string representation of a polynomial."""
    return str(p)


def format_input(generators: list) -> str:
    """
    Concatenate all generators into a single input string.

    Example: [g1, g2, g3] → "g1 | g2 | g3"
    """
    return SEP.join(poly_to_str(g) for g in generators)


def format_target(basis: list) -> str:
    """
    Concatenate all border basis elements into a single target string.

    Example: [b1, b2, b3] → "b1 | b2 | b3"
    """
    return SEP.join(poly_to_str(b) for b in basis)
