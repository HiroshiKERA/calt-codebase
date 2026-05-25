"""
Helper functions to render SageMath polynomial objects as text strings.

Format
------
Input  : "f1 | f2"             (generators separated by ' | ')
Target : "g1 | g2 | ... | gk"  (Gröbner basis elements)

These are NOT used by the CALT DatasetPipeline (which calls str() on each
polynomial automatically and joins lists with ' | '). They exist for:
  - inspecting generated samples in notebooks/REPL
  - building unit tests
"""

SEP = " | "


def format_input(generators: list) -> str:
    """[f1, f2] → 'f1 | f2'"""
    return SEP.join(str(g) for g in generators)


def format_target(basis: list) -> str:
    """[g1, ..., gk] → 'g1 | ... | gk'"""
    return SEP.join(str(g) for g in basis)
