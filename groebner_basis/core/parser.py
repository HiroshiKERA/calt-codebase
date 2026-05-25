"""
Load-time preprocessor for lex order training of the Gröbner basis task.

For the default pipeline (degrevlex), no custom preprocessor is needed:
CALT's text pipeline reads "f1 | f2 # g1 | ... | gk" directly.

For lex order training, the train.py wires up a ChainLoadPreprocessor:
    TextToSageLoadPreprocessor  →  GroebnerLexOrderPreprocessor

The first stage parses text into SageMath polynomials in the (degrevlex) source ring.
The second stage converts F to a lex ring and recomputes the lex Gröbner basis.

Reference
---------
Mirrors `_GroebnerLexOrderPreprocessor` from
issac2026_experiments/groebner/train.py in HiroshiKERA/calt@experiment/issac2026.
"""

from typing import Any

import sage.all  # noqa: F401  # ensure Sage is initialised
from sage.all import PolynomialRing  # type: ignore


class GroebnerLexOrderPreprocessor:
    """
    Load-time converter: dict {'problem': [f1,f2,...], 'solution': [...]} →
    (input_text, target_text) in lex order.

    F and the lex Gröbner basis are joined with ' | '.
    Requires a zero-dimensional ideal for the Gröbner basis to terminate.

    Parameters
    ----------
    ring_src : PolynomialRing
        The source ring used for generation (typically degrevlex).
    delimiter : str
        Separator character (default '|').
    """

    def __init__(self, ring_src, delimiter: str = "|"):
        self.R_src = ring_src
        base = ring_src.base_ring()
        names = ring_src.variable_names()
        self.R_lex = PolynomialRing(base, names, order="lex")
        self.delimiter = delimiter

    def process_sample(self, source: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(source, dict):
            raise TypeError(
                f"GroebnerLexOrderPreprocessor expects dict source, got {type(source).__name__}"
            )
        F_src = source.get("problem") or []
        F_lex = [self.R_lex(f) for f in F_src]
        I_lex = self.R_lex.ideal(F_lex)
        # Force `libsingular:std` instead of the default. The default tries
        # `stdhilb`, which raises `intvec`/`bigintvec` on some QQ-lex inputs in
        # SageMath 10.6. `libsingular:std` is the standard, stable choice for
        # both QQ and GF(p).
        G_lex = list(I_lex.groebner_basis("libsingular:std"))

        sep = f" {self.delimiter} "
        input_text = sep.join(str(f) for f in F_lex)
        target_text = sep.join(str(g) for g in G_lex)
        return input_text, target_text
