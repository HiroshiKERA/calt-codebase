"""
Generate (problem, answer) pairs for the border basis task.

Math
----
R = k[x, y]   (typically k = GF(p) for guaranteed 0-dimensional ideals)
I = ⟨g1, g2, g3⟩ ⊂ R

Given F = [g1, g2, g3] sampled at random, the learning task is to predict
the **border basis** G of I (Algorithm 4.1 BBasis from the reference paper).

Reference
---------
Implements the data-generation pattern for the paper
*Computational Algebra with Attention: Transformer Oracles for Border Basis
Algorithms* (Kera, Pelleriti, Ishihara, Zimmer, Pokutta — arXiv:2505.23696, 2025).

The underlying algorithm lives in `border_basis.core.algorithm.BorderBasisCalculator`,
which is a verbatim port of
`HiroshiKERA/OracleBorderBasis/src/border_basis_lib/border_basis.py`.

Class-based generator pattern matches `issac2026_experiments/groebner/generate_dataset.py`.
"""

import sage.all  # noqa: F401  # ensure Sage is initialised
import sage.misc.randstate as randstate  # type: ignore

from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler

from .algorithm import BorderBasisCalculator


class BorderBasisGenerator:
    """
    Problem generator: F = [g1, ..., g_{num_polynomials}],
    answer G = border basis of ⟨F⟩.

    Parameters
    ----------
    sampler : PolynomialSampler
        Builds the random polynomials. Best used with a GF(p) field so that
        the ideal is zero-dimensional with high probability.
    num_polynomials : int
        Number of generators (default 3).
    use_fast_elimination : bool
        If True, the Gaussian elimination step uses a sorted-list optimisation.
    lstabilization_only : bool
        If True, stop at the L-stable span (no order-ideal optimisation +
        no basis transformation). Much faster, but you only get an intermediate
        basis rather than the final border basis.
    """

    def __init__(
        self,
        sampler: PolynomialSampler,
        num_polynomials: int = 3,
        use_fast_elimination: bool = True,
        lstabilization_only: bool = False,
    ):
        self.sampler = sampler
        self.num_polynomials = num_polynomials
        self.use_fast_elimination = use_fast_elimination
        self.lstabilization_only = lstabilization_only

    def __call__(self, seed: int) -> tuple[list, list]:
        randstate.set_random_seed(seed)
        R = self.sampler.get_ring()
        calculator = BorderBasisCalculator(R)

        max_retries = 100
        last_F, last_G = None, None
        for _ in range(max_retries):
            try:
                F = self.sampler.sample(num_samples=self.num_polynomials)

                # Border basis requires a 0-dimensional ideal.
                ideal = R.ideal(F)
                if ideal.dimension() != 0:
                    continue

                # Uniform weights over all terms up to max input degree.
                d = max(f.degree() for f in F)
                terms = calculator.terms_up_to_degree(d)
                weights = {t: 1 for t in terms}

                G, _O, _timings = calculator.compute_border_basis(
                    F,
                    weights,
                    use_fast_elimination=self.use_fast_elimination,
                    lstabilization_only=self.lstabilization_only,
                )

                last_F, last_G = list(F), list(G)
                if last_G:
                    return last_F, last_G
            except Exception:
                # MILP / Sage edge case — try next seed.
                continue

        return last_F, last_G
