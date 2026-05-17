from sage.all import ZZ, QQ, RR
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular

import click
import warnings

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler
from omegaconf import OmegaConf


class CumulativeSumInstanceGenerator:
    """
    Instance generator for cumulative sum problems involving polynomials.

    This generator creates instances in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the answer is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 + f_2 + ... + f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        """
        Initialize cumulative sum instance generator.

        Args:
            sampler: Polynomial sampler
            min_polynomials: Minimum number of polynomials in F
            max_polynomials: Maximum number of polynomials in F
        """

        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial system F
        - Answer: polynomial system G (cumulative sums of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, G)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate cumulative sums for answer
        G = [sum(F[: i + 1]) for i in range(len(F))]

        return F, G


class PolyStatisticsCalculator:
    """
    Per-sample statistics for polynomial cumulative-sum datasets.

    Used with :class:`DatasetPipeline` to summarize each (problem, answer) pair
    produced by :class:`CumulativeSumInstanceGenerator`. Each side is typically
    a list of polynomials; a single polynomial is wrapped in a one-element list.
    """

    def __call__(
        self,
        problem: list[MPolynomial_libsingular] | MPolynomial_libsingular,
        answer: list[MPolynomial_libsingular] | MPolynomial_libsingular,
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for one polynomial cumulative-sum sample.

        Args:
            problem: Polynomial list ``F``, or a single polynomial.
            answer: Cumulative-sum list ``G``, or a single polynomial.

        Returns:
            Dictionary with keys ``"problem"`` and ``"answer"``. Each value is a
            sub-dictionary returned by :meth:`poly_system_stats`.
        """
        return {
            "problem": self.poly_system_stats(
                problem if isinstance(problem, list) else [problem]
            ),
            "answer": self.poly_system_stats(
                answer if isinstance(answer, list) else [answer]
            ),
        }

    def _extract_coefficients(self, poly: MPolynomial_libsingular) -> list[float | int]:
        """
        Return absolute coefficient magnitudes for statistics.

        Supports polynomials over ``QQ`` (numerators and denominators), ``ZZ``,
        ``RR``, and finite fields. Returns an empty list for unsupported base rings.
        """
        coeff_field = poly.parent().base_ring()
        if coeff_field == QQ:
            return [abs(float(c.numerator())) for c in poly.coefficients()] + [
                abs(float(c.denominator())) for c in poly.coefficients()
            ]
        elif coeff_field in (RR, ZZ):
            return [abs(float(c)) for c in poly.coefficients()]
        elif coeff_field.is_field() and coeff_field.characteristic() > 0:
            return [int(c) for c in poly.coefficients()]
        return []

    def poly_system_stats(
        self, polys: list[MPolynomial_libsingular]
    ) -> dict[str, int | float]:
        """
        Calculate aggregate statistics for a list of polynomials.

        Args:
            polys: Non-empty list of polynomials in one problem or answer side.

        Returns:
            Dictionary with keys:
            - ``num_polynomials``: Length of ``polys``.
            - ``sum_total_degree``, ``min_total_degree``, ``max_total_degree``:
              Aggregates over per-polynomial degrees. Univariate rings (``ngens() == 1``)
              use ``degree()``; multivariate rings use ``total_degree()``. Zero
              polynomials are counted as 0 (Sage returns ``-1`` for the degree).
            - ``sum_num_terms``, ``min_num_terms``, ``max_num_terms``:
              Aggregates over monomial counts.
            - ``min_abs_coeff``, ``max_abs_coeff``: Min/max over absolute
              coefficients (0 if no coefficients are extracted).

        Raises:
            ValueError: If ``polys`` is empty.
        """
        if not polys:
            raise ValueError(
                "Cannot calculate statistics for empty list of polynomials"
            )

        R = polys[0].parent()
        univariate = R.ngens() == 1

        degrees = [
            int(max(p.degree() if univariate else p.total_degree(), 0)) for p in polys
        ]

        num_terms = [len(p.monomials()) for p in polys]
        coeffs = [c for p in polys for c in self._extract_coefficients(p)]

        return {
            "num_polynomials": len(polys),
            "sum_total_degree": sum(degrees),
            "min_total_degree": min(degrees),
            "max_total_degree": max(degrees),
            "sum_num_terms": sum(num_terms),
            "min_num_terms": min(num_terms),
            "max_num_terms": max(num_terms),
            "min_abs_coeff": min(coeffs) if coeffs else 0,
            "max_abs_coeff": max(coeffs) if coeffs else 0,
        }


@click.command()
@click.option("--save_dir", type=str, default="")
@click.option(
    "--n_jobs", type=int, default=32
)  # set the number of jobs for parallel processing (check your machine's capacity by command `nproc`)
def main(save_dir, n_jobs):
    if save_dir == "":
        save_dir = "dataset/cumulative_sum/GF7_n=3"
        warnings.warn(
            f"No save directory provided. Using default save directory {save_dir}."
        )

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        symbols="x,y,z",
        field_str="GF(7)",
        order="grevlex",
        max_num_terms=5,
        max_degree=10,
        min_degree=1,
    )

    # Initialize instance generator
    instance_generator = CumulativeSumInstanceGenerator(
        sampler=sampler,
        min_polynomials=2,
        max_polynomials=5,
    )

    # Initialize statistics calculator
    statistics_calculator = PolyStatisticsCalculator()

    config = {
        "save_dir": save_dir,
        "num_train_samples": 100000,
        "num_test_samples": 1000,
        "batch_size": 10000,
        "n_jobs": n_jobs,
        "root_seed": 42,
        "verbose": True,
        "backend": "sagemath",
        "save_text": True,
        "save_json": True,
    }

    pipeline = DatasetPipeline.from_config(
        OmegaConf.create(config),
        instance_generator=instance_generator,
        statistics_calculator=statistics_calculator,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
