import sys

sys.path.insert(0, "../calt/src")  # use calt in local dir, not from library
sys.path.append("src")

from sage.all import ZZ, QQ, RR, GF, PolynomialRing
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular

import click
import warnings

# Import from local calt library (prioritized over pip-installed calt)
from calt.dataset_generator.sagemath import (
    PolynomialSampler,
    DatasetGenerator,
    DatasetWriter,
    BaseStatisticsCalculator,
)

class PartialSumProblemGenerator:
    """
    Problem generator for partial sum problems involving polynomials.

    This generator creates problems in which the problem is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the solution is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 + f_2 + ... + f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        """
        Initialize polynomial partial sum sampler.

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
        - Solution: polynomial system G (partial sums of F)

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

        # Generate partial sums for solution
        G = [sum(F[: i + 1]) for i in range(len(F))]

        return F, G


class PolyStatisticsCalculator(BaseStatisticsCalculator):
    """
    Statistics calculator for polynomial problems.
    """

    def __call__(
        self,
        problem: list[MPolynomial_libsingular] | MPolynomial_libsingular,
        solution: list[MPolynomial_libsingular] | MPolynomial_libsingular,
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for a single generated sample.

        Args:
            problem: Either a list of polynomials or a single polynomial
            solution: Either a list of polynomials or a single polynomial

        Returns:
            Dictionary with keys "problem" and "solution", each mapping to a sub-dictionary
            containing descriptive statistics including:
            - num_polynomials: Number of polynomials in the system
            - sum_total_degree: Sum of total degrees of all polynomials in the system
            - min_total_degree: Minimum degree of any polynomial in the system
            - max_total_degree: Maximum degree of any polynomial in the system
            - sum_num_terms: Total number of terms across all polynomials in the system
            - min_num_terms: Minimum number of terms in any polynomial in the system
            - max_num_terms: Maximum number of terms in any polynomial in the system
            - min_abs_coeff: Minimum absolute coefficient value in the system
            - max_abs_coeff: Maximum absolute coefficient value in the system

        Examples:
            >>> stats_calculator = PolyStatisticsCalculator()
            >>> stats = stats_calculator(problem=[x^2 + 1, x^3 + 2], solution=[x^2 + 1, x^3 + 2])
            >>> stats['problem']['num_polynomials']
            2
            >>> stats['solution']['num_polynomials']
            2
        """
        return {
            "problem": self.poly_system_stats(
                problem if isinstance(problem, list) else [problem]
            ),
            "solution": self.poly_system_stats(
                solution if isinstance(solution, list) else [solution]
            ),
        }

    def _extract_coefficients(self, poly: MPolynomial_libsingular) -> list[float | int]:
        """Extract coefficients from polynomial based on field type."""
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
        Calculate statistics for a list of polynomials.

        Args:
            polys: List of polynomials

        Returns:
            Dictionary containing statistical information about the polynomials
        """
        if not polys:
            raise ValueError(
                "Cannot calculate statistics for empty list of polynomials"
            )

        degrees = [
            max(p.total_degree(), 0) for p in polys
        ]  # if polynomial p is zero, then p.total_degree() is -1, so we need to set it to 0
        num_terms = [len(p.monomials()) for p in polys]
        coeffs = [c for p in polys for c in self._extract_coefficients(p)]

        return {
            # System size statistics
            "num_polynomials": len(polys),
            # Degree statistics
            "sum_total_degree": sum(degrees),
            "min_total_degree": min(degrees),
            "max_total_degree": max(degrees),
            # Term count statistics
            "sum_num_terms": sum(num_terms),
            "min_num_terms": min(num_terms),
            "max_num_terms": max(num_terms),
            # Coefficient statistics
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
        # warning
        save_dir = "dataset/partial_sum/GF7_n=3"
        warnings.warn(
            f"No save directory provided. Using default save directory {save_dir}."
        )

    # Initialize polynomial ring
    R = PolynomialRing(GF(7), 3, "x", order="degrevlex")

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        ring=R,
        max_num_terms=5,
        max_degree=10,
        min_degree=1,
        degree_sampling="uniform",  # "uniform" or "fixed"
        term_sampling="uniform",  # "uniform" or "fixed"
        max_coeff=None,  # Used for RR and ZZ
        num_bound=None,  # Used for QQ
        strictly_conditioned=False,
        nonzero_instance=True,
        nonzero_coeff=True,
    )

    # Initialize problem generator
    problem_generator = PartialSumProblemGenerator(
        sampler=sampler,
        min_polynomials=2,
        max_polynomials=5,
    )

    # Initialize statistics calculator
    statistics_calculator = PolyStatisticsCalculator()

    # Initialize dataset generator
    dataset_generator = DatasetGenerator(
        backend="multiprocessing",
        n_jobs=n_jobs,
        verbose=True,
        root_seed=100,
    )

    # Initialize writer
    dataset_writer = DatasetWriter(
        save_dir=save_dir,
        save_text=True,  # whether to save raw text files
        save_json=True,  # whether to save JSON files
    )

    # Generate datasets with batch processing
    dataset_generator.run(
        dataset_sizes={"train": 100000, "test": 1000},
        batch_size=100000,  # set batch size
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
        dataset_writer=dataset_writer,
    )


if __name__ == "__main__":
    main()
