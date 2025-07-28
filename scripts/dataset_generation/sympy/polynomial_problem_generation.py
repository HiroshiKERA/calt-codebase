import random
from sympy import QQ, RR, ZZ
from sympy.polys.rings import PolyElement

import click
import warnings

from calt.dataset_generator.sympy import (
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
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial partial sum sampler.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """

        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> tuple[list[PolyElement], list[PolyElement]]:
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

        # Set random seed
        random.seed(seed)

        # Choose number of polynomials for this sample
        num_polys = random.randint(self.min_polynomials, self.max_polynomials)

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
        problem: list[PolyElement] | PolyElement,
        solution: list[PolyElement] | PolyElement,
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
            - max_total_degree: Maximum degree of any polynomial in the system
            - min_total_degree: Minimum degree of any polynomial in the system
            - sum_num_terms: Total number of terms across all polynomials in the system
            - max_num_terms: Maximum number of terms in any polynomial in the system
            - min_num_terms: Minimum number of terms in any polynomial in the system
            - max_abs_coeff: Maximum absolute coefficient value in the system
            - min_abs_coeff: Minimum absolute coefficient value in the system
        """
        return {
            "problem": self.poly_system_stats(
                problem if isinstance(problem, list) else [problem]
            ),
            "solution": self.poly_system_stats(
                solution if isinstance(solution, list) else [solution]
            ),
        }

    def _extract_coefficients(self, poly: PolyElement) -> list[float]:
        """Extract coefficients from polynomial based on field type."""
        coeff_field = poly.ring.domain
        if coeff_field == QQ:
            return [abs(float(c.numerator)) for c in poly.coeffs()] + [
                abs(float(c.denominator)) for c in poly.coeffs()
            ]
        elif coeff_field in (RR, ZZ):
            return [abs(float(c)) for c in poly.coeffs()]
        elif coeff_field.is_FiniteField:
            return [int(c) for c in poly.coeffs()]
        return []

    def poly_system_stats(self, polys: list[PolyElement]) -> dict[str, int | float]:
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

        # Calculate basic statistics
        degrees = [self.total_degree(p) for p in polys]
        num_terms = [len(p.terms()) for p in polys]
        coeffs = [c for p in polys for c in self._extract_coefficients(p)]

        return {
            # System size statistics
            "num_polynomials": len(polys),
            # Degree statistics
            "sum_total_degree": sum(degrees),
            "max_total_degree": max(degrees),
            "min_total_degree": min(degrees),
            # Term count statistics
            "sum_num_terms": sum(num_terms),
            "max_num_terms": max(num_terms),
            "min_num_terms": min(num_terms),
            # Coefficient statistics
            "max_abs_coeff": max(coeffs) if coeffs else 0,
            "min_abs_coeff": min(coeffs) if coeffs else 0,
        }

    def total_degree(self, poly: PolyElement) -> int:
        """Compute total degree of a polynomial.

        The total degree of a polynomial is the maximum sum of exponents among all
        monomials in the polynomial. For example, in x**2*y + x*y, the total degree
        is 3 (from x**2*y where 2+1=3).

        Args:
            poly: Polynomial

        Returns:
            Total degree of the polynomial

        Examples:
            >>> calc = PolyStatisticsCalculator()
            >>> p = x**2*y + x*y**2 + x + y
            >>> calc.total_degree(p)
            3
        """
        if poly.is_zero:
            return 0
        else:
            return max(sum(monom) for monom in poly.monoms())


@click.command()
@click.option("--save_dir", type=str, default="")
@click.option(
    "--n_jobs", type=int, default=32
)  # set the number of jobs for parallel processing (check your machine's capacity by command `nproc`)
def main(save_dir, n_jobs):
    if save_dir == "":
        # warning
        save_dir = "dataset/sympy/partial_sum/GF7_n=3"
        warnings.warn(
            f"No save directory provided. Using default save directory {save_dir}."
        )

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        symbols="x, y, z",  # "x, y, z, ... " or "x0, x1, x2, ... "
        field_str="GF(7)",  # "QQ", "RR", "ZZ", "GF(p)", "GFp", where p is a prime number
        order="grevlex",  # "lex", "grevlex", "grlex", "ilex", "igrevlex", "igrlex"
        max_num_terms=5,
        max_degree=10,
        min_degree=1,
        degree_sampling="uniform",  # "uniform" or "fixed"
        term_sampling="uniform",  # "uniform" or "fixed"
        max_coeff=None,  # Used for RR and ZZ
        num_bound=None,  # Used for QQ
        strictly_conditioned=True,
        nonzero_instance=True,
        max_attempts=1000,
    )

    # Initialize problem generator
    problem_generator = PartialSumProblemGenerator(
        sampler=sampler,
        max_polynomials=5,
        min_polynomials=2,
    )

    # Initialize statistics calculator
    statistics_calculator = PolyStatisticsCalculator()

    # Initialize dataset generator
    dataset_generator = DatasetGenerator(
        backend="multiprocessing",
        n_jobs=n_jobs,
        verbose=True,  # Whether to show progress
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
        dataset_sizes={
            "train": 100000,
            "test": 1000,
        },  # train: 100000 samples, test: 1000 samples
        batch_size=100000,  # set batch size
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
        dataset_writer=dataset_writer,
    )


if __name__ == "__main__":
    main()
