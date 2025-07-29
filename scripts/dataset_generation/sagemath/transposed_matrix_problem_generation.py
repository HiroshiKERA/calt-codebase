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


class PolyMatrixTransposeProblemGenerator:
    """
    Problem generator for transposed matrix problems involving polynomials.

    This generator creates problems in which the problem is a matrix of polynomials F,
    and the solution is the transposed matrix of F.
    """

    def __init__(
        self, sampler: PolynomialSampler, min_matrix_size: int, max_matrix_size: int
    ):
        """
        Initialize polynomial partial sum sampler.

        Args:
            sampler: Polynomial sampler
            min_matrix_size: Minimum size of the matrix
            max_matrix_size: Maximum size of the matrix
        """

        self.sampler = sampler
        self.min_matrix_size = min_matrix_size
        self.max_matrix_size = max_matrix_size

    def __call__(
        self, seed: int
    ) -> tuple[
        list[list[MPolynomial_libsingular]], list[list[MPolynomial_libsingular]]
    ]:
        """
        Generate a single sample.

        Each sample consists of:
        - Problem: polynomial matrix F
        - Solution: polynomial matrix G (transposed of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, G)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        matrix_size = randint(self.min_matrix_size, self.max_matrix_size)

        # Generate matrix of polynomials using sampler
        matrix_list = self.sampler.sample(
            size=(matrix_size, matrix_size), num_samples=1
        )
        _F = matrix_list[0]

        # Transpose the matrix
        _G = _F.transpose()

        # Convert the matrix to a two-level nested list of polynomials
        F = list(map(list, _F))
        G = list(map(list, _G))

        return F, G


class PolyMatrixStatsCalculator(BaseStatisticsCalculator):
    """
    Statistics calculator specialized for polynomial matrix problems.

    This calculator is designed specifically for problems involving polynomial matrices,
    such as transposed matrix problems where both problem and solution are matrices
    of polynomials.

    Requirements:
        - Both problem and solution must be polynomial matrices (two-level nested lists)
        - Each element in the matrices must be a polynomial
        - The matrices must not be empty
    """

    def __call__(
        self,
        problem: list[list[MPolynomial_libsingular]],
        solution: list[list[MPolynomial_libsingular]],
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for a polynomial matrix problem.

        Args:
            problem: A matrix of polynomials (two-level nested list). Must be a non-empty matrix.
            solution: A matrix of polynomials (two-level nested list). Must be a non-empty matrix.

        Returns:
            Dictionary with keys "problem" and "solution", each mapping to a sub-dictionary
            containing descriptive statistics including:
            - num_polynomials: Number of polynomials in the matrix
            - matrix_dimensions: Tuple of (rows, columns) for the matrix
            - sum_total_degree: Sum of total degrees of all polynomials in the matrix
            - min_total_degree: Minimum degree of any polynomial in the matrix
            - max_total_degree: Maximum degree of any polynomial in the matrix
            - sum_num_terms: Total number of terms across all polynomials in the matrix
            - min_num_terms: Minimum number of terms in any polynomial in the matrix
            - max_num_terms: Maximum number of terms in any polynomial in the matrix
            - min_abs_coeff: Minimum absolute coefficient value in the matrix
            - max_abs_coeff: Maximum absolute coefficient value in the matrix

        Examples:
            >>> stats_calculator = PolyMatrixStatsCalculator()
            >>> stats = stats_calculator(
            ...     problem=[[x^2 + 1, x^3 + 2], [x^4 + 3, x^5 + 4]],
            ...     solution=[[x^2 + 1, x^3 + 2], [x^4 + 3, x^5 + 4]]
            ... )
            >>> stats['problem']['num_polynomials']
            4
            >>> stats['solution']['num_polynomials']
            4

        Raises:
            ValueError: If either problem or solution is not a valid polynomial matrix
        """
        # Flatten the matrices to get all polynomials
        flattened_problem = [poly for row in problem for poly in row]
        flattened_solution = [poly for row in solution for poly in row]

        # Get basic statistics using existing poly_system_stats method
        problem_stats = self.poly_system_stats(flattened_problem)
        solution_stats = self.poly_system_stats(flattened_solution)

        return {
            "problem": problem_stats,
            "solution": solution_stats,
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
        save_dir = "dataset/transposed_matrix/GF7_n=3"
        warnings.warn(
            f"No save directory provided. Using default save directory {save_dir}."
        )

    # Initialize polynomial ring
    R = PolynomialRing(GF(7), 3, "x", order="degrevlex")

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        ring=R,
        max_num_terms=2,
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
    problem_generator = PolyMatrixTransposeProblemGenerator(
        sampler=sampler,
        min_matrix_size=2,
        max_matrix_size=2,
    )

    # Initialize statistics calculator
    statistics_calculator = PolyMatrixStatsCalculator()

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
