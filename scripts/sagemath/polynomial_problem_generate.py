from typing import Any, List, Tuple, Dict, Union
from sage.all import PolynomialRing, GF, ZZ, QQ, RR
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular
from calt.generator.sagemath import (
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

    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
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

    def __call__(self, seed: int) -> Tuple[List[MPolynomial_libsingular], List[MPolynomial_libsingular]]:
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
        G = [sum(F[:i + 1]) for i in range(len(F))]

        return F, G


class PolyStatisticsCalculator(BaseStatisticsCalculator):
    """
    Statistics calculator for polynomial problems.
    """

    def __init__(self, ring: PolynomialRing):
        """
        Initialize polynomial statistics calculator.

        Args:
            ring: Polynomial ring
        """
        self.ring = ring
        self.num_vars = ring.ngens()
        self.coeff_field = ring.base_ring()

    def __call__(
        self,
        problem: Union[List[MPolynomial_libsingular], MPolynomial_libsingular],
        solution: Union[List[MPolynomial_libsingular], MPolynomial_libsingular],
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a single generated sample.

        Args:
            problem: Either a list of polynomials or a single polynomial
            solution: Either a list of polynomials or a single polynomial

        Returns:
            Dictionary with keys "input" and "output", each mapping to a sub-dictionary 
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
            - density: Density of the system (ratio of total terms to maximum possible terms)
        """

        if isinstance(problem, list):
            problem_stats = self.poly_system_stats(problem)
        else:
            problem_stats = self.poly_system_stats([problem])
        if isinstance(solution, list):
            solution_stats = self.poly_system_stats(solution)
        else:
            solution_stats = self.poly_system_stats([solution])

        return {
            "input": problem_stats,
            "output": solution_stats,
        }

    def poly_system_stats(self, polys: List[MPolynomial_libsingular]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of polynomials.

        Args:
            polys: List of polynomials

        Returns:
            Dictionary containing statistical information about the polynomials
        """
        num_polys = len(polys) # Number of polynomials in the system

        if num_polys == 0:
            return {"num_polynomials": 0, "total_degree": 0, "total_terms": 0}

        degrees = [max(p.total_degree(), 0) for p in polys] # if polynomial p is zero, then p.total_degree() is -1, so we need to set it to 0
        num_terms = [len(p.monomials()) for p in polys]


        coeffs = []
        for p in polys:
            if self.coeff_field == QQ:
                # For QQ, consider both numerators and denominators
                coeffs.extend([abs(float(c.numerator())) for c in p.coefficients()])
                coeffs.extend([abs(float(c.denominator())) for c in p.coefficients()])
            elif self.coeff_field == RR:
                # For RR, take absolute values
                coeffs.extend([abs(float(c)) for c in p.coefficients()])
            elif self.coeff_field == ZZ:
                # For ZZ, take absolute values
                coeffs.extend([abs(int(c)) for c in p.coefficients()])
            elif self.coeff_field.is_field() and self.coeff_field.characteristic() > 0:  # GF
                # For finite fields, just take the values
                coeffs.extend([int(c) for c in p.coefficients()])

        stats = {
            # System size statistics
            "num_polynomials": num_polys, # Number of polynomials in the system
            # Degree statistics
            "sum_total_degree": sum(degrees), # Sum of total degrees of all polynomials in the system
            "max_total_degree": max(degrees), # Maximum degree of any polynomial in the system
            "min_total_degree": min(degrees), # Minimum degree of any polynomial in the system
            # Term count statistics
            "sum_num_terms": sum(num_terms), # Total number of terms across all polynomials in the system
            "max_num_terms": max(num_terms), # Maximum number of terms in any polynomial in the system
            "min_num_terms": min(num_terms), # Minimum number of terms in any polynomial in the system
            # Coefficient statistics
            "max_abs_coeff": max(coeffs) if coeffs else 0, # Maximum absolute coefficient value in the system
            "min_abs_coeff": min(coeffs) if coeffs else 0, # Minimum absolute coefficient value in the system
            # Additional system properties
            "density": float(sum(num_terms)) / (num_polys * (1 + max(degrees)) ** self.num_vars), # Density of the system (ratio of total terms to maximum possible terms))
        }

        return stats
        

def main():
    save_dir = "dataset/sagemath/polynomial_problem/GF7_n=2"

    # set up polynomial ring
    R = PolynomialRing(GF(7), 2, "x", order="degrevlex")

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
    )

    # Initialize problem generator
    problem_generator = PartialSumProblemGenerator(
        sampler=sampler,
        max_polynomials=5,
        min_polynomials=2,
    )

    # Initialize statistics calculator
    statistics_calculator = PolyStatisticsCalculator(ring=R)

    # Initialize dataset generator
    dataset_generator = DatasetGenerator(
        backend="multiprocessing", 
        n_jobs=-1, 
        verbose=True, 
        root_seed=100
    )

    # Generate training set
    train_samples, train_stats = dataset_generator.run(
        train=True,
        num_samples=1000,
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
    )

    # Generate test set
    test_samples, test_stats = dataset_generator.run(
        train=False,
        num_samples=1000,
        problem_generator=problem_generator,
        statistics_calculator=statistics_calculator,
    )

    # Initialize writer
    dataset_writer = DatasetWriter(save_dir)

    # Save datasets
    dataset_writer.save_dataset(train_samples, train_stats, "train")
    dataset_writer.save_dataset(test_samples, test_stats, "test")


if __name__ == "__main__":
    main()
