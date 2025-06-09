from typing import Any, List, Tuple
from sage.all import PolynomialRing, GF  # QQ, GF, RR, ZZ
import sage.misc.randstate as randstate
from sage.misc.prandom import randint

from other_problems import (
    SumProblemGenerator
)  # GCDProblemGenerator, ProductProblemGenerator, PartialProdProblemGenerator

from transformer_algebra import PolynomialSampler, DatasetGenerator, DatasetWriter


class PartialSumProblemGenerator:
    """
    Problem generator for polynomial partial sum problems.

    This class generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 + f_2 + ... + f_i.
    """

    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        """
        Initialize polynomial partial sum generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """

        self.sampler = sampler
        self.ring = sampler.ring
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], List[Any]]:
        """
        Generate a single sample.

        Each sample consists of:
        - F: a list of randomly generated polynomials.
        - G: a list of partial sums of polynomials in F.

        Args:
            seed: Seed for SageMath's random state

        Returns:
            Tuple containing (F, G)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial sums for output
        G = []
        current_sum = self.ring(0)
        for f in F:
            current_sum += f
            G.append(current_sum)

        return F, G


def main():
    save_dir = "data/sum_problem/GF7_n=2"

    # set up polynomial ring
    ring = PolynomialRing(GF(7), 2, "x", order="degrevlex")

    # Initialize polynomial sampler
    sampler = PolynomialSampler(
        ring=ring,
        max_num_terms=10,
        max_degree=5,
        min_degree=1,
        degree_sampling="uniform",  # "uniform" or "fixed"
        term_sampling="uniform",  # "uniform" or "fixed"
        max_coeff=None,  # Used for RR and ZZ
        num_bound=None,  # Used for QQ
        strictly_conditioned=True,
        nonzero_instance=True,
    )

    # Initailize problem generators
    problem_generator = SumProblemGenerator(
        sampler=sampler,
        max_polynomials=2,
        min_polynomials=2,
    )

    # problem_generator = PartialSumProblemGenerator(
    #     sampler=sampler,
    #     max_polynomials=5,
    #     min_polynomials=2,
    # )

    # problem_generator = GCDProblemGenerator(
    #     sampler=sampler,
    # )

    # problem_generator = ProductProblemGenerator(
    #     sampler=sampler,
    #     max_polynomials=5,
    #     min_polynomials=2,
    # )

    # problem_generator = PartialProdProblemGenerator(
    #     sampler=sampler,
    #     max_polynomials=5,
    #     min_polynomials=2,
    # )

    dataset_generator = DatasetGenerator(problem_type="polynomial", ring=ring, n_jobs=-1, verbose=True, root_seed=42)

    # Generate training set
    train_samples, train_stats = dataset_generator.run(num_samples=100000, train=True, problem_generator=problem_generator)

    # Generate test set
    test_samples, test_stats = dataset_generator.run(num_samples=1000, train=False, problem_generator=problem_generator)

    # Initialize writer
    dataset_writer = DatasetWriter(save_dir)

    # Save datasets
    dataset_writer.save_dataset(train_samples, train_stats, "train")
    dataset_writer.save_dataset(test_samples, test_stats, "test")


if __name__ == "__main__":
    main()
