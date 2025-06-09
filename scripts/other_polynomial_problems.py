from typing import Any, List, Tuple
import random
from calt import PolynomialSampler


class SumProblemGenerator:
    """
    Problem generator for polynomial sum problems.

    This generator generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a single polynomial g = f_1 + f_2 + ... + f_n.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial sum generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - Input polynomial system F
        - Output polynomial g (sum of F)

        Args:
            seed: Seed for random number generator
        """
        random.seed(seed)

        # Choose number of polynomials for this sample
        num_polys = random.randint(self.min_polynomials, self.max_polynomials)

        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate output polynomial g (sum of F)
        g = sum(F)

        return F, g


class GCDProblemGenerator:
    """
    Problem generator for polynomial GCD problems.

    This generator generates problems in which the input is a pair of polynomials F = [f_1, f_2],
    and the output is a single polynomial g = GCD(f_1, f_2).
    """

    def __init__(self, sampler: PolynomialSampler):
        """
        Initialize polynomial GCD generator.

        Args:
            sampler: Polynomial sampler
        """
        self.sampler = sampler

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - Input polynomial system F
        - Output polynomial g (GCD of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, g)
        """
        random.seed(seed)

        # Generate input polynomials using sampler
        base_gcd, q1, q2 = self.sampler.sample(num_samples=3)

        # Generate output polynomial g (GCD of F)
        extra = q1.gcd(q2)
        new_gcd = base_gcd * extra
        q1 = q1.quo(extra)
        q2 = q2.quo(extra)

        F = [new_gcd * q1, new_gcd * q2]
        g = new_gcd.monic()

        return F, g


class ProductProblemGenerator:
    """
    Problem generator for polynomial product problems.

    This generator generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a single polynomial g = f_1 * f_2 * ... * f_n.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial product generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - Input polynomial system F
        - Output polynomial g (product of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, g)
        """
        random.seed(seed)

        # Choose number of polynomials for this sample
        num_polys = random.randint(self.min_polynomials, self.max_polynomials)

        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate output polynomial g (product of F)
        g = 1
        for f in F:
            g *= f

        return F, g


class PartialProdProblemGenerator:
    """
    Problem generator for polynomial product problems.

    This generator generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 * f_2 * ... * f_i.
    """

    def __init__(
        self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int
    ):
        """
        Initialize polynomial product generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], List[Any]]:
        """
        Generate a single sample.

        Each sample consists of:
        - Input polynomial system F
        - Output polynomial system G (partial products of F)

        Args:
            seed: Seed for random number generator

        Returns:
            Tuple containing (F, G)
        """
        random.seed(seed)

        # Choose number of polynomials for this sample
        num_polys = random.randint(self.min_polynomials, self.max_polynomials)

        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial products for output
        G = []
        current_prod = 1
        for f in F:
            current_prod *= f
            G.append(current_prod)

        return F, G
