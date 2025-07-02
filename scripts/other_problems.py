from typing import Any, List, Tuple
import sage.misc.randstate as randstate
from sage.misc.prandom import randint
from calt import PolynomialSampler


class SumProblemGenerator:
    """
    Problem generator for polynomial sum problems.

    This class generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a single polynomial g = f_1 + f_2 + ... + f_n.
    """

    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        """
        Initialize polynomial sum generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """
        self.sampler = sampler
        self.ring = sampler.ring
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - F: a list of randomly generated polynomials.
        - g: the sum of all polynomials in F.

        Args:
            seed: Seed for SageMath's random state

        Returns:
            Tuple containing (F, g)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Randomly determine the number of polynomials in F
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate the input list F using the sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Compute the sum g of all polynomials in F
        g = sum(F)

        return F, g


class GCDProblemGenerator:
    """
    Problem generator for polynomial GCD problems.

    This class generates problems in which the input is a pair of polynomials F = [f_1, f_2],
    and the output is a single polynomial g = GCD(f_1, f_2).
    """

    def __init__(self, sampler: PolynomialSampler):
        """
        Initialize polynomial GCD generator.

        Args:
            sampler: Polynomial sampler
        """

        self.sampler = sampler
        self.ring = sampler.ring

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - F: a list of two polynomials.
        - g: the GCD of the two polynomials in F.

        Args:
            seed: Seed for SageMath's random state

        Returns:
            Tuple containing (F, g)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Generate three random polynomials: gcd (the intended GCD), and co-factors q1 and q2
        gcd, q1, q2 = self.sampler.sample(num_samples=3)

        # Normalize q1 and q2 so that their GCD is 1
        _gcd = q1.gcd(q2)
        gcd, q1, q2 = gcd * _gcd, self.ring(q1 / _gcd), self.ring(q2 / _gcd)

        # Construct input polynomials
        F = [gcd * q1, gcd * q2]

        # Normalize the GCD to have leading coefficient 1
        g = self.ring(gcd / gcd.lc())

        return F, g


class ProductProblemGenerator:
    """
    Problem generator for polynomial product problems.

    This class generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a single polynomial g = f_1 * f_2 * ... * f_n.
    """

    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        """
        Initialize polynomial product generator.

        Args:
            sampler: Polynomial sampler
            max_polynomials: Maximum number of polynomials in F
            min_polynomials: Minimum number of polynomials in F
        """

        self.sampler = sampler
        self.ring = sampler.ring
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int) -> Tuple[List[Any], Any]:
        """
        Generate a single sample.

        Each sample consists of:
        - F: a list of randomly generated polynomials.
        - g: the product of all polynomials in F.

        Args:
            seed: Seed for SageMath's random state

        Returns:
            Tuple containing (F, g)
        """

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate input polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate output polynomial g (product of F)
        g = self.ring(1)
        for f in F:
            g *= f

        return F, g


class PartialProdProblemGenerator:
    """
    Problem generator for polynomial partial product problems.

    This class generates problems in which the input is a list of polynomials F = [f_1, f_2, ..., f_n],
    and the output is a list of polynomials G = [g_1, g_2, ..., g_n], where g_i = f_1 * f_2 * ... * f_i.
    """

    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        """
        Initialize polynomial partial product generator.

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
        - G: a list of partial products of polynomials in F.

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

        # Generate partial products for output
        G = []
        current_prod = self.ring(1)
        for f in F:
            current_prod *= f
            G.append(current_prod)

        return F, G
