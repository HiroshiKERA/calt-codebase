# Problem Generator Examples

This document showcases various problem generator implementations available in the CALT library. These examples demonstrate different mathematical operations and can be used as templates for creating your own problem generators.

**Reference files**:

- `scripts/dataset_generation/sagemath/other_polynomial_problems.py` - Polynomial problem generators
- `scripts/dataset_generation/sagemath/arithmetic_problem_generation.py` - Arithmetic problem generators
- `scripts/dataset_generation/sagemath/transposed_matrix_problem_generation.py` - Polynomial matrix problem generators

## Symbolic Problems

### 1. Sum Problem Generator

**Problem**: Given a list of polynomials $F = [f_1, f_2, ..., f_n]$, compute their sum $g = f_1 + f_2 + ... + f_n$.

Please refer to `scripts/dataset_generation/sagemath/other_polynomial_problems.py`.


```python
class SumProblemGenerator:
    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate solution polynomial g (sum of F)
        g = sum(F)

        return F, g
```

### 2. GCD Problem Generator

**Problem**: Given a pair of polynomials $F = [f_1, f_2]$, compute their greatest common divisor $g = \text{GCD}(f_1, f_2)$.

Please refer to `scripts/dataset_generation/sagemath/other_polynomial_problems.py`.

```python
class GCDProblemGenerator:
    def __init__(self, sampler: PolynomialSampler):
        self.sampler = sampler

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Generate problem polynomials using sampler
        gcd, q1, q2 = self.sampler.sample(num_samples=3)

        # Generate solution polynomial g (GCD of F)
        _gcd = q1.gcd(q2)
        gcd, q1, q2 = gcd * _gcd, ring(q1 / _gcd), ring(q2 / _gcd)
        F = [gcd * q1, gcd * q2]
        g = ring(gcd / gcd.lc())

        return F, g
```

### 3. Product Problem Generator

**Problem**: Given a list of polynomials $F = [f_1, f_2, ..., f_n]$, compute their product $g = f_1 \times f_2 \times ... \times f_n$.

Please refer to `scripts/dataset_generation/sagemath/other_polynomial_problems.py`.

```python
class ProductProblemGenerator:
    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], MPolynomial_libsingular]:
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate solution polynomial g (product of F)
        g = ring(1)
        for f in F:
            g *= f

        return F, g
```

### 4. Partial Sum Problem Generator

**Problem**: Given a list of polynomials $F = [f_1, f_2, ..., f_n]$, compute partial sums $G = [g_1, g_2, ..., g_n]$ where $g_i = f_1 + f_2 + ... + f_i$.

Please refer to `scripts/dataset_generation/sagemath/polynomial_problem_generation.py`.

```python
class PartialSumProblemGenerator:
    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial sums for solution
        G = [sum(F[: i + 1]) for i in range(len(F))]

        return F, G
```

### 5. Partial Product Problem Generator

**Problem**: Given a list of polynomials $F = [f_1, f_2, ..., f_n]$, compute partial products $G = [g_1, g_2, ..., g_n]$ where $g_i = f_1 \times f_2 \times ... \times f_i$.

Please refer to `scripts/dataset_generation/sagemath/other_polynomial_problems.py`.

```python
class PartialProdProblemGenerator:
    def __init__(
        self, sampler: PolynomialSampler, min_polynomials: int, max_polynomials: int
    ):
        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(
        self, seed: int
    ) -> tuple[list[MPolynomial_libsingular], list[MPolynomial_libsingular]]:
        # Get ring from sampler
        ring = self.sampler.get_ring()

        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial products for solution
        G = []
        current_prod = ring(1)
        for f in F:
            current_prod *= f
            G.append(current_prod)

        return F, G
```

### 6. Matrix Transpose Problem Generator

**Problem**: Given a polynomial matrix $F$, compute its transpose matrix $G = F^\mathrm{T}$.

Please refer to `scripts/dataset_generation/sagemath/transposed_matrix_problem_generation.py`.

```python
class PolyMatrixTransposeProblemGenerator:
    def __init__(
        self, sampler: PolynomialSampler, min_matrix_size: int, max_matrix_size: int
    ):
        self.sampler = sampler
        self.min_matrix_size = min_matrix_size
        self.max_matrix_size = max_matrix_size

    def __call__(
        self, seed: int
    ) -> tuple[
        list[list[MPolynomial_libsingular]], list[list[MPolynomial_libsingular]]
    ]:
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose matrix size for this sample
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
```

## Arithmetic Problems

### Integer Factorization Problem Generator

**Problem**: Given an integer $n$, find its prime factorization as a list of prime factors in ascending order.

Please refer to `scripts/dataset_generation/sagemath/arithmetic_problem_generation.py`.

```python
class IntFactorProblemGenerator:
    def __init__(self, prime_upper_bound: int, min_factors: int, max_factors: int):
        self.prime_upper_bound = prime_upper_bound
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.prime_lst = list(prime_range(2, self.prime_upper_bound + 1))

    def __call__(self, seed: int) -> tuple[int, list[int]]:
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of factors for this sample
        num_factors = randint(self.min_factors, self.max_factors)

        # Generate random prime factors
        factors = [choice(self.prime_lst) for _ in range(num_factors)]

        # Sort factors in ascending order
        factors.sort()

        # Calculate problem integer by multiplying factors
        n = 1
        for p in factors:
            n *= p

        return n, factors
```
