# Polynomial Problems

This document showcases various polynomial problems available in the CALT library. These problems demonstrate different mathematical operations and can be used to train Transformer models for polynomial manipulation tasks.

## Polynomial Sum Problem

The **SumProblemGenerator** creates problems where the input is a list of polynomials $F = [f_1, f_2, ..., f_n]$ and the expected output is a single polynomial $g = f_1 + f_2 + ... + f_n$.

### Problem Structure
- **Input**: List of polynomials $F = [f_1, f_2, ..., f_n]$
- **Output**: Single polynomial $g = \sum_{i=1}^{n} f_i$

### Example Implementation
```python
class SumProblemGenerator:
    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int):
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

## Polynomial GCD Problem

The **GCDProblemGenerator** creates problems where the input is a pair of polynomials $F = [f_1, f_2]$ and the expected output is their greatest common divisor $g = \text{GCD}(f_1, f_2)$.

### Problem Structure
- **Input**: Pair of polynomials $F = [f_1, f_2]$
- **Output**: Single polynomial $g = \text{GCD}(f_1, f_2)$

### Example Implementation
```python
class GCDProblemGenerator:
    def __init__(self, sampler: PolynomialSampler):
        self.sampler = sampler

    def __call__(self, seed: int):
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

## Polynomial Product Problem

The **ProductProblemGenerator** creates problems where the input is a list of polynomials $F = [f_1, f_2, ..., f_n]$ and the expected output is their product $g = f_1 \times f_2 \times ... \times f_n$.

### Problem Structure
- **Input**: List of polynomials $F = [f_1, f_2, ..., f_n]$
- **Output**: Single polynomial $g = \prod_{i=1}^{n} f_i$

### Example Implementation
```python
class ProductProblemGenerator:
    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int):
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

## Partial Product Problem

The **PartialProdProblemGenerator** creates problems where the input is a list of polynomials $F = [f_1, f_2, ..., f_n]$ and the expected output is a list of partial products $G = [g_1, g_2, ..., g_n]$, where $g_i = f_1 \times f_2 \times ... \times f_i$.

### Problem Structure
- **Input**: List of polynomials $F = [f_1, f_2, ..., f_n]$
- **Output**: List of polynomials $G = [g_1, g_2, ..., g_n]$ where $g_i = \prod_{j=1}^{i} f_j$

### Example Implementation
```python
class PartialProdProblemGenerator:
    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        self.sampler = sampler
        self.max_polynomials = max_polynomials
        self.min_polynomials = min_polynomials

    def __call__(self, seed: int):
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

## Usage

To use these problem generators in your own project, follow the pattern shown in the Quick Start guide:

1. **Import the generator**: Import the desired problem generator class
2. **Initialize with sampler**: Create an instance with a `PolynomialSampler` and appropriate parameters
3. **Generate instances**: Call the generator with a seed to create problem-solution pairs
4. **Train your model**: Use the generated data to train a Transformer model

Each generator follows the same interface pattern, making it easy to experiment with different polynomial problems and compare model performance across various mathematical tasks.
