# Custom Dataset Generator

This guide explains how to create custom datasets for training Transformer models on mathematical tasks using the CALT library. The library provides a flexible framework for generating various types of mathematical problems, from polynomial operations to integer arithmetic.

## Overview

The CALT library offers a modular approach to dataset generation, allowing you to create custom problem generators for any mathematical task. After following the Quick Start guide, you can extend your experiments by creating custom datasets. The process involves several components:

1. **Custom Problem Generator**: Defines how to create problem-solution pairs
2. **Custom Statistics Calculator (optional)**: Analyzes the generated dataset


## 1. Custom Problem Generator

To create your own problem generator, you need to define a class with a `__call__` method according to your specific mathematical task.

### Basic Structure

```python
import sage.misc.randstate as randstate

class CustomProblemGenerator:
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        # Add other parameters as needed
        
    def __call__(self, seed: int):
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)
        
        # Your problem generation logic here
        # ...
        
        # Return a tuple (problem, solution)
        return problem, solution
```

### Key Points

- **Method Input**: The `__call__` method takes a `seed` parameter for reproducibility
- **Reproducibility**: Always use `randstate.set_random_seed(seed)` at the beginning
- **Sampler**: Use the provided sampler for generating mathematical objects (required for polynomial tasks, optional for arithmetic tasks)
- **Method Output**: Return a tuple `(problem, solution)`

<!-- #### Problem and Solution Types -->

**Note**: The `problem` and `solution` can be of various types depending on your mathematical task:

- **Single values**: `int`, `float`, `MPolynomial_libsingular`
- **Lists**: `list[int]`, `list[float]`,  or `list[MPolynomial_libsingular]`
- **Matrices**: Nested lists, e.g., `list[list[int]]` or `list[list[MPolynomial_libsingular]]`
<!-- - **Complex objects**: Any combination of the above -->

For concrete examples of problem generators, see [Problem Generator Examples](problem_generator_examples.md).

<!-- **Examples**

- `(int, list[int])` - Integer factorization: input is an integer, output is list of prime factors
- `(list[MPolynomial_libsingular], MPolynomial_libsingular)` - Polynomial sum: input is list of polynomials, output is single polynomial
- `(list[list[MPolynomial_libsingular]], list[list[MPolynomial_libsingular]])` - Matrix transpose: input and output are polynomial matrices -->


## 2. Custom Statistics Calculator (optional)

Create a custom statistics calculator to analyze your generated data. The calculator should inherit from `BaseStatisticsCalculator` and implement the `__call__` method.

### Basic Structure

```python
from calt.dataset_generator.sagemath import BaseStatisticsCalculator

class CustomStatisticsCalculator(BaseStatisticsCalculator):
    def __call__(
        self, problem: Any, solution: Any
    ) -> dict[str, dict[str, int | float]]:
        """
        Calculate statistics for a single sample.

        Args:
            problem: The problem data
            solution: The solution data

        Returns:
            Dictionary with keys "problem" and "solution", each mapping to a sub-dictionary
            containing descriptive statistics.
        """
        return {
            "problem": self.calculate_problem_stats(problem),
            "solution": self.calculate_solution_stats(solution)
        }
    
    def calculate_problem_stats(self, problem) -> dict[str, int | float]:
        """Calculate statistics for the problem data."""
        # Your problem statistics logic here
        pass
    
    def calculate_solution_stats(self, solution) -> dict[str, int | float]:
        """Calculate statistics for the solution data."""
        # Your solution statistics logic here
        pass
```

### Key Points

- **Inheritance**: Must inherit from `BaseStatisticsCalculator`
- **Method Input**: The `__call__` method takes `problem` and `solution`.
- **Method Output**: Return a dictionary `dict[str, dict[str, int |float]]` with keys "problem" and "solution".

For concrete examples of statistics calculators, see [Statistics Calculator Examples](statistics_calculator_examples.md).


<!-- ## 3. Dataset Generator

### Command Line Interface

Use the provided scripts with custom parameters:

```bash
python scripts/dataset_generation/sagemath/polynomial_problem_generation.py --save_dir dataset/my_custom_problem --n_jobs 16
```

### Programmatic Interface

```python
# Initialize dataset generator
dataset_generator = DatasetGenerator(
    backend="multiprocessing",
    n_jobs=32,
    verbose=True,
    root_seed=100,
)

# Initialize writer
dataset_writer = DatasetWriter(
    save_dir="dataset/my_custom_problem",
    save_text=True,
    save_json=True,
)

# Generate datasets
dataset_generator.run(
    dataset_sizes={"train": 100000, "test": 1000, "val": 1000},
    batch_size=10000,
    problem_generator=your_problem_generator,
    statistics_calculator=your_statistics_calculator,
    dataset_writer=dataset_writer,
)
```

## Configuration for Training

After generating your dataset, update the training configuration:

```yaml
# config/train_custom.yaml
train_dataset_path: dataset/my_custom_problem/train_raw.txt
test_dataset_path: dataset/my_custom_problem/test_raw.txt
num_variables: 3
max_degree: 15
max_coeff: 20
field: GF7
```

**Important**: Ensure that `max_coeff` and `max_degree` are large enough to cover all coefficients and degrees in your dataset to avoid tokenization errors.

## Best Practices

1. **Reproducibility**: Always use the seed parameter for consistent results
2. **Validation**: Test your problem generator with a small dataset first
3. **Statistics**: Use statistics calculators to understand your data distribution
4. **Documentation**: Document the mathematical formulation of your problem
5. **Testing**: Verify that your problem-solution pairs are mathematically correct

<!-- ## Next Steps

Once you have generated your dataset, proceed to the training phase as described in the Quick Start guide:

```bash
python scripts/train/train.py --config config/train_custom.yaml
```

This modular approach allows you to experiment with various mathematical tasks and compare model performance across different problem types. For more examples and detailed usage, refer to the scripts in `scripts/dataset_generation/sagemath/`. --> 
