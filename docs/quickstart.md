# Quick Start

This tutorial provides a minimal introduction on how to build your own project to experiment with the CALT library.

## Setup

CALT is based on Python and several popular frameworks including SageMath, PyTorch, and HuggingFace. We provide a conda environment that offers a simple setup.

First, install conda enviroment from [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux). Then, clone this repository and create an conda enviroment with our enviroment.yml. 

```bash
git clone https://github.com/HiroshiKERA/calt-codebase.git
cd calt-codebase
conda env create -f environment.yml  # see environment.yml to define your environment name (default: calt-env)
```

After creating the environment, activate it:

```bash
conda activate calt-env  # replace `calt-env` with the one you named
```

The environment is now all set.

## First Run

The codebase is ready to generate a simple dataset and train a Transformer. You can first try this to see if everything works properly.

### 1. Dataset Construction

```bash
python scripts/sagemath/polynomial_problem_generation.py
```

By default, this generates train and test sets of the partial polynomial sum task. The generated dataset can be found in `dataset/partial_sum`.

### 2. Training Transformer

```bash
python scripts/sagemath/train.py --config config/train_example.yaml
```

This trains a Transformer model with the setup described in `config/train_example.yaml`. This file specifies training parameters, Transformer architecture, paths to load data, and the directory to save the results and logs. The training process can also be viewed in the WandB platform (the link will be printed once training starts).

**Note**: Initial runs may require your WandB API key. Visit [WandB](https://wandb.ai/), create your account, and copy & paste your API key to the terminal as required.

## Your Own Project

Now you can create custom script files for your own project. In `scripts/sagemath/`, you can find three script files for dataset construction as examples: one for numerical tasks, another for polynomial tasks, and the last for other tasks.

Let's take `polynomial_problem_generation.py` as an example. You can find two classes: `PartialSumProblemGenerator` and `PolyStatisticsCalculator`. The former is the main part of instance generation, and the latter computes statistics of generated instances.

Here, the task input is a list of polynomials $F = [f_1, ..., f_s]$, and the expected output is the list of cumulative sums $G = [g_1, ..., g_s]$, where $g_k = f_1 + \cdots + f_k$.

```python
class PartialSumProblemGenerator:
    def __init__(self, sampler: PolynomialSampler, ...):
        self.sampler = sampler
        ...

    def __call__(self, seed: int):
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        F = self.sampler.sample(num_samples=num_polys)

        # Generate partial sums for solution
        G = [sum(F[:i + 1]) for i in range(len(F))]

        return F, G
```

This includes three essential components:
- `sampler` for random sampling of polynomials, which is necessary in most cases
- `seed` for reproducibility. Make sure you have this argument in your custom problem generators
- `__call__` for generating an instance. The expected output is a pair of input and output

Define your own class by renaming the class and redefining `__call__`. If needed, also define your own PolyStatisticsCalculator for dataset analysis. Finally, rewrite `main()` accordingly. 

If your instance generation requires a sophisticated process, you may add subroutines in the class, and further add some utility files in `src/`. Below is an example. 

```python
class GroebnerProblemGenerator:
    def __init__(self, sampler: PolynomialSampler, max_polynomials: int, min_polynomials: int):
        self.sampler = sampler
        self.min_polynomials = min_polynomials
        self.max_polynomials = max_polynomials

    def __call__(self, seed: int):
        # Set random seed for SageMath's random state
        randstate.set_random_seed(seed)

        # Choose number of polynomials for this sample
        num_polys = randint(self.min_polynomials, self.max_polynomials)

        # Generate problem polynomials using sampler
        G = self.sampling_groebner_basis()
        F = self.ideal_invariant_transform(G)

        return F, G
    
    def sampling_groebner_basis(self):
        '''Randomly sample a Gröbner basis G'''
        pass

    def ideal_invariant_transform(self, G):
        '''Generate non-Gröbner basis F such that <F> = <G>'''
        pass
```

To train the Transformer model on the custom dataset, rewrite the path name and other dataset setup in `config/train_example.yaml`:

```yaml
train_dataset_path: data/GB_problem/GF7_n=2/train_raw.txt
test_dataset_path: data/GB_problem/GF7_n=2/test_raw.txt
num_variables: 2
max_degree: 14
max_coeff: 10
field: GF7
```

**Note**: Make sure that `max_coeff` and `max_degree` are large enough to cover all coefficients and degrees appearing in instances. Otherwise, you may encounter tokenization errors with the unknown token `[UNK]`.
