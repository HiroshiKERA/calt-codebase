"""
Generate (problem, answer) pairs for [TASK NAME].

Math
----
[Describe the mathematical problem:
  - What is the input space?
  - What is the output? Which algorithm computes it?
  - What constraints ensure the sample is non-degenerate?]

Output format
-------------
The generator returns (problem, answer). Each side can be:
  - a string             → written as-is
  - a list of objects    → DatasetPipeline joins via str() and ' | '
  - a SageMath polynomial / similar object → str() is called automatically

Reference
---------
Class-based generator pattern from
issac2026_experiments/groebner/generate_dataset.py.
"""

# TODO: import the math libraries you need
# import sage.misc.randstate as randstate
# from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler


class TaskGenerator:
    """
    Problem generator for [TASK NAME].

    Parameters
    ----------
    [your params]
    """

    def __init__(self, **params):
        # TODO: store sampler / hyperparameters
        pass

    def __call__(self, seed: int):
        """
        Generate one (problem, answer) pair.

        Returns
        -------
        (problem, answer)
            Each can be str, list of str, or SageMath/sympy objects.
        """
        # TODO: set random seed
        # TODO: generate input
        # TODO: compute answer
        # TODO: optionally retry on degenerate samples
        raise NotImplementedError
