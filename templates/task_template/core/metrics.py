"""
Metrics and per-sample statistics for [TASK NAME].

Two functions:
  instance_stats  →  passed to DatasetPipeline as statistics_calculator
                     called per sample during generation
  success_rate    →  called after training to report accuracy
"""


def instance_stats(problem, answer) -> dict:
    """
    Compute descriptive statistics for one (problem, answer) instance.

    These are aggregated across all samples and saved in {split}_stats.yaml.
    Useful for characterizing the dataset (e.g., average degree, sequence length).

    Returns
    -------
    dict[str, int | float]
        Keys are statistic names; values are numbers.

    TODO: add statistics relevant to your task.
    Example:
        return {
            "input_length": len(problem),
            "output_length": len(answer),
        }
    """
    raise NotImplementedError


def success_rate(predictions: list[str], targets: list[str]) -> float:
    """
    Return the fraction of predictions that exactly match the target.

    Exact string match (after stripping whitespace) is the standard metric.
    """
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return correct / len(targets)
