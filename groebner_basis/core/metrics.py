"""
Metrics for the Gröbner basis computation task.

success_rate() is called after training to report evaluation accuracy.
Exact string match: the model wins only when its output is character-for-character
identical to the ground-truth Gröbner basis string.
"""


def success_rate(predictions: list[str], targets: list[str]) -> float:
    """
    Return the fraction of predictions that exactly match the target.

    Exact string match (after stripping whitespace) is the standard metric
    for this task — any difference in coefficients or ordering counts as failure.
    """
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return correct / len(targets)
