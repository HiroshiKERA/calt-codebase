"""
Metrics for the border basis task.

success_rate() is called after training to report evaluation accuracy.
Exact string match: the model wins only when its output is character-for-character
identical to the ground-truth border basis string.
"""


def success_rate(predictions: list[str], targets: list[str]) -> float:
    """
    Return the fraction of predictions that exactly match the target.

    The ordering of border basis elements is deterministic (fixed by SageMath),
    so exact string match is the correct metric.
    """
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return correct / len(targets)
