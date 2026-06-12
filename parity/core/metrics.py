"""
Metrics for the parity task.

success_rate() is the exact-match accuracy used after training to score the model.
"""


def success_rate(predictions: list[str], targets: list[str]) -> float:
    """
    Return the fraction of predictions that exactly match the target parity.

    Both "+1" and "-1" are single tokens — exact match is the right metric.
    """
    if not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return correct / len(targets)
