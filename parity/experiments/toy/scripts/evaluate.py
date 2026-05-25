"""Print evaluation results for the parity toy experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shared.paths import output_dir
from shared.plotting import load_eval_results, show_examples

if __name__ == "__main__":
    results = output_dir(__file__) / "results"
    generated, references = load_eval_results(results)

    rate = sum(g == r for g, r in zip(generated, references)) / len(references)
    print(f"Success rate: {100 * rate:.1f}%  ({sum(g == r for g, r in zip(generated, references))}/{len(references)})")

    even_total = sum(1 for r in references if r.strip() == "+1")
    odd_total = len(references) - even_total
    print(f"  Even (+1): {even_total}  |  Odd (−1): {odd_total}")

    print()
    show_examples(generated, references, n=5, successes=True)
    show_examples(generated, references, n=5, successes=False)
