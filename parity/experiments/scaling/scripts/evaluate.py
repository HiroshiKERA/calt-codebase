"""Load a trained model and evaluate it on the test set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click

from shared.paths import output_dir
from shared.plotting import load_eval_results, show_examples


@click.command()
@click.option("--n", type=int, default=None, help="Same as train.py: --n N reads ../outputs/results_nN/.")
def main(n: int | None) -> None:
    # train.py saves to ../outputs/results_n<n> when --n is given.
    results = output_dir(__file__) / (f"results_n{n}" if n is not None else "results")
    generated, references = load_eval_results(results)

    rate = sum(g == r for g, r in zip(generated, references)) / len(references)
    print(f"Success rate: {100 * rate:.1f}%  ({sum(g == r for g, r in zip(generated, references))}/{len(references)})")
    print()
    show_examples(generated, references, n=5, successes=True)
    print()
    show_examples(generated, references, n=5, successes=False)


if __name__ == "__main__":
    main()
