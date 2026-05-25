"""Load a trained model and evaluate it on the test set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from shared.paths import config_dir, output_dir
from shared.plotting import load_eval_results, show_examples


@click.command()
@click.option(
    "--training_order",
    type=click.Choice(["degrevlex", "lex"]),
    default="degrevlex",
    help="Monomial order used at training time (default: degrevlex).",
)
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to train config YAML (defaults to ../configs/train.yaml).",
)
def main(training_order: str, config_path: str | None) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)

    base_dir = cfg.train.get("save_dir", "../outputs/results")
    results_rel = base_dir.rstrip("/") + f"_{training_order}"
    results = (Path(__file__).parent / results_rel).resolve()

    generated, references = load_eval_results(results)

    n_correct = sum(g == r for g, r in zip(generated, references))
    rate = n_correct / len(references)
    print(f"Success rate: {100 * rate:.1f}%  ({n_correct}/{len(references)})")

    print()
    show_examples(generated, references, n=5, successes=True)
    print()
    show_examples(generated, references, n=5, successes=False)


if __name__ == "__main__":
    main()
