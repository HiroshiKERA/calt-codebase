"""Train the Gröbner basis model for the toy experiment.

Reference: issac2026_experiments/groebner/train.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from groebner_basis.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to train config YAML (defaults to ../configs/train.yaml).",
)
@click.option(
    "--data_config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to data config YAML (defaults to ../configs/data.yaml). Used for lex training_order.",
)
@click.option(
    "--training_order",
    type=click.Choice(["degrevlex", "lex"]),
    default="degrevlex",
    help="Monomial order: degrevlex (default) or lex (FGLM at load time).",
)
@click.option("--dryrun", is_flag=True, help="Reduced settings for a fast smoke test.")
def main(
    config_path: str | None,
    data_config_path: str | None,
    training_order: str,
    dryrun: bool,
) -> None:
    """Train a Transformer to compute Gröbner bases from polynomial generators."""
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.train.seed)

    data_cfg = None
    if training_order == "lex":
        data_path = Path(data_config_path) if data_config_path else config_dir(__file__) / "data.yaml"
        data_cfg = OmegaConf.load(data_path)

    success = run_training(cfg, data_cfg=data_cfg, training_order=training_order, dryrun=dryrun)
    print(f"Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
