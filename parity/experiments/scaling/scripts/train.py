"""Train the parity model at a chosen permutation length.

Usage: python train.py --n 7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from parity.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option("--n", type=int, default=None, help="Permutation length to train on.")
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to train config YAML.",
)
@click.option("--dryrun", is_flag=True, help="Reduced settings for a fast smoke test.")
def main(n: int | None, config_path: str | None, dryrun: bool) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)

    if n is not None:
        cfg.data.train_dataset_path = f"../data_n{n}/train_raw.txt"
        cfg.data.test_dataset_path = f"../data_n{n}/test_raw.txt"
        cfg.train.save_dir = f"../outputs/results_n{n}"
        if hasattr(cfg.train, "wandb") and hasattr(cfg.train.wandb, "name"):
            cfg.train.wandb.name = f"scaling_n{n}"

    set_seed(cfg.train.seed)
    success = run_training(cfg, dryrun=dryrun)
    print(f"Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
