"""Train border basis model over GF(p).

Usage: python train.py --p 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from border_basis.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option("--p", type=int, default=None, help="Prime for GF(p).")
@click.option("--config_path", type=click.Path(exists=True), default=None)
@click.option("--dryrun", is_flag=True)
def main(p: int | None, config_path: str | None, dryrun: bool) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)

    if p is not None:
        cfg.data.train_dataset_path = f"../data/GF{p}/train_raw.txt"
        cfg.data.test_dataset_path = f"../data/GF{p}/test_raw.txt"
        cfg.train.save_dir = f"../outputs/results_GF{p}"
        if hasattr(cfg.train, "wandb") and hasattr(cfg.train.wandb, "name"):
            cfg.train.wandb.name = f"finite_field_GF{p}"

    set_seed(cfg.train.seed)
    success = run_training(cfg, dryrun=dryrun)
    print(f"Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
