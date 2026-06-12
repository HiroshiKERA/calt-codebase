"""Gröbner basis ablation: degrevlex vs lex training order.

Usage: python train.py --training_order {degrevlex,lex}
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
    "--training_order",
    type=click.Choice(["degrevlex", "lex"]),
    default="degrevlex",
)
@click.option("--dryrun", is_flag=True)
def main(training_order: str, dryrun: bool) -> None:
    cfg = OmegaConf.load(config_dir(__file__) / "train.yaml")

    data_cfg = None
    if training_order == "lex":
        # Reuse the toy data config so we know the source ring.
        data_cfg = OmegaConf.load(config_dir(__file__).parent.parent / "toy/configs/data.yaml")

    cfg.train.wandb.name = "ablation"
    set_seed(cfg.train.seed)
    success = run_training(cfg, data_cfg=data_cfg, training_order=training_order, dryrun=dryrun)
    print(f"[{training_order}] Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
