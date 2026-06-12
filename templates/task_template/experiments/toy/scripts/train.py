"""Train model for [TASK NAME] toy experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click

# TODO: replace 'task_template' with your task module name
from task_template.core.train import run_training
from shared.config import load_config
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option("--dryrun", is_flag=True)
def main(dryrun: bool):
    cfg = load_config(config_dir(__file__) / "train.yaml")
    set_seed(cfg.train.seed)
    success = run_training(cfg, dryrun=dryrun)
    print(f"Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
