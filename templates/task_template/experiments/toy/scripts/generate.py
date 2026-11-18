"""Generate the dataset for the [TASK NAME] toy experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline

# TODO: replace 'task_template' with your task module name
from task_template.core.generator import TaskGenerator
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to data config YAML (defaults to ../configs/data.yaml).",
)
def main(config_path: str | None) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "data.yaml"
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.dataset.root_seed)

    # problem_generator in data.yaml is passed to TaskGenerator as keyword arguments.
    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True) or {}
    problem_generator = TaskGenerator(**gen_cfg)

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=problem_generator,
        # Once core/metrics.py:instance_stats is implemented, add
        # statistics_calculator=instance_stats to add per-sample statistics
        # to {split}_stats.yaml.
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
