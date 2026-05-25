"""Generate the parity dataset for the toy experiment.

Reference: class-based instance_generator pattern from
issac2026_experiments/digit_product/generate_dataset.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline

from parity.core.generator import ParityGenerator
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

    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True)
    problem_generator = ParityGenerator(**gen_cfg)

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=problem_generator,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
