"""Generate the parity scaling dataset.

Usage: python generate.py --n 7
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
@click.option("--n", type=int, default=None, help="Permutation length (overrides config).")
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to data config YAML.",
)
def main(n: int | None, config_path: str | None) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "data.yaml"
    cfg = OmegaConf.load(cfg_path)
    if n is not None:
        cfg.problem_generator.n = n
    cfg.dataset.save_dir = f"../data_n{cfg.problem_generator.n}"

    set_seed(cfg.dataset.root_seed)

    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True)
    problem_generator = ParityGenerator(**gen_cfg)

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=problem_generator,
    )
    pipeline.run()
    print(f"Dataset generation completed (n={cfg.problem_generator.n})")


if __name__ == "__main__":
    main()
