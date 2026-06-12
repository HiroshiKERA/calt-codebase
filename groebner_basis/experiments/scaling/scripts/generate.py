"""Generate the large-scale Gröbner basis dataset for the scaling experiment.

Reference: issac2026_experiments/groebner/generate_dataset.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler

from groebner_basis.core.generator import GroebnerGenerator
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option("--config_path", type=click.Path(exists=True), default=None)
def main(config_path: str | None) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "data.yaml"
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.dataset.root_seed)

    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    sampler = PolynomialSampler(**sampler_cfg)

    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True)
    problem_generator = GroebnerGenerator(sampler=sampler, **gen_cfg)

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=problem_generator,
    )
    pipeline.run()
    print("Dataset generation completed")


if __name__ == "__main__":
    main()
