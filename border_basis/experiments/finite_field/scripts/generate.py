"""Generate border basis data over GF(p), p selectable via --p.

Usage: python generate.py --p 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from calt.dataset import DatasetPipeline
from calt.dataset.sagemath.utils.polynomial_sampler import PolynomialSampler

from border_basis.core.generator import BorderBasisGenerator
from shared.paths import config_dir
from shared.seed import set_seed


@click.command()
@click.option("--p", type=int, default=None, help="Prime for GF(p).")
@click.option("--config_path", type=click.Path(exists=True), default=None)
def main(p: int | None, config_path: str | None) -> None:
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "data.yaml"
    cfg = OmegaConf.load(cfg_path)

    if p is not None:
        cfg.sampler.field_str = f"GF{p}"
        cfg.dataset.save_dir = f"../data/GF{p}"

    set_seed(cfg.dataset.root_seed)

    sampler_cfg = dict(OmegaConf.to_container(cfg.sampler, resolve=True))
    sampler = PolynomialSampler(**sampler_cfg)

    gen_cfg = OmegaConf.to_container(cfg.problem_generator, resolve=True)
    problem_generator = BorderBasisGenerator(sampler=sampler, **gen_cfg)

    pipeline = DatasetPipeline.from_config(
        cfg.dataset,
        instance_generator=problem_generator,
    )
    pipeline.run()
    print(f"Dataset generation completed (GF{p or cfg.sampler.field_str[2:]})")


if __name__ == "__main__":
    main()
