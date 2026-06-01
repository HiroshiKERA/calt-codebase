"""Materialize a preprocessing cache for the toy border basis experiment.

Mirrors groebner_basis/experiments/toy/scripts/preprocess.py — see that file
for the rationale. By default builds a *pre-tokenized* cache so that training
performs zero tokenization at runtime.

Usage
-----
    cd border_basis/experiments/toy/scripts
    python preprocess.py                     # pretokenized cache (default)
    python preprocess.py --no-pretokenize    # strings cache (skip SageMath only)
    python preprocess.py --force             # rebuild even if hash matches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from border_basis.core.train import build_load_preprocessor
from shared.calt_adapter import preprocess_to_ids, run_preprocess
from shared.paths import config_dir
from shared.user_postprocessor import user_postprocessor_hash_contribution


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
    help="Path to data config YAML (defaults to ../configs/data.yaml).",
)
@click.option("--force", is_flag=True, help="Rebuild even if hash matches.")
@click.option(
    "--pretokenize/--no-pretokenize",
    default=True,
    help="Pre-tokenize all the way to input_ids (default ON).",
)
def main(
    config_path: str | None,
    data_config_path: str | None,
    force: bool,
    pretokenize: bool,
) -> None:
    """Preprocess raw .txt → JSONL cache for fast training startup."""
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)

    data_path = Path(data_config_path) if data_config_path else config_dir(__file__) / "data.yaml"
    data_cfg = OmegaConf.load(data_path) if data_path.exists() else None

    load_preprocessor, lexer_format = build_load_preprocessor(cfg, data_cfg)
    training_order = "border"  # constant for this task
    hooks_hash = user_postprocessor_hash_contribution("border_basis")

    if pretokenize:
        cache_dir = preprocess_to_ids(
            cfg, data_cfg, training_order, lexer_format, load_preprocessor,
            force=force, extra_hash_bytes=hooks_hash,
        )
    else:
        cache_dir = run_preprocess(
            cfg, data_cfg, training_order, lexer_format, load_preprocessor,
            force=force, extra_hash_bytes=hooks_hash,
        )
    print(f"[done] cache at: {cache_dir}")


if __name__ == "__main__":
    main()
