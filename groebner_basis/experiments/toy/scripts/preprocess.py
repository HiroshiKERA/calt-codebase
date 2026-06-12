"""Materialize a JSONL preprocessing cache for the toy Gröbner experiment.

Runs the load-time chain (TextToSage / FGLM-lex / ExpandedForm — whichever applies
to your training_order + lexer.yaml) ONCE over train_raw.txt + test_raw.txt and
writes the result to `<data>/processed_<order>_<format>/`. Training will detect
the cache automatically and skip the SageMath/FGLM step on subsequent runs.

Re-run this script whenever you change lexer.yaml, data.yaml's sampler, or the
--training_order. The cache is hash-keyed so a stale cache is detected and
ignored (train.py falls back to raw + on-the-fly preprocessing).

Usage
-----
    cd groebner_basis/experiments/toy/scripts
    python preprocess.py                          # degrevlex + lexer from train.yaml
    python preprocess.py --training_order lex     # build the lex cache
    python preprocess.py --force                  # rebuild even if hash matches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from groebner_basis.core.train import build_load_preprocessor
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
@click.option(
    "--training_order",
    type=click.Choice(["degrevlex", "lex"]),
    default="degrevlex",
    help="Monomial order: must match what you'll pass to train.py.",
)
@click.option("--force", is_flag=True, help="Rebuild even if hash matches.")
@click.option(
    "--pretokenize/--no-pretokenize",
    default=True,
    help="Pre-tokenize all the way to input_ids (default ON). "
         "OFF stores post-load-preprocessor strings only; training still tokenizes online.",
)
def main(
    config_path: str | None,
    data_config_path: str | None,
    training_order: str,
    force: bool,
    pretokenize: bool,
) -> None:
    """Preprocess raw .txt → JSONL cache for fast training startup."""
    cfg_path = Path(config_path) if config_path else config_dir(__file__) / "train.yaml"
    cfg = OmegaConf.load(cfg_path)

    data_path = Path(data_config_path) if data_config_path else config_dir(__file__) / "data.yaml"
    data_cfg = OmegaConf.load(data_path)

    load_preprocessor, lexer_format = build_load_preprocessor(cfg, data_cfg, training_order)
    hooks_hash = user_postprocessor_hash_contribution("groebner_basis")
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
