"""
Training runner for the Gröbner basis computation task.

Orchestrates the four CALT pipelines:
  1. IOPipeline      — load text data, tokenize with UnifiedLexer
  2. ModelPipeline   — build the Transformer
  3. TrainerPipeline — train (HuggingFace Trainer under the hood)
  4. evaluate_and_save_generation — decode predictions, compute success rate

For `training_order="lex"`, a ChainLoadPreprocessor (TextToSage → GroebnerLexOrder)
is wired into the IOPipeline; it rebuilds F over a lex ring and recomputes
the Gröbner basis at load time.

Reference
---------
Mirrors `issac2026_experiments/groebner/train.py` (HiroshiKERA/calt@experiment/issac2026).
"""

import os

import sage.all  # noqa: F401  # initialise Sage before any submodule
from omegaconf import DictConfig, OmegaConf
from sage.all import GF, QQ, RR, ZZ, PolynomialRing  # type: ignore

from calt.io import (
    ChainLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings

from .parser import GroebnerLexOrderPreprocessor


def _build_source_ring(data_cfg: DictConfig):
    """Reconstruct the source PolynomialRing used at generation time."""
    sampler_cfg = dict(OmegaConf.to_container(data_cfg.sampler, resolve=True))
    symbols = sampler_cfg.get("symbols", "x,y")
    field_str = sampler_cfg.get("field_str", "QQ")
    order = sampler_cfg.get("order", "degrevlex")

    if field_str == "QQ":
        field = QQ
    elif field_str == "RR":
        field = RR
    elif field_str == "ZZ":
        field = ZZ
    elif field_str.startswith("GF"):
        p = int(field_str[2:]) if field_str[2:].isdigit() else None
        if not p:
            raise ValueError(f"Unsupported field_str for GF: {field_str!r}")
        field = GF(p)
    else:
        raise ValueError(f"Unsupported field_str: {field_str!r}")

    names = [s.strip() for s in symbols.split(",")]
    return PolynomialRing(field, names, order=order)


def run_training(
    cfg: DictConfig,
    data_cfg: DictConfig | None = None,
    training_order: str = "degrevlex",
    dryrun: bool = False,
) -> float:
    """
    Run the full training pipeline for the Gröbner basis task.

    Parameters
    ----------
    cfg : DictConfig
        Loaded train.yaml (must contain 'model', 'train', 'data' sections).
    data_cfg : DictConfig | None
        Loaded data.yaml (required when training_order='lex' to rebuild the ring).
    training_order : "degrevlex" | "lex"
        'degrevlex' uses the stored basis as-is.
        'lex' converts F,G to lex order at load time and recomputes the basis.
    dryrun : bool
        Reduced settings for a fast smoke test.

    Returns
    -------
    float
        Exact-match success rate on the test set (between 0 and 1).
    """
    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    save_dir = save_dir.rstrip("/") + f"_{training_order}"
    cfg.train.save_dir = save_dir

    if (
        hasattr(cfg.train, "wandb")
        and hasattr(cfg.train.wandb, "name")
        and cfg.train.wandb.name
    ):
        cfg.train.wandb.name = f"{cfg.train.wandb.name}_{training_order}"

    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)

    if training_order == "lex":
        if data_cfg is None:
            raise ValueError("training_order='lex' requires data_cfg to rebuild the source ring.")
        R_src = _build_source_ring(data_cfg)
        text_to_sage = TextToSageLoadPreprocessor(delimiter="|", ring=R_src)
        lex_pre = GroebnerLexOrderPreprocessor(R_src, delimiter="|")
        io_pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(text_to_sage, lex_pre)

    io_dict = io_pipeline.build()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
