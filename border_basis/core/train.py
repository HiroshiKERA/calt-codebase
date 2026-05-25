"""
Training runner for the border basis task.

Orchestrates the four CALT pipelines:
  1. IOPipeline      — load text data, tokenize with UnifiedLexer
  2. ModelPipeline   — build the Transformer
  3. TrainerPipeline — train (HuggingFace Trainer under the hood)
  4. evaluate_and_save_generation — decode predictions, compute success rate

Usage
-----
    from border_basis.core.train import run_training
    from shared.config import load_config

    cfg = load_config("configs/train.yaml")
    success_rate = run_training(cfg)
    print(f"Success rate: {100 * success_rate:.1f}%")
"""

import os

from omegaconf import DictConfig, OmegaConf

from shared.calt_adapter import (
    IOPipeline,
    ModelPipeline,
    TrainerPipeline,
    apply_dryrun_settings,
)


def run_training(cfg: DictConfig, dryrun: bool = False) -> float:
    """
    Run the full training pipeline for the border basis task.

    Parameters
    ----------
    cfg : DictConfig
        Config with 'model', 'train', 'data' sections.
    dryrun : bool
        If True, run for 1 epoch on a tiny subset.

    Returns
    -------
    float
        Exact-match success rate on the test set (between 0 and 1).
    """
    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)
    io_dict = io_pipeline.build()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
