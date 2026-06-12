"""
Training runner for the parity task.

The parity task uses plain-text storage (no pickle), so no custom
load_preprocessor is needed — CALT's TextDefaultLoadPreprocessor handles it.
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
    Run the full training pipeline for the permutation parity task.

    Parameters
    ----------
    cfg : DictConfig
        Config with 'model', 'train', 'data' sections.
    dryrun : bool
        If True, run for 1 epoch on a tiny subset.

    Returns
    -------
    float
        Exact-match success rate on the test set.
    """
    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    # No custom preprocessor: text format is handled automatically
    io_dict = IOPipeline.from_config(cfg.data).build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
