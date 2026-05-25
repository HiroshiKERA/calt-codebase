"""
Training runner for [TASK NAME].

Copy this file to your task's core/train.py and adjust:
  1. The import of the custom parser (or remove it if not needed).
  2. The run_training signature if your task has extra options.
"""

import os

from omegaconf import DictConfig, OmegaConf

from shared.calt_adapter import (
    IOPipeline,
    ModelPipeline,
    TrainerPipeline,
    apply_dryrun_settings,
)

from .parser import TaskParser


def run_training(cfg: DictConfig, dryrun: bool = False) -> float:
    """
    Run the full training pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Config with 'model', 'train', 'data' sections.
    dryrun : bool
        Run 1 epoch on a tiny subset for a smoke test.

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

    io_pipeline = IOPipeline.from_config(cfg.data)
    io_pipeline.dataset_load_preprocessor = TaskParser()  # remove if using plain text
    io_dict = io_pipeline.build()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
