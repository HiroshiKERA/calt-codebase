"""
Thin wrapper around the four CALT pipelines.

This module re-exports the CALT API in a single place so that task scripts
only need one import instead of four, and so that future CALT API changes
only require updating this file.

The four CALT pipelines:
    DatasetPipeline  →  generate (problem, answer) pairs and write to disk
    IOPipeline       →  load data, apply preprocessors, tokenize
    ModelPipeline    →  build the Transformer
    TrainerPipeline  →  train + evaluate

Typical usage in a task's core/train.py
----------------------------------------
    from shared.calt_adapter import (
        DatasetPipeline,
        IOPipeline,
        ModelPipeline,
        TrainerPipeline,
        apply_dryrun_settings,
    )
"""

from calt.dataset import DatasetPipeline
from calt.io import IOPipeline
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings

__all__ = [
    "DatasetPipeline",
    "IOPipeline",
    "ModelPipeline",
    "TrainerPipeline",
    "apply_dryrun_settings",
]


def run_standard_training(cfg, load_preprocessor=None, dryrun: bool = False) -> float:
    """
    Run a complete training pipeline: load data → build model → train → evaluate.

    This covers the common case where no custom Trainer is needed.
    For tasks requiring a custom loss or metrics, call the pipelines directly
    (see core/train.py in each task).

    Parameters
    ----------
    cfg : DictConfig
        Config with 'data', 'model', and 'train' sections.
    load_preprocessor : object | None
        Optional load-time preprocessor (must implement process_sample).
        Required for tasks that store data as Python objects (e.g., SageMath
        polynomials saved as pickle).
    dryrun : bool
        If True, reduce epochs and data for a quick sanity check.

    Returns
    -------
    float
        Exact-match success rate on the test set.
    """
    import os
    from omegaconf import OmegaConf

    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    io_pipeline = IOPipeline.from_config(cfg.data)
    if load_preprocessor is not None:
        io_pipeline.dataset_load_preprocessor = load_preprocessor

    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
