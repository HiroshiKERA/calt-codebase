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
        detect_lexer_format,
        ExpandedFormLoadPreprocessor,
        ChainLoadPreprocessor,
    )
"""

import yaml
from pathlib import Path

from calt.dataset import DatasetPipeline
from calt.io import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings

__all__ = [
    "DatasetPipeline",
    "IOPipeline",
    "ModelPipeline",
    "TrainerPipeline",
    "apply_dryrun_settings",
    # Load preprocessors
    "ChainLoadPreprocessor",
    "ExpandedFormLoadPreprocessor",
    "TextToSageLoadPreprocessor",
    # Helpers
    "detect_lexer_format",
]


def detect_lexer_format(lexer_yaml_path: str | Path) -> str:
    """
    Detect whether a lexer.yaml uses the 'raw' or 'expanded' polynomial format.

    Returns
    -------
    "expanded"  if vocab.range has BOTH `coefficients` and `exponents` keys
                 (e.g. `coefficients: ["C", -99, 99]`, `exponents: ["E", 0, 5]`)
    "raw"       otherwise (default: vocab.range has just `numbers`)

    Background
    ----------
    Polynomials can be tokenized in two ways (see paper §2.2):
      - raw      : text-direct, e.g. "x ^ 2 + y" (used by ISSAC2026/groebner)
      - expanded : C/E form,   e.g. "C1 E2 E0 + C1 E0 E1" (used by ISSAC2026/polynomial_reduction)

    The vocabulary required differs between the two formats. By inspecting the
    `vocab.range` keys, we can detect which format the user wants:

        vocab.range.numbers      → raw text format
        vocab.range.coefficients → C/E expanded format
        vocab.range.exponents    → C/E expanded format

    """
    with open(lexer_yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    rng = (cfg.get("vocab") or {}).get("range") or {}
    if "coefficients" in rng and "exponents" in rng:
        return "expanded"
    return "raw"


def maybe_wrap_with_expanded_preprocessor(
    io_pipeline,
    lexer_yaml_path: str | Path,
    *,
    delimiter: str = "|",
    ring=None,
):
    """
    If the lexer is in C/E expanded format, wire `ExpandedFormLoadPreprocessor`
    into the io_pipeline's load chain. Otherwise leave the pipeline unchanged.

    The chain becomes:
        [existing dataset_load_preprocessor if any]
        → TextToSageLoadPreprocessor (raw text → Sage polynomials)
        → ExpandedFormLoadPreprocessor (Sage polys → "C1 E2 E0" text)

    Parameters
    ----------
    io_pipeline : IOPipeline
        Mutated in place if expanded format is detected.
    lexer_yaml_path : str | Path
        Path to the lexer.yaml whose format determines the behavior.
    delimiter : str
        Inner separator between polynomials (default "|").
    ring : sage PolynomialRing or None
        Required for TextToSageLoadPreprocessor when format is expanded.
        Must be provided by the caller (different rings per task).

    Returns
    -------
    str : the detected format ("raw" or "expanded"), for logging.
    """
    fmt = detect_lexer_format(lexer_yaml_path)
    if fmt != "expanded":
        return fmt
    if ring is None:
        raise ValueError(
            "Expanded format requires a `ring` argument to TextToSageLoadPreprocessor. "
            "Pass the SageMath PolynomialRing matching your data."
        )
    text_to_sage = TextToSageLoadPreprocessor(delimiter=delimiter, ring=ring)
    expanded_form = ExpandedFormLoadPreprocessor(delimiter=delimiter)
    existing = io_pipeline.dataset_load_preprocessor
    if existing is None:
        io_pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(text_to_sage, expanded_form)
    else:
        # Insert expanded_form at the end of the existing chain
        io_pipeline.dataset_load_preprocessor = ChainLoadPreprocessor(existing, expanded_form)
    return fmt


def run_standard_training(cfg, load_preprocessor=None, dryrun: bool = False) -> float:
    """
    Run a complete training pipeline: load data → build model → train → evaluate.

    Parameters
    ----------
    cfg : DictConfig
    load_preprocessor : object | None
        Optional load-time preprocessor (must implement process_sample).
    dryrun : bool

    Returns
    -------
    float : exact-match success rate on the test set.
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
