"""
Training runner for the parity task.

The parity task stores plain text ("input # output" lines), so it needs no
load preprocessor of its own — CALT's TextDefaultLoadPreprocessor handles it.
The optional user post-processing hooks (shared/base_postprocessor.py,
parity/core/postprocessor.py) are applied when present, as in the other tasks.
"""

import os

from omegaconf import DictConfig, OmegaConf

from shared.calt_adapter import (
    ChainLoadPreprocessor,
    IOPipeline,
    ModelPipeline,
    TrainerPipeline,
    apply_dryrun_settings,
)


def build_load_preprocessor():
    """Return the load preprocessor for the user hooks, or None without hooks.

    See `shared/user_postprocessor.py`. Both hook files are opt-in: if neither
    exists, CALT reads the text lines as before.
    """
    from shared.user_postprocessor import (
        RawLineToTupleAdapter,
        UserPostProcessorAdapter,
        get_user_postprocessors,
    )

    user_hooks = get_user_postprocessors("parity")
    if not user_hooks:
        return None
    return ChainLoadPreprocessor(RawLineToTupleAdapter(), UserPostProcessorAdapter(user_hooks))


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

    io_pipeline = IOPipeline.from_config(cfg.data)
    load_preprocessor = build_load_preprocessor()
    if load_preprocessor is not None:
        io_pipeline.dataset_load_preprocessor = load_preprocessor
    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
