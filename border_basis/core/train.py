"""
Training runner for the border basis task.

Orchestrates the four CALT pipelines:
  1. IOPipeline      — load text data, tokenize with UnifiedLexer
  2. ModelPipeline   — build the Transformer
  3. TrainerPipeline — train (HuggingFace Trainer under the hood)
  4. evaluate_and_save_generation — decode predictions, compute success rate

Lexer format
------------
If `data.lexer_config` points to a file with C/E expanded vocab, the pipeline
automatically wires `ExpandedFormLoadPreprocessor` to transform raw polynomial
strings into "C1 E2 E0 + ..." format before tokenization.

Usage
-----
    from border_basis.core.train import run_training
    from shared.config import load_config

    cfg = load_config("configs/train.yaml")
    success_rate = run_training(cfg, data_cfg=load_config("configs/data.yaml"))
"""

import os

import sage.all  # noqa: F401  # initialise Sage before any submodule
from omegaconf import DictConfig, OmegaConf

from shared.calt_adapter import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    ModelPipeline,
    TextToSageLoadPreprocessor,
    TrainerPipeline,
    apply_dryrun_settings,
    build_io_pipeline_with_cache,
    build_ring_from_sampler,
    detect_lexer_format,
)


def build_load_preprocessor(cfg: DictConfig, data_cfg: DictConfig | None):
    """
    Build the load-time preprocessor chain. Used by run_training AND the offline
    preprocess.py so the two stay in sync.

    For border_basis the only load preprocessor is ExpandedFormLoadPreprocessor
    (when lexer_format == "expanded"). Otherwise None.

    Returns
    -------
    tuple (preprocessor_or_None, format_str)
    """
    lexer_format = detect_lexer_format(cfg.data.lexer_config)

    chain = []
    if lexer_format == "expanded":
        if data_cfg is None:
            raise ValueError("lexer format='expanded' requires data_cfg to rebuild the source ring.")
        R_src = build_ring_from_sampler(data_cfg.sampler)
        chain.append(TextToSageLoadPreprocessor(delimiter="|", ring=R_src))
        chain.append(ExpandedFormLoadPreprocessor(delimiter=" | "))

    # Wire optional user post-processing hooks (task 3). See
    # `shared/user_postprocessor.py` for details. Both slots are opt-in:
    # if neither postprocessor file exists, this branch is a no-op.
    from shared.user_postprocessor import (
        get_user_postprocessors,
        RawLineToTupleAdapter,
        UserPostProcessorAdapter,
    )
    user_hooks = get_user_postprocessors("border_basis")
    if user_hooks:
        if not chain:
            chain.append(RawLineToTupleAdapter())
        chain.append(UserPostProcessorAdapter(user_hooks))

    if not chain:
        return None, lexer_format
    if len(chain) == 1:
        return chain[0], lexer_format
    return ChainLoadPreprocessor(*chain), lexer_format


def run_training(cfg: DictConfig, data_cfg: DictConfig | None = None, dryrun: bool = False) -> float:
    """
    Run the full training pipeline for the border basis task.

    Parameters
    ----------
    cfg : DictConfig
        Loaded train.yaml.
    data_cfg : DictConfig | None
        Loaded data.yaml. Required when the lexer is in expanded format
        (we need to rebuild the source ring to call TextToSageLoadPreprocessor).
    dryrun : bool

    Returns
    -------
    float : exact-match success rate on the test set.
    """
    if dryrun:
        apply_dryrun_settings(cfg)

    save_dir = cfg.train.get("save_dir", cfg.train.get("output_dir", "./results"))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "train.yaml"))

    # Cache hierarchy (pretok → strings → raw) is handled by the CALT library.
    # `training_order` is hard-coded "border" for border_basis (there is no FGLM
    # variant) so we use that as the hash discriminator.
    from shared.user_postprocessor import user_postprocessor_hash_contribution

    hooks_hash = user_postprocessor_hash_contribution("border_basis")

    load_preprocessor, lexer_format = build_load_preprocessor(cfg, data_cfg)
    training_order = "border"  # constant for this task
    io_pipeline, cache_kind = build_io_pipeline_with_cache(
        cfg, data_cfg, training_order, lexer_format, load_preprocessor,
        extra_hash_bytes=hooks_hash,
    )
    print(f"[run_training] cache={cache_kind} (format={lexer_format})")

    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
