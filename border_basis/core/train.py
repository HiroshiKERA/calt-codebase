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
from sage.all import GF, QQ, RR, ZZ, PolynomialRing  # type: ignore

from shared.calt_adapter import (
    ChainLoadPreprocessor,
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    ModelPipeline,
    TextToSageLoadPreprocessor,
    TrainerPipeline,
    apply_dryrun_settings,
    detect_lexer_format,
)


def _build_source_ring(data_cfg: DictConfig):
    """Reconstruct the source PolynomialRing used at generation time."""
    sampler_cfg = dict(OmegaConf.to_container(data_cfg.sampler, resolve=True))
    symbols = sampler_cfg.get("symbols", "x,y")
    field_str = sampler_cfg.get("field_str", "GF7")
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
        R_src = _build_source_ring(data_cfg)
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

    # Cache hierarchy: pretok > strings > raw. Mirrors groebner_basis/core/train.py.
    # `training_order` is hard-coded "border" for border_basis (there is no FGLM
    # variant) so we use that as the hash discriminator.
    from calt.preprocess import (
        maybe_use_pretokenized_cache,
        maybe_use_processed_cache,
    )
    from shared.user_postprocessor import user_postprocessor_hash_contribution
    hooks_hash = user_postprocessor_hash_contribution("border_basis")

    load_preprocessor, lexer_format = build_load_preprocessor(cfg, data_cfg)
    training_order = "border"  # constant for this task

    used_pretok = maybe_use_pretokenized_cache(cfg, data_cfg, training_order, lexer_format, extra_hash_bytes=hooks_hash)
    if used_pretok:
        print(f"[run_training] using PRE-TOKENIZED cache (format={lexer_format})")
        io_pipeline = IOPipeline.from_config(cfg.data)
    else:
        used_strings = maybe_use_processed_cache(cfg, data_cfg, training_order, lexer_format, extra_hash_bytes=hooks_hash)
        if used_strings:
            print(f"[run_training] using strings cache (format={lexer_format})")
            io_pipeline = IOPipeline.from_config(cfg.data)
        else:
            io_pipeline = IOPipeline.from_config(cfg.data)
            if load_preprocessor is not None:
                io_pipeline.dataset_load_preprocessor = load_preprocessor
            print(f"[run_training] no cache; lexer format={lexer_format}")

    io_dict = io_pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
