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
    ExpandedFormLoadPreprocessor,
    IOPipeline,
    TextToSageLoadPreprocessor,
)
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline, apply_dryrun_settings

from shared.calt_adapter import detect_lexer_format

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


def build_load_preprocessor(
    cfg: DictConfig,
    data_cfg: DictConfig | None,
    training_order: str = "degrevlex",
):
    """
    Build the load-time preprocessor chain that converts raw text data into the
    string form expected by the tokenizer. Used by both run_training() and the
    offline preprocess.py script so the two stay in sync.

    Chain composition:
      training_order=lex   → TextToSage + GroebnerLexOrder
      lexer format=expanded → TextToSage + ExpandedForm (appended)

    Returns
    -------
    tuple (preprocessor_or_None, format_str)
        preprocessor is None when no preprocessing is needed (raw text + degrevlex).
        format_str is "raw" or "expanded" for logging.
    """
    lexer_format = detect_lexer_format(cfg.data.lexer_config)
    needs_ring = (training_order == "lex") or (lexer_format == "expanded")
    R_src = _build_source_ring(data_cfg) if (needs_ring and data_cfg is not None) else None
    if needs_ring and R_src is None:
        raise ValueError(
            "Either training_order='lex' or lexer format='expanded' requires data_cfg "
            "to rebuild the source ring."
        )

    chain = []
    if training_order == "lex":
        chain.append(TextToSageLoadPreprocessor(delimiter="|", ring=R_src))
        chain.append(GroebnerLexOrderPreprocessor(R_src, delimiter="|"))
    if lexer_format == "expanded":
        if not chain:
            chain.append(TextToSageLoadPreprocessor(delimiter="|", ring=R_src))
        chain.append(ExpandedFormLoadPreprocessor(delimiter=" | "))

    # Wire optional user post-processing hooks (task 3).
    # - shared/base_postprocessor.py   (applies to all tasks)
    # - groebner_basis/core/postprocessor.py (per-task, runs after base)
    # Both are opt-in: if neither file exists, this branch is a no-op.
    from shared.user_postprocessor import (
        get_user_postprocessors,
        RawLineToTupleAdapter,
        UserPostProcessorAdapter,
    )
    user_hooks = get_user_postprocessors("groebner_basis")
    if user_hooks:
        # If no task chain step has run yet, parse the raw line first so the
        # hooks receive a (str, str) tuple.
        if not chain:
            chain.append(RawLineToTupleAdapter())
        chain.append(UserPostProcessorAdapter(user_hooks))

    if not chain:
        return None, lexer_format
    if len(chain) == 1:
        return chain[0], lexer_format
    return ChainLoadPreprocessor(*chain), lexer_format


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

    # ---- Cache hierarchy ----
    # 1. Pre-tokenized cache (input_ids on disk) → ZERO online tokenization.
    # 2. Strings cache (post-load-preprocessor strings) → tokenization still online,
    #    but SageMath/FGLM/Expanded skipped.
    # 3. Raw .txt + load_preprocessor (original CALT behavior).
    from calt.preprocess import (
        maybe_use_pretokenized_cache,
        maybe_use_processed_cache,
    )
    from shared.user_postprocessor import user_postprocessor_hash_contribution
    hooks_hash = user_postprocessor_hash_contribution("groebner_basis")

    load_preprocessor, lexer_format = build_load_preprocessor(cfg, data_cfg, training_order)

    used_pretok = maybe_use_pretokenized_cache(cfg, data_cfg, training_order, lexer_format, extra_hash_bytes=hooks_hash)
    if used_pretok:
        print(f"[run_training] using PRE-TOKENIZED cache (format={lexer_format}, order={training_order})")
        io_pipeline = IOPipeline.from_config(cfg.data)
    else:
        used_strings = maybe_use_processed_cache(cfg, data_cfg, training_order, lexer_format, extra_hash_bytes=hooks_hash)
        if used_strings:
            print(f"[run_training] using strings cache (format={lexer_format}, order={training_order})")
            io_pipeline = IOPipeline.from_config(cfg.data)
        else:
            io_pipeline = IOPipeline.from_config(cfg.data)
            if load_preprocessor is not None:
                io_pipeline.dataset_load_preprocessor = load_preprocessor
            print(f"[run_training] no cache; lexer format={lexer_format}, training_order={training_order}")

    io_dict = io_pipeline.build()

    model = ModelPipeline.from_io_dict(cfg.model, io_dict).build()
    trainer_pipeline = TrainerPipeline.from_io_dict(cfg.train, model, io_dict).build()

    trainer_pipeline.train()
    trainer_pipeline.save_model()
    return trainer_pipeline.evaluate_and_save_generation()
