"""Train parity (toy) with a CUSTOM input embedding — a copy-and-adapt example.

This is the plug-and-play companion to DOCUMENTATION.md §11.7. It shows the one
thing you cannot express in YAML alone: defining your own embedding module and
registering it so the config can select it by name.

How it works
------------
1. We define a custom input-embedding module (`LayerNormTokenEmbedding` below).
2. We `register_input_embedding("layernorm_token", ...)` at module import time —
   i.e. BEFORE `run_training` builds the model. This is the key requirement: the
   registration must run before `ModelPipeline(...).build()` is called.
3. The config `configs/train_custom_embedding.yaml` selects it with
   `model.input_embedding_type: layernorm_token`.

To use your own embedding: replace `LayerNormTokenEmbedding` with your module,
pick a name, register it, and set that name in the config. The same pattern
works for positional embeddings via `register_positional_embedding`.

Run
---
    cd parity/experiments/toy/scripts
    python train_custom_embedding.py                 # uses train_custom_embedding.yaml
    python train_custom_embedding.py --dryrun        # quick smoke test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
import torch.nn as nn
from omegaconf import OmegaConf

from calt.models import register_input_embedding, register_positional_embedding
from parity.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


# --------------------------------------------------------------------------- #
# 1) Define a custom input embedding.                                          #
#    Contract: __call__(input_ids: (B, S) long) -> (B, S, d_model) float.      #
# --------------------------------------------------------------------------- #
class LayerNormTokenEmbedding(nn.Module):
    """A plain token table followed by LayerNorm (a common, cheap stabilizer)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        return self.norm(self.emb(input_ids))


# --------------------------------------------------------------------------- #
# 2) Register it (runs at import, before the model is built).                  #
#    The name must match `model.input_embedding_type` in the YAML.            #
# --------------------------------------------------------------------------- #
register_input_embedding(
    "layernorm_token",
    lambda vocab_size, d_model, **kw: LayerNormTokenEmbedding(vocab_size, d_model),
)

# Want a custom *positional* embedding too? Same idea — define a module taking
# (B, S, d_model) -> (B, S, d_model), register it, and set
# `model.use_positional_embedding: <name>` in the YAML. Example:
#
#   register_positional_embedding(
#       "my_pe", lambda d_model, max_len, **kw: MyPositional(d_model, max_len))
_ = register_positional_embedding  # imported for the snippet above; no-op here


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to train config YAML (defaults to ../configs/train_custom_embedding.yaml).",
)
@click.option("--dryrun", is_flag=True, help="Reduced settings for a fast smoke test.")
def main(config_path: str | None, dryrun: bool) -> None:
    """Train parity with a custom input embedding selected by config."""
    cfg_path = (
        Path(config_path)
        if config_path
        else config_dir(__file__) / "train_custom_embedding.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.train.seed)

    success = run_training(cfg, dryrun=dryrun)
    print(f"Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
