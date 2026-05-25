"""Ablation: vary the model size for the parity task.

Usage: python train.py --model_size {small,medium,large}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from parity.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


MODEL_SIZES = {
    "small":  dict(d_model=64,  num_encoder_layers=2, num_decoder_layers=2,
                   encoder_ffn_dim=256, decoder_ffn_dim=256,
                   num_encoder_heads=4, num_decoder_heads=4),
    "medium": dict(d_model=128, num_encoder_layers=4, num_decoder_layers=4,
                   encoder_ffn_dim=512, decoder_ffn_dim=512,
                   num_encoder_heads=8, num_decoder_heads=8),
    "large":  dict(d_model=256, num_encoder_layers=6, num_decoder_layers=6,
                   encoder_ffn_dim=1024, decoder_ffn_dim=1024,
                   num_encoder_heads=8, num_decoder_heads=8),
}


@click.command()
@click.option("--model_size", type=click.Choice(list(MODEL_SIZES)), default="small")
@click.option("--dryrun", is_flag=True)
def main(model_size: str, dryrun: bool) -> None:
    cfg = OmegaConf.load(config_dir(__file__) / "train.yaml")
    overrides = MODEL_SIZES[model_size]
    cfg.model = OmegaConf.create({**OmegaConf.to_container(cfg.model), **overrides})
    cfg.train.save_dir = f"../outputs/results_{model_size}"
    cfg.train.wandb.name = f"ablation_{model_size}"

    set_seed(cfg.train.seed)
    success = run_training(cfg, dryrun=dryrun)
    print(f"[{model_size}] Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
