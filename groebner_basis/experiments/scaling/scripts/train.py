"""Gröbner basis scaling: vary model size to study scaling behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import click
from omegaconf import OmegaConf

from groebner_basis.core.train import run_training
from shared.paths import config_dir
from shared.seed import set_seed


MODEL_SIZES = {
    "small":  dict(d_model=128, num_encoder_layers=2, num_decoder_layers=2,
                   encoder_ffn_dim=512, decoder_ffn_dim=512,
                   num_encoder_heads=4, num_decoder_heads=4),
    "medium": dict(d_model=256, num_encoder_layers=4, num_decoder_layers=4,
                   encoder_ffn_dim=1024, decoder_ffn_dim=1024,
                   num_encoder_heads=8, num_decoder_heads=8),
    "large":  dict(d_model=512, num_encoder_layers=6, num_decoder_layers=6,
                   encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                   num_encoder_heads=8, num_decoder_heads=8),
}


@click.command()
@click.option("--model_size", type=click.Choice(list(MODEL_SIZES)), default="large")
@click.option("--training_order", type=click.Choice(["degrevlex", "lex"]), default="degrevlex")
@click.option("--dryrun", is_flag=True)
def main(model_size: str, training_order: str, dryrun: bool) -> None:
    cfg = OmegaConf.load(config_dir(__file__) / "train.yaml")
    cfg.model = OmegaConf.create({**OmegaConf.to_container(cfg.model), **MODEL_SIZES[model_size]})
    cfg.train.wandb.name = f"scaling_{model_size}"

    data_cfg = None
    if training_order == "lex":
        data_cfg = OmegaConf.load(config_dir(__file__) / "data.yaml")

    set_seed(cfg.train.seed)
    success = run_training(cfg, data_cfg=data_cfg, training_order=training_order, dryrun=dryrun)
    print(f"[{model_size}/{training_order}] Success rate: {100 * success:.1f}%")


if __name__ == "__main__":
    main()
