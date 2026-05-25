"""OmegaConf helpers: load YAML configs with optional command-line overrides."""

from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def load_config(path: Union[str, Path], overrides: list[str] | None = None) -> DictConfig:
    """
    Load a YAML config file and optionally apply dot-notation overrides.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.
    overrides : list[str] | None
        List of OmegaConf overrides in "key=value" format.
        Example: ["train.learning_rate=0.001", "train.num_train_epochs=50"]

    Returns
    -------
    DictConfig
        The loaded (and optionally overridden) config.

    Example
    -------
    >>> cfg = load_config("configs/train.yaml", ["train.batch_size=32"])
    >>> print(cfg.train.learning_rate)
    """
    cfg = OmegaConf.load(path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def save_config(cfg: DictConfig, path: Union[str, Path]) -> None:
    """Save a config to a YAML file (useful for logging experiment settings)."""
    OmegaConf.save(cfg, path)
