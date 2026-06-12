"""
Path helpers for experiment scripts.

Each experiment script lives at:
    <task>/experiments/<exp_name>/scripts/<script>.py

The helpers below derive standard paths relative to any script file,
so scripts work correctly regardless of which directory you cd into.

Example usage in a script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

    from shared.paths import experiment_dir, data_dir, config_dir

    DATA   = data_dir(__file__)    # → <task>/experiments/<exp_name>/data/
    CONFIG = config_dir(__file__)  # → <task>/experiments/<exp_name>/configs/
    OUT    = output_dir(__file__)  # → <task>/experiments/<exp_name>/outputs/
"""

from pathlib import Path


def experiment_dir(script_file: str) -> Path:
    """Return the experiment directory (parent of the scripts/ folder)."""
    return Path(script_file).resolve().parent.parent


def data_dir(script_file: str) -> Path:
    """Return the data directory for the current experiment."""
    return experiment_dir(script_file) / "data"


def output_dir(script_file: str) -> Path:
    """Return the outputs directory for the current experiment."""
    return experiment_dir(script_file) / "outputs"


def config_dir(script_file: str) -> Path:
    """Return the configs directory for the current experiment."""
    return experiment_dir(script_file) / "configs"


def codebase_root(script_file: str) -> Path:
    """
    Return the calt-codebase-new root (4 levels above a script in
    <task>/experiments/<exp>/scripts/).
    """
    return Path(script_file).resolve().parents[4]
