"""Plotting and inspection utilities for experiment analysis notebooks.

Includes a `showcase()` helper that mirrors `calt-codebase/examples/demos/utils.py`
(the canonical helper used by the official `minimal_demo.ipynb`).

For raw access to the eval results, `load_eval_results` and `plot_success_rate`
delegate to `calt.io.visualization.comparison_vis.load_eval_results` — the
canonical loader for the eval JSON written by `evaluate_and_save_generation`.

The on-disk format is a JSON list of dicts:
    [{"generated": "...", "reference": "..."}, ...]
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from calt.io.visualization.comparison_vis import (
    load_eval_results as _calt_load_eval_results,
)


# ---------------------------------------------------------------------------
# Eval result loading
# ---------------------------------------------------------------------------

def _resolve_eval_results_path(
    results_dir: str | Path,
    eval_results_path: str | None = None,
) -> Path:
    """
    Resolve the path to an eval results JSON.

    Preference order:
      1. Explicit `eval_results_path` if given.
      2. ``<results_dir>/eval_results.json`` if present.
      3. The most recent ``step_*.json`` inside ``<results_dir>/eval_results/``.

    Raises FileNotFoundError with a helpful message otherwise.
    """
    if eval_results_path is not None:
        return Path(eval_results_path)

    results_dir = Path(results_dir)
    direct = results_dir / "eval_results.json"
    if direct.is_file():
        return direct

    step_dir = results_dir / "eval_results"
    if step_dir.is_dir():
        steps = sorted(
            step_dir.glob("step_*.json"),
            key=lambda p: int(p.stem.replace("step_", "")),
        )
        if steps:
            return steps[-1]

    raise FileNotFoundError(
        f"No eval_results.json or eval_results/step_*.json under {results_dir}. "
        "Run train.py first."
    )


def load_eval_results(results_dir: str | Path) -> tuple[list[str], list[str]]:
    """
    Load the latest evaluation results from a results directory.

    Returns (generated_texts, reference_texts).
    """
    path = _resolve_eval_results_path(results_dir)
    return _calt_load_eval_results(str(path))


# ---------------------------------------------------------------------------
# showcase: port of calt-codebase/examples/demos/utils.py::showcase
# ---------------------------------------------------------------------------

def showcase(
    dataset,
    success_cases: bool = True,
    num_show: int = 5,
    eval_results_path: str | None = None,
    results_dir: str = "results",
) -> None:
    """
    Print up to `num_show` success (or failure) cases from the latest eval results.

    Direct port of `calt-codebase/examples/demos/utils.py::showcase` so notebook
    output matches the official `minimal_demo.ipynb`.

    Parameters
    ----------
    dataset
        The test_dataset returned by `IOPipeline.build()`. Its `preprocessor`
        is used to decode the tokenized strings stored in eval results.
    success_cases : bool
        True → print correct predictions. False → print mistakes.
    num_show : int
        Maximum number of cases to print.
    eval_results_path : str | None
        Explicit path to an eval JSON. If None, auto-resolved from `results_dir`.
    results_dir : str
        Output directory of the training run (default "results").
    """
    if success_cases:
        indicator_fn = lambda gen, ref: gen == ref  # noqa: E731
        tag = "success"
    else:
        indicator_fn = lambda gen, ref: gen != ref  # noqa: E731
        tag = "failure"

    path = _resolve_eval_results_path(results_dir, eval_results_path)
    gen_texts, ref_texts = _calt_load_eval_results(str(path))
    cases = [
        (i, gen, ref)
        for i, (gen, ref) in enumerate(zip(gen_texts, ref_texts))
        if indicator_fn(gen, ref)
    ]

    print("-------------------------")
    print(f""" {tag} cases ({len(cases)} total) """)
    print("-------------------------")
    for i, gen, ref in cases[:num_show]:
        gen_expr = dataset.preprocessor.decode(gen)
        ref_expr = dataset.preprocessor.decode(ref)
        print(f"  [{i}] gen: {gen_expr}  |  ref: {ref_expr}")


# ---------------------------------------------------------------------------
# show_examples: print n success or failure cases from pre-loaded lists
# ---------------------------------------------------------------------------

def show_examples(
    generated: list[str],
    references: list[str],
    n: int = 5,
    successes: bool = True,
) -> None:
    """
    Print up to `n` success (or failure) cases from pre-loaded result lists.

    Parameters
    ----------
    generated : list[str]
        Model predictions (as returned by load_eval_results).
    references : list[str]
        Ground-truth strings.
    n : int
        Maximum number of cases to print.
    successes : bool
        True → print correct predictions. False → print mistakes.
    """
    tag = "success" if successes else "failure"
    cases = [
        (i, gen, ref)
        for i, (gen, ref) in enumerate(zip(generated, references))
        if (gen == ref) == successes
    ]
    print("-------------------------")
    print(f" {tag} cases ({len(cases)} total) ")
    print("-------------------------")
    for i, gen, ref in cases[:n]:
        print(f"  [{i}] predicted : {gen}")
        print(f"       reference : {ref}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_success_rate(results_dir: str | Path, ax=None) -> plt.Axes:
    """
    Plot exact-match success rate over training steps.

    Reads all ``step_*.json`` files in ``<results_dir>/eval_results/`` and
    plots success rate vs step.
    """
    results_dir = Path(results_dir)
    step_dir = results_dir / "eval_results"
    if not step_dir.is_dir():
        raise FileNotFoundError(f"No eval_results/ directory in {results_dir}")

    steps, rates = [], []
    for p in sorted(step_dir.glob("step_*.json"), key=lambda p: int(p.stem.replace("step_", ""))):
        step = int(p.stem.replace("step_", ""))
        with open(p) as f:
            data = json.load(f)
        if not data:
            continue
        gen = [item.get("generated", "") for item in data]
        ref = [item.get("reference", "") for item in data]
        rate = sum(g == r for g, r in zip(gen, ref)) / len(ref)
        steps.append(step)
        rates.append(rate * 100)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rates, marker="o", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Exact-match success rate")
    ax.grid(True, alpha=0.3)
    return ax
