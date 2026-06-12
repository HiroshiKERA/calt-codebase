"""
CLI to inspect preprocessing caches.

Usage from the repo root
------------------------
    python shared/inspect_cache.py                                  # scan all data/ dirs
    python shared/inspect_cache.py groebner_basis/.../data/QQ       # one specific dir

For each cache directory found (`processed_*` or `processed_*_ids`), prints:
  - kind (strings / pretokenized)
  - training_order + lexer_format
  - sample counts (train / test)
  - creation time
  - hash digest (12 first chars)
  - tokenizer vocab size (for pretokenized only)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def inspect_cache_dir(cache_dir: Path) -> None:
    """Print a one-cache summary."""
    meta_path = cache_dir / "_meta.yaml"
    hash_path = cache_dir / "_hash.txt"
    is_pretok = "_ids" in cache_dir.name

    print(f"\n  {cache_dir.relative_to(ROOT) if cache_dir.is_relative_to(ROOT) else cache_dir}")
    print(f"    kind          : {'pretokenized (input_ids)' if is_pretok else 'strings (problem/answer)'}")

    if meta_path.exists():
        try:
            import yaml as _yaml
            meta = _yaml.safe_load(meta_path.read_text()) or {}
        except Exception:
            meta = {}
        order = meta.get("training_order", "?")
        fmt = meta.get("lexer_format", "?")
        created = meta.get("created_at", "?")
        n_train = meta.get("n_train", "?")
        n_test = meta.get("n_test", "?")
        load_pp = meta.get("load_preprocessor", "?")
        vocab = meta.get("tokenizer_vocab_size", None)
        print(f"    order/format  : {order} / {fmt}")
        print(f"    samples       : train={n_train}, test={n_test}")
        print(f"    created       : {created}")
        print(f"    load chain    : {load_pp}")
        if vocab is not None:
            print(f"    vocab size    : {vocab}")
    else:
        # Fallback: read JSONL line counts directly
        train_name = "train_ids.jsonl" if is_pretok else "train_processed.jsonl"
        test_name = "test_ids.jsonl" if is_pretok else "test_processed.jsonl"
        n_tr = _count_jsonl_lines(cache_dir / train_name)
        n_te = _count_jsonl_lines(cache_dir / test_name)
        print(f"    samples       : train={n_tr}, test={n_te} (no _meta.yaml)")

    if hash_path.exists():
        h = hash_path.read_text().strip()
        print(f"    hash          : {h[:12]}...")
    else:
        print(f"    hash          : MISSING")


def scan(root: Path) -> list[Path]:
    """Find all cache directories under root."""
    found = []
    for d in sorted(root.rglob("processed_*")):
        if d.is_dir():
            found.append(d)
    return found


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    if argv:
        target = Path(argv[0]).resolve()
        if not target.exists():
            print(f"Directory not found: {target}", file=sys.stderr)
            return 2
    else:
        target = ROOT

    caches = scan(target)
    if not caches:
        print(f"No cache directories found under {target}")
        print("Hint: run `python <task>/experiments/<exp>/scripts/preprocess.py` to build one.")
        return 0

    print(f"Found {len(caches)} cache director{'y' if len(caches)==1 else 'ies'} under {target}:")
    for d in caches:
        inspect_cache_dir(d)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
