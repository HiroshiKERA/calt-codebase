"""
CLI to manage user post-processing hooks (task 3).

Usage from the repo root
------------------------
    python shared/hooks.py list                  # show active vs available
    python shared/hooks.py enable base           # activate base hook
    python shared/hooks.py enable groebner       # activate per-task hook
    python shared/hooks.py disable base          # remove base hook
    python shared/hooks.py disable groebner      # remove per-task hook

Each `enable` copies the corresponding `.example` template to the live path.
Each `disable` deletes the live file. After both operations, run
`python <task>/.../preprocess.py --force` to rebuild caches.

Templates available (matches `get_user_postprocessors`):
    shared/base_postprocessor.py.example          → shared/base_postprocessor.py
    groebner_basis/core/postprocessor.py.example  → groebner_basis/core/postprocessor.py
    border_basis/core/postprocessor.py.example    → border_basis/core/postprocessor.py
    parity/core/postprocessor.py.example          → parity/core/postprocessor.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Repo root inferred from this file location
ROOT = Path(__file__).resolve().parent.parent

SLOTS = {
    # name → (live path relative to ROOT, template path relative to ROOT)
    "base": (
        "shared/base_postprocessor.py",
        "shared/base_postprocessor.py.example",
    ),
    "groebner": (
        "groebner_basis/core/postprocessor.py",
        "groebner_basis/core/postprocessor.py.example",
    ),
    "border": (
        "border_basis/core/postprocessor.py",
        "border_basis/core/postprocessor.py.example",
    ),
    "parity": (
        "parity/core/postprocessor.py",
        "parity/core/postprocessor.py.example",
    ),
}


def cmd_list() -> int:
    """Print a table of slot status: active / template-only / missing."""
    print(f"Repo root: {ROOT}\n")
    print(f"{'Slot':<12} {'Status':<22} {'File'}")
    print("-" * 70)
    for name, (live_rel, tmpl_rel) in SLOTS.items():
        live = ROOT / live_rel
        tmpl = ROOT / tmpl_rel
        if live.exists():
            status = "✓ ACTIVE"
            path = str(live_rel)
        elif tmpl.exists():
            status = "○ template only"
            path = str(tmpl_rel)
        else:
            status = "✗ MISSING"
            path = f"(expected at {live_rel})"
        print(f"{name:<12} {status:<22} {path}")
    print()
    print("Edit a live file (✓ ACTIVE) to change the hook behavior.")
    print("After enable/disable/edit, rebuild caches: python <task>/.../preprocess.py --force")
    return 0


def cmd_enable(name: str) -> int:
    if name not in SLOTS:
        print(f"Unknown slot: {name!r}. Valid: {list(SLOTS)}", file=sys.stderr)
        return 2
    live_rel, tmpl_rel = SLOTS[name]
    live = ROOT / live_rel
    tmpl = ROOT / tmpl_rel
    if live.exists():
        print(f"Slot {name!r} is already ACTIVE at {live_rel}")
        print(f"  → edit it directly, then `--force` your preprocess.py to rebuild caches.")
        return 0
    if not tmpl.exists():
        print(f"Template not found: {tmpl_rel}", file=sys.stderr)
        return 2
    shutil.copyfile(tmpl, live)
    print(f"✓ Enabled slot {name!r}: copied {tmpl_rel} → {live_rel}")
    print(f"  Now edit {live_rel} to write your transformation.")
    print(f"  Then rebuild caches: python <task>/.../preprocess.py --force")
    return 0


def cmd_disable(name: str) -> int:
    if name not in SLOTS:
        print(f"Unknown slot: {name!r}. Valid: {list(SLOTS)}", file=sys.stderr)
        return 2
    live_rel, _ = SLOTS[name]
    live = ROOT / live_rel
    if not live.exists():
        print(f"Slot {name!r} was not active (no file at {live_rel}). Nothing to do.")
        return 0
    live.unlink()
    print(f"✓ Disabled slot {name!r}: removed {live_rel}")
    print(f"  Caches were hashed with the hook source — they will be detected STALE on next run.")
    print(f"  Rebuild caches: python <task>/.../preprocess.py --force")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if cmd == "list":
        return cmd_list()
    if cmd == "enable":
        if len(argv) < 2:
            print("Usage: enable <slot>     (slots: base, groebner, border, parity)", file=sys.stderr)
            return 2
        return cmd_enable(argv[1])
    if cmd == "disable":
        if len(argv) < 2:
            print("Usage: disable <slot>    (slots: base, groebner, border, parity)", file=sys.stderr)
            return 2
        return cmd_disable(argv[1])
    print(f"Unknown command: {cmd!r}. Valid: list | enable | disable", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
