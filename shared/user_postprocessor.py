"""
User-defined post-processing hooks (task 3).

Two optional slots, applied AFTER each task's load-time chain has produced
`(input_text, target_text)`. Both are pure-Python functions the user can write.

    1. Base hook   — file `shared/base_postprocessor.py`
                     Applied to ALL tasks that wire postprocessing in their
                     `build_load_preprocessor`.

    2. Per-task hook — file `<task>/core/postprocessor.py`
                       Applied only to that specific task, AFTER the base hook.

Each hook must expose a function with the exact signature:

    def postprocess(input_text: str, target_text: str) -> tuple[str, str]:
        return input_text, target_text   # default: identity

Both files are OPT-IN: if the file does not exist, the corresponding hook is
skipped and the chain stays exactly as it was before task 3 — no behavior
change, no cache invalidation.

Hash sensitivity
----------------
When a hook is wired, its source code is included in the preprocessing cache
hash, so any edit to a hook function invalidates the cache (preventing stale
data from being reused with mismatched preprocessing).

Usage from a task's `build_load_preprocessor`
---------------------------------------------
    from shared.user_postprocessor import (
        get_user_postprocessors,
        UserPostProcessorAdapter,
        RawLineToTupleAdapter,
    )

    funcs = get_user_postprocessors(task_pkg="groebner_basis")
    if funcs:
        # If task's own chain is empty, ensure we parse raw line first
        if not chain:
            chain.append(RawLineToTupleAdapter())
        chain.append(UserPostProcessorAdapter(funcs))
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import linecache
from pathlib import Path
from typing import Callable

PostFn = Callable[[str, str], tuple[str, str]]


# --------------------------------------------------------------------------- #
# Hook loading                                                                 #
# --------------------------------------------------------------------------- #

def _try_import_postprocess(module_name: str) -> PostFn | None:
    """Import `<module_name>.postprocess` if it exists; return None otherwise.

    Raises a clear error if the file exists but the function is malformed
    (wrong name, wrong signature, wrong return type) instead of silently
    skipping the hook.
    """
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    fn = getattr(mod, "postprocess", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"Hook file `{module_name}` exists but does not expose a callable "
            f"named `postprocess`.\n"
            f"  Required signature:\n"
            f"      def postprocess(input_text: str, target_text: str) -> tuple[str, str]:\n"
            f"          return input_text, target_text\n"
            f"  Did you typo the function name? It MUST be exactly `postprocess`."
        )

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn  # builtin/unanalyzable; trust the user.
    params = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) != 2:
        raise TypeError(
            f"Hook `{module_name}.postprocess` must take exactly 2 positional "
            f"arguments `(input_text, target_text)`, got {len(params)}: "
            f"{[p.name for p in params]}.\n"
            f"  Required signature:\n"
            f"      def postprocess(input_text: str, target_text: str) -> tuple[str, str]:"
        )

    # Dummy call: catches runtime errors + validates return type early.
    try:
        result = fn("dummy_input", "dummy_target")
    except Exception as e:
        raise RuntimeError(
            f"Hook `{module_name}.postprocess` crashed on dummy input "
            f"('dummy_input', 'dummy_target'):\n"
            f"  {type(e).__name__}: {e}\n"
            f"  Make sure your function handles any (str, str) input without crashing."
        ) from e
    if not (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], str)
        and isinstance(result[1], str)
    ):
        raise TypeError(
            f"Hook `{module_name}.postprocess` must return `tuple[str, str]`, "
            f"got {type(result).__name__} = {result!r}.\n"
            f"  Fix the return statement:\n"
            f"      return input_text, target_text   # both strings, in a 2-tuple"
        )
    return fn


def get_user_postprocessors(task_pkg: str) -> list[PostFn]:
    """
    Return the list of postprocess hooks to apply, in execution order.

    Order: [base, task_specific]. Empty list if neither file exists.

    Parameters
    ----------
    task_pkg : str
        The Python package name of the task (e.g. "groebner_basis").
        Looks for `<task_pkg>.core.postprocessor.postprocess`.
    """
    hooks: list[PostFn] = []
    base = _try_import_postprocess("shared.base_postprocessor")
    if base is not None:
        hooks.append(base)
    task = _try_import_postprocess(f"{task_pkg}.core.postprocessor")
    if task is not None:
        hooks.append(task)
    return hooks


# --------------------------------------------------------------------------- #
# Cache-hash contribution                                                      #
# --------------------------------------------------------------------------- #

def user_postprocessor_hash_contribution(task_pkg: str) -> bytes:
    """
    Return the bytes that should be folded into the preprocessing cache hash.

    For each hook present, includes the FULL source of its module. This way
    edits to a hook function invalidate the cache; absent hooks contribute
    empty bytes (no behavior change for users who never write a hook).
    """
    h = hashlib.sha256()
    for module_name in ("shared.base_postprocessor", f"{task_pkg}.core.postprocessor"):
        try:
            mod = importlib.import_module(module_name)
            # Invalidate linecache for this file so inspect.getsource picks up
            # post-edit content rather than returning a stale cached source.
            src_file = inspect.getsourcefile(mod)
            if src_file:
                linecache.checkcache(src_file)
            src = inspect.getsource(mod)
            h.update(module_name.encode())
            h.update(b"\x00")
            h.update(src.encode())
            h.update(b"\x00")
        except (ModuleNotFoundError, OSError, TypeError):
            # Module absent OR source unavailable → contribute nothing for this slot
            continue
    return h.digest()


# --------------------------------------------------------------------------- #
# Adapters that plug into CALT's ChainLoadPreprocessor                         #
# --------------------------------------------------------------------------- #

class RawLineToTupleAdapter:
    """
    Splits a raw `"input # target"` line into `(input, target)`.

    Used as the FIRST step of the chain when a task has no other load
    preprocessor (e.g. groebner degrevlex+raw) but the user wired hooks.
    CALT's chain expects the first preprocessor to handle the raw source.

    Same parsing as `calt.io.read.read_data_from_file` (single `#` split).
    """

    def process_sample(self, source):
        if isinstance(source, str):
            line = source.strip()
            if "#" not in line:
                raise ValueError(
                    f"Raw line must contain '#' separating input and target: {line[:80]!r}..."
                )
            input_part, target_part = line.split("#", 1)
            return input_part.strip(), target_part.strip()
        # Already a tuple/dict from an earlier preprocessor — pass through unchanged.
        return source


class UserPostProcessorAdapter:
    """
    Wraps user hooks into a CALT-compatible LoadPreprocessor.

    Receives `(input_text, target_text)` from the previous step, applies each
    user function in order, returns `(input_text, target_text)`.
    """

    def __init__(self, hooks: list[PostFn]):
        if not hooks:
            raise ValueError("UserPostProcessorAdapter requires at least one hook")
        self.hooks = list(hooks)

    def process_sample(self, source):
        if isinstance(source, tuple) and len(source) == 2:
            inp, tgt = source
        elif isinstance(source, dict):
            # Should be unusual at this stage; convert to tuple for hooks.
            inp = source.get("problem")
            tgt = source.get("answer", source.get("solution"))
        else:
            raise TypeError(
                f"UserPostProcessorAdapter expects (str, str) tuple or dict, got {type(source).__name__}"
            )
        for fn in self.hooks:
            inp, tgt = fn(inp, tgt)
        return inp, tgt


__all__ = [
    "PostFn",
    "get_user_postprocessors",
    "user_postprocessor_hash_contribution",
    "RawLineToTupleAdapter",
    "UserPostProcessorAdapter",
]
