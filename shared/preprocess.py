"""
Re-export shim for `calt.preprocess`.

The canonical implementation now lives in the CALT library
(`calt/io/preprocess.py`, formerly `calt/preprocess.py`) so that any project
building on CALT can benefit from offline preprocessing without re-implementing
it.

This file is kept only for backwards compatibility with the small number of
imports that already write `from shared.preprocess import ...`. New code should
import from `calt.io.preprocess` directly (or from `shared.calt_adapter`, which
shims both the new and old module locations).

If you ever `pip install --upgrade calt-x` and the `calt/preprocess.py` file
disappears from site-packages, restore it from this project's backup:

    /data/t-maxime/backups/calt-sitepackages_before-preprocess_20260526.tar.gz

or copy the previous version of this file (see git history) which contained
the full implementation inline.
"""

try:  # new calt-x: module relocated under calt.io
    from calt.io.preprocess import (  # noqa: F401
        PREPROCESS_VERSION,
        compute_config_hash,
        get_cache_dir,
        maybe_use_processed_cache,
        preprocess_to_ids,
        run_preprocess,
    )
except ImportError:  # old calt-x: module at top level
    from calt.preprocess import (  # noqa: F401
        PREPROCESS_VERSION,
        compute_config_hash,
        get_cache_dir,
        maybe_use_processed_cache,
        preprocess_to_ids,
        run_preprocess,
    )

__all__ = [
    "PREPROCESS_VERSION",
    "compute_config_hash",
    "get_cache_dir",
    "maybe_use_processed_cache",
    "preprocess_to_ids",
    "run_preprocess",
]
