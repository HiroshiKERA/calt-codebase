#!/bin/bash
# Run the full border_basis/toy experiment end-to-end.
#
# Usage:
#   bash run.sh                          # defaults: raw lexer, with pretok cache
#   bash run.sh --lexer expanded         # use C/E expanded vocab
#   bash run.sh --no-preprocess          # skip preprocess.py
#   bash run.sh --dryrun                 # fast smoke test
#   bash run.sh --force                  # rebuild caches even if hash matches

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python}"
if ! IMPORT_ERR="$("$PY" -c "import calt, sage" 2>&1)"; then
    echo "❌ \`$PY\` cannot import \`calt\` and \`sage\`:" >&2
    echo "   $(printf '%s\n' "$IMPORT_ERR" | tail -n 1)" >&2
    if printf '%s\n' "$IMPORT_ERR" | grep -q CXXABI; then
        echo "   This is the CXXABI error from pip's PyTorch in a conda env. Recreate calt-env" >&2
        echo "   with PyTorch from conda-forge (no pip torch), as the project README.md says:" >&2
        echo "     conda deactivate; conda env remove -n calt-env -y" >&2
        echo "     conda create -n calt-env -c conda-forge \"calt-x>=1.6.1\" \"pytorch>=2.6=cuda*\" sage matplotlib click sortedcontainers" >&2
        echo "     (no NVIDIA GPU / Mac: \"pytorch>=2.6\" instead), then:  conda activate calt-env" >&2
        exit 1
    fi
    echo "   Run:  conda activate calt-env   (or set PYTHON=/path/to/calt-env/bin/python)" >&2
    exit 1
fi

LEXER="raw"
PREPROCESS=1
DRYRUN=""
FORCE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --lexer)            LEXER="$2"; shift 2 ;;
        --no-preprocess)    PREPROCESS=0; shift ;;
        --dryrun)           DRYRUN="--dryrun"; shift ;;
        --force)            FORCE="--force"; shift ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

case $LEXER in
    raw|expanded) ;;
    *) echo "Invalid --lexer $LEXER (must be raw or expanded)" >&2; exit 2 ;;
esac

LEXER_PATH="../configs/lexer.yaml"
[[ "$LEXER" == "expanded" ]] && LEXER_PATH="../configs/lexer_expanded.yaml"

echo "================================================================="
echo "  border_basis/toy — end-to-end run"
echo "  lexer   : $LEXER  ($LEXER_PATH)"
echo "  preproc : $([ $PREPROCESS -eq 1 ] && echo enabled || echo skipped)"
echo "  dryrun  : $([ -n "$DRYRUN" ] && echo YES || echo no)"
echo "================================================================="

TMP_TRAIN_YAML="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_TRAIN_YAML"' EXIT
sed "s|lexer_config:.*|lexer_config: $LEXER_PATH|" ../configs/train.yaml > "$TMP_TRAIN_YAML"

if [[ ! -f ../data/GF7/train_raw.txt ]]; then
    echo "=== Step 1/3: Generate data ==="
    "$PY" generate.py
else
    echo "=== Step 1/3: Generate data (already exists, skipping) ==="
fi

if [[ $PREPROCESS -eq 1 ]]; then
    echo "=== Step 2/3: Preprocess (build cache) ==="
    "$PY" preprocess.py --config_path "$TMP_TRAIN_YAML" $FORCE
else
    echo "=== Step 2/3: Preprocess SKIPPED ==="
fi

echo "=== Step 3/3: Train + Evaluate ==="
"$PY" train.py --config_path "$TMP_TRAIN_YAML" $DRYRUN

echo "================================================================="
echo "  DONE. Output: ../outputs/results/"
echo "================================================================="
