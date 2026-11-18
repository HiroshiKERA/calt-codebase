#!/bin/bash
# Run the full groebner_basis/toy experiment end-to-end.
#
# Usage:
#   bash run.sh                                  # defaults: degrevlex, raw, with pretok cache
#   bash run.sh --order lex                      # use lex order (FGLM)
#   bash run.sh --lexer expanded                 # use C/E expanded vocab
#   bash run.sh --order lex --lexer expanded     # combine
#   bash run.sh --no-preprocess                  # skip preprocess.py (legacy training path)
#   bash run.sh --dryrun                         # fast smoke test
#   bash run.sh --force                          # rebuild caches even if hash matches
#
# What this script does (in order):
#   1. python generate.py             # raw text data from SageMath (if not already done)
#   2. python preprocess.py [flags]   # build the pretokenized cache (unless --no-preprocess)
#   3. python train.py [flags]        # train (cache picked up automatically)
#   4. python evaluate.py --training_order <order>   # print the success rate and examples

set -e
ORIGINAL_ARGS="$*"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------- Preflight: check the right Python is on PATH --------------
# We need calt-x AND sage installed. If `python -c 'import calt, sage'` fails,
# the user is usually on the wrong env (typical: base conda instead of
# calt-env), or calt-env has pip's PyTorch in it (CXXABI error).
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
    echo "   Fix one of these:" >&2
    echo "     1. Activate the conda env:  conda activate calt-env" >&2
    echo "     2. Or set PYTHON explicitly: PYTHON=/path/to/calt-env/bin/python bash run.sh" >&2
    echo "" >&2
    echo "   Looking for a conda env that has calt installed..." >&2
    CONDA_ROOT=""
    if [[ -n "${CONDA_EXE:-}" ]]; then CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"; fi
    FOUND=""
    for cand in ${CONDA_ROOT:+"$CONDA_ROOT"/envs/*/bin/python} \
                ${MAMBA_ROOT_PREFIX:+"$MAMBA_ROOT_PREFIX"/envs/*/bin/python} \
                "$HOME"/miniforge3/envs/*/bin/python "$HOME"/mambaforge/envs/*/bin/python \
                "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python \
                "$HOME"/.conda/envs/*/bin/python /opt/conda/envs/*/bin/python; do
        if [[ -x "$cand" ]] && "$cand" -c "import calt, sage" >/dev/null 2>&1; then
            FOUND="$cand"
            echo "   Found: $cand" >&2
            echo "     → re-run with: PYTHON=$(printf '%q' "$cand") bash run.sh ${ORIGINAL_ARGS:-}" >&2
            break
        fi
    done
    [[ -n "$FOUND" ]] || echo "   (none found)" >&2
    exit 1
fi
# From here on, all python invocations use $PY.

# ------------- Parse flags -----------------------------------------------
ORDER="degrevlex"
LEXER="raw"
PREPROCESS=1
DRYRUN=""
FORCE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --order)            ORDER="$2"; shift 2 ;;
        --lexer)            LEXER="$2"; shift 2 ;;
        --no-preprocess)    PREPROCESS=0; shift ;;
        --dryrun)           DRYRUN="--dryrun"; shift ;;
        --force)            FORCE="--force"; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# ------------- Validate flags --------------------------------------------
case $ORDER in
    degrevlex|lex) ;;
    *) echo "Invalid --order $ORDER (must be degrevlex or lex)" >&2; exit 2 ;;
esac
case $LEXER in
    raw|expanded) ;;
    *) echo "Invalid --lexer $LEXER (must be raw or expanded)" >&2; exit 2 ;;
esac

# ------------- Plan -----------------------------------------------------
LEXER_PATH="../configs/lexer.yaml"
[[ "$LEXER" == "expanded" ]] && LEXER_PATH="../configs/lexer_expanded.yaml"

echo "================================================================="
echo "  groebner_basis/toy — end-to-end run"
echo "  order   : $ORDER"
echo "  lexer   : $LEXER  ($LEXER_PATH)"
echo "  preproc : $([ $PREPROCESS -eq 1 ] && echo enabled || echo skipped)"
echo "  dryrun  : $([ -n "$DRYRUN" ] && echo YES || echo no)"
echo "================================================================="

# ------------- Apply lexer override --------------------------------------
# train.yaml & data.yaml live in ../configs. We don't modify them on disk;
# instead we generate a temp train.yaml with the lexer override.
TMP_TRAIN_YAML="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_TRAIN_YAML"' EXIT
sed "s|lexer_config:.*|lexer_config: $LEXER_PATH|" ../configs/train.yaml > "$TMP_TRAIN_YAML"

# ------------- Step 1: generate ------------------------------------------
if [[ ! -f ../data/QQ/train_raw.txt ]]; then
    echo "=== Step 1/4: Generate data ==="
    "$PY" generate.py
else
    echo "=== Step 1/4: Generate data ==="
    echo "    ../data/QQ/train_raw.txt already exists, skipping (delete to regenerate)"
fi

# ------------- Step 2: preprocess (optional) -----------------------------
if [[ $PREPROCESS -eq 1 ]]; then
    echo "=== Step 2/4: Preprocess (build cache) ==="
    "$PY" preprocess.py \
        --config_path "$TMP_TRAIN_YAML" \
        --training_order "$ORDER" \
        $FORCE
else
    echo "=== Step 2/4: Preprocess SKIPPED (--no-preprocess) ==="
fi

# ------------- Step 3: train ---------------------------------------------
echo "=== Step 3/4: Train ==="
"$PY" train.py \
    --config_path "$TMP_TRAIN_YAML" \
    --training_order "$ORDER" \
    $DRYRUN

echo "=== Step 4/4: Evaluate ==="
"$PY" evaluate.py --training_order "$ORDER"

echo "================================================================="
echo "  DONE. Output: ../outputs/results_${ORDER}/"
echo "================================================================="
