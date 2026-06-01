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
#   3. python train.py [flags]        # train + evaluate (cache picked up automatically)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------- Preflight: check the right Python is on PATH --------------
# We need calt-x AND sage installed. If `python -c 'import calt, sage'` fails,
# the user is on the wrong env (typical: base conda instead of calt-env).
PY="${PYTHON:-python}"
if ! "$PY" -c "import calt, sage" >/dev/null 2>&1; then
    echo "❌ Wrong Python environment: \`$PY\` cannot import \`calt\` or \`sage\`." >&2
    echo "   Fix one of these:" >&2
    echo "     1. Activate the conda env:  conda activate calt-env" >&2
    echo "     2. Or set PYTHON explicitly: PYTHON=/path/to/calt-env/bin/python bash run.sh" >&2
    echo "" >&2
    echo "   Looking for ANY env that has calt installed..." >&2
    for cand in /home/t-maxime/miniconda3/envs/*/bin/python /opt/conda/envs/*/bin/python; do
        if [[ -x "$cand" ]] && "$cand" -c "import calt" >/dev/null 2>&1; then
            echo "   Found: $cand" >&2
            echo "     → re-run with: PYTHON=$cand bash run.sh ${ORIGINAL_ARGS:-}" >&2
            break
        fi
    done
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
    echo "=== Step 1/3: Generate data ==="
    "$PY" generate.py
else
    echo "=== Step 1/3: Generate data ==="
    echo "    ../data/QQ/train_raw.txt already exists, skipping (delete to regenerate)"
fi

# ------------- Step 2: preprocess (optional) -----------------------------
if [[ $PREPROCESS -eq 1 ]]; then
    echo "=== Step 2/3: Preprocess (build cache) ==="
    "$PY" preprocess.py \
        --config_path "$TMP_TRAIN_YAML" \
        --training_order "$ORDER" \
        $FORCE
else
    echo "=== Step 2/3: Preprocess SKIPPED (--no-preprocess) ==="
fi

# ------------- Step 3: train + evaluate ----------------------------------
echo "=== Step 3/3: Train + Evaluate ==="
"$PY" train.py \
    --config_path "$TMP_TRAIN_YAML" \
    --training_order "$ORDER" \
    $DRYRUN

echo "=== Step 4/3: Evaluate (optional, results already printed by train) ==="
"$PY" evaluate.py --training_order "$ORDER" || echo "(evaluate.py is optional — train.py already wrote eval_results/)"

echo "================================================================="
echo "  DONE. Output: ../outputs/results_${ORDER}/"
echo "================================================================="
