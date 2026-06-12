#!/bin/bash
# Run the full parity/toy experiment end-to-end.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1/3: Generate data ==="
python generate.py

echo "=== Step 2/3: Train ==="
python train.py "$@"

echo "=== Step 3/3: Evaluate ==="
python evaluate.py
