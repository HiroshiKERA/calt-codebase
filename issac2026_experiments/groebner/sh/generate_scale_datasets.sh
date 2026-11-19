#!/usr/bin/env bash
# Datasets for the remaining scales of Table 6 (all over GF(7)).
#
#   bash sh/generate_scale_datasets.sh
#
# The dataset on disk is degrevlex in every case; train.py --training_order lex
# moves the system into a lex ring and recomputes the basis at load time, so one
# dataset per scale serves both orders.
#
# These take substantially longer than the degree 4 dataset: the degree 32
# systems are the slow ones, and the 5x5 square systems are the ones that can
# occasionally blow up. Generation is deterministic from root_seed, so an
# interrupted run can be repeated without changing the data.
set -u
cd "$(dirname "$0")/.."

for cfg in configs/data_GF7_deg16.yaml configs/data_GF7_deg32.yaml configs/data_GF7_5x5.yaml; do
    echo "=== generating $cfg"
    python3 generate_dataset.py --config_path "$cfg"
done

echo "datasets written under ./data/"
