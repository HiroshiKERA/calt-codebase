#!/usr/bin/env bash
# Section 5.5: standard vs monomial token embedding, on P-Groebner over GF(7).
#
#   bash sh/train_representation.sh
#
# Both arms read the same C/E expanded-form data (--expanded_form) and differ
# only in the embedding. This is deliberately *not* the Table 5 setup: Table 5
# trains on raw polynomial strings for 64 epochs, these run on expanded form for
# 32, so the success rates are not comparable across the two tables.
set -u
cd "$(dirname "$0")/.."

mkdir -p results_GF7_repr_standard results_GF7_repr_monomial

CUDA_VISIBLE_DEVICES=0 nohup python3 train.py \
  --config_path configs/train_GF7_repr_standard.yaml \
  --data_config_path configs/data_GF7.yaml \
  --training_order degrevlex \
  --expanded_form \
  > results_GF7_repr_standard/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 train.py \
  --config_path configs/train_GF7_repr_monomial.yaml \
  --data_config_path configs/data_GF7.yaml \
  --training_order degrevlex \
  --expanded_form \
  > results_GF7_repr_monomial/train.log 2>&1 &

echo "2 runs launched"
