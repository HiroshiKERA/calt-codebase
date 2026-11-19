#!/usr/bin/env bash
# Section 5.5: standard vs monomial token embedding, on P-Reduction.
#
#   bash sh/train_representation.sh
#
# The target is the pair (quotient, remainder), i.e. --target_mode full, which
# is the P-Reduction of the paper. Everything except the embedding is held fixed
# between the two arms — same data, same architecture, same 32-epoch budget.
#
# 8 runs (4 fields x 2 embeddings), one GPU each; edit the device assignments to
# match the machine.
set -u
cd "$(dirname "$0")/.."

launch () {
    local gpu="$1" field="$2" which="$3"
    local tag="${field}_repr_${which}"
    mkdir -p "results_${tag}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 train.py \
      --train_config_path "configs/${field}/train_repr_${which}.yaml" \
      --data_config_path "configs/${field}/data.yaml" \
      --target_mode full \
      --wandb_runname_postfix "$which" \
      > "results_${tag}/train.log" 2>&1 &
}

gpu=0
for field in ZZ GF7 GF31 GF97; do
    for which in standard monomial; do
        launch "$gpu" "$field" "$which"
        gpu=$(( (gpu + 1) % 8 ))
    done
done

echo "8 runs launched"
