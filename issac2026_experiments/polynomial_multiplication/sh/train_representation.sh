#!/usr/bin/env bash
# Section 5.5: standard vs monomial token embedding, on P-Multiplication and
# P-Multiplication+.
#
#   bash sh/train_representation.sh
#
# One dataset serves both tasks: --target_mode last_element keeps only the final
# product (P-Multiplication), --target_mode full keeps the cumulative products
# (P-Multiplication+). Everything except the embedding is held fixed between the
# two arms — same data, same architecture, same 32-epoch budget — which is the
# whole point of the comparison.
#
# 16 runs (4 fields x 2 embeddings x 2 target modes), one GPU each; edit the
# device assignments to match the machine.
set -u
cd "$(dirname "$0")/.."

launch () {
    local gpu="$1" field="$2" which="$3" mode="$4"
    local tag="${field}_repr_${which}_${mode}"
    mkdir -p "results_${tag}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 train.py \
      --train_config_path "configs/${field}/train_repr_${which}.yaml" \
      --data_config_path "configs/${field}/data.yaml" \
      --target_mode "$mode" \
      --wandb_runname_postfix "${which}_${mode}" \
      > "results_${tag}/train.log" 2>&1 &
}

gpu=0
for field in ZZ GF7 GF31 GF97; do
    for which in standard monomial; do
        for mode in last_element full; do
            launch "$gpu" "$field" "$which" "$mode"
            gpu=$(( (gpu + 1) % 8 ))
        done
    done
done

echo "16 runs launched; P-Multiplication is the last_element half, P-Multiplication+ the full half"
