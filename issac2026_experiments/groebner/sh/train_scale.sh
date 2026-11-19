#!/usr/bin/env bash
# Groebner basis learning, lex vs degrevlex, at the remaining scales of Table 6.
#
#   bash sh/generate_scale_datasets.sh   # first, this writes ./data/GF7_*
#   bash sh/train_scale.sh
#
# sh/train.sh covers the degree <= 4 setting over QQ and GF(7), which is what
# Table 5 reports. The point of this script is to put the *learning* gap between
# the two orders next to the *classical* gap at the scales where the classical
# gap is large: 4.7x at degree 16, 448x at degree 32, and 80.8x on a 5x5 system.
#
# Six runs, one GPU each; edit the CUDA_VISIBLE_DEVICES assignments to match the
# machine. Runs are long — these are 64-epoch runs on longer sequences than the
# degree 4 setting.
set -u
cd "$(dirname "$0")/.."

mkdir -p \
  results_GF7_deg16_degrevlex results_GF7_deg16_lex \
  results_GF7_deg32_degrevlex results_GF7_deg32_lex \
  results_GF7_5x5_degrevlex   results_GF7_5x5_lex

launch () {
    local gpu="$1" tag="$2" train_cfg="$3" data_cfg="$4" order="$5"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 train.py \
      --config_path "$train_cfg" \
      --data_config_path "$data_cfg" \
      --training_order "$order" \
      > "results_${tag}_${order}/train.log" 2>&1 &
}

# 2 variables, degree <= 16
launch 0 GF7_deg16 configs/train_GF7_deg16.yaml configs/data_GF7_deg16.yaml degrevlex
launch 1 GF7_deg16 configs/train_GF7_deg16.yaml configs/data_GF7_deg16.yaml lex

# 2 variables, degree <= 32
launch 2 GF7_deg32 configs/train_GF7_deg32.yaml configs/data_GF7_deg32.yaml degrevlex
launch 3 GF7_deg32 configs/train_GF7_deg32.yaml configs/data_GF7_deg32.yaml lex

# 5 polynomials in 5 variables
launch 4 GF7_5x5 configs/train_GF7_5x5.yaml configs/data_GF7_5x5.yaml degrevlex
launch 5 GF7_5x5 configs/train_GF7_5x5.yaml configs/data_GF7_5x5.yaml lex

echo "six runs launched; success rates land in results_*/"
