#!/usr/bin/env bash
# Degree and term count of the Groebner basis, lex vs degrevlex.
#
#   bash sh/measure_gb_basis_stats.sh
#
# This is the "how much does the model have to generate" half of the Section 5.4
# comparison; sh/measure_gb_timing.sh is the "how long does the CAS take" half.
# The paper's Table 7 wants the first setting only (GF(7), 2 variables, degree
# <= 4); the rest of the sweep is here because the claim under discussion is
# that lexicographic bases grow in degree *and* in length, and whether the two
# move together is exactly what changes with scale.
#
# Same sampler configs, same seeds and the same QQ rejection rule as
# measure_gb_timing.sh, so both scripts describe one sample set.
set -u
cd "$(dirname "$0")/.."

SAGE_BIN="${SAGE_BIN:-$HOME/micromamba/envs/sage/bin}"
CALT_SRC="${CALT_SRC:-$HOME/calt_issac_rerun/src}"
RUN="env PATH=$SAGE_BIN:$PATH PYTHONPATH=$CALT_SRC $SAGE_BIN/python measure_gb_basis_stats.py"
OUT="gb_basis_stats.json"

# (a) Table 7: the paper's own setting, over GF(7) and QQ.
for cfg in configs/data_GF7.yaml configs/data.yaml; do
    echo "=== paper setting: $cfg"
    $RUN --config_path "$cfg" --num_samples 1000 --output "$OUT"
done

# (b) The same scale points as the timing table, so the degree/length growth can
# be read next to the 4.7x / 448x / 80.8x time ratios.
for degree in 16 32; do
    echo "=== degree control: GF7, 2 variables, max_degree=$degree"
    $RUN --config_path configs/data_GF7.yaml \
         --max_degree "$degree" \
         --num_samples 200 \
         --timeout_s 120 \
         --output "$OUT"
done

echo "=== square system: GF7, 5 polynomials in 5 variables"
$RUN --config_path configs/data_GF7.yaml \
     --symbols "x,y,z,w,v" \
     --num_polynomials 5 \
     --num_samples 200 \
     --timeout_s 120 \
     --output "$OUT"

echo "all basis statistics written to $OUT"
