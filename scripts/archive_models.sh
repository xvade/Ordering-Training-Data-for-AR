#!/bin/bash
# Copy the headline models out of /gscratch/scrubbed (purged after 21 days without
# access) into the group's durable storage, one tar file per family of runs.
#
# One tar per family, not loose files: the group fileset is close to its limit on
# the NUMBER of files (95% on 2026-11-15, `hyakstorage /gscratch/amath`).
#
# Each tar holds, for every seed of the family: config.json, meta.json (code
# revision, timings), evals.json (validation metrics at every epoch), the test
# evaluations (eval_test*.json) and the checkpoints named below. Checkpoints are
# named by update step; 351 steps = 1 epoch, so epoch e is ckpt_{351*e:06d}.pt.
# Load one with arorder.train.load_checkpoint (see scripts/final_eval.py, --ckpt
# and --weights ema for the weight-averaged model).
#
# Usage (inside a Slurm job; light I/O):  bash scripts/archive_models.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
DEST=${AR_MODELS:-/gscratch/amath/sgvtc/ordering-ar/models}
mkdir -p "$DEST"

archive() {  # archive NAME VARIANT_DIR(relative to $AR_RUNS) EPOCH...
    local name=$1 dir=$2; shift 2
    local files=()
    for s in "$AR_RUNS/$dir"/seed*; do
        local rel=${s#"$AR_RUNS"/}
        for f in config.json meta.json evals.json ckpt_final.pt; do files+=("$rel/$f"); done
        for f in "$s"/eval_test*.json; do [ -e "$f" ] && files+=("$rel/$(basename "$f")"); done
        for e in "$@"; do files+=("$rel/$(printf 'ckpt_%06d.pt' $((351 * e)))"); done
    done
    tar -C "$AR_RUNS" -cf "$DEST/$name.tar" "${files[@]}"
    echo "$name: ${#files[@]} files, $(du -h "$DEST/$name.tar" | cut -f1)"
}

# SmallCNN, CIFAR-10. Epochs are the operating points selected on val (docs/FINDINGS.md, section 5).
archive smallcnn_c10_gradmatch_eps2   p7_gm/gm_e2           6 9 13 14
archive smallcnn_c10_gradmatch_eps1   p7_gm/gm_e1           5 7 11 17
archive smallcnn_c10_uniform          p9_ref/uniform_ck     4 6 12
archive smallcnn_c10_uniform_lr0025   p9_ref/uni_lr0025     5 7 9 10
archive smallcnn_c10_advtrain_eps2    p9_ref/at_e2
archive smallcnn_c10_advtrain_eps8    p6_at_rules/at_uniform
# Other architecture / dataset
archive resnet20_c10_gradmatch_eps2   p8_general/rn20_gm_e2 11 12 16
archive resnet20_c10_uniform          p8_general/rn20_uniform
archive smallcnn_c100_gradmatch_eps2  p8_general/c100_gm_e2 11 15
archive smallcnn_c100_uniform         p8_general/c100_uniform
ls -la "$DEST"
