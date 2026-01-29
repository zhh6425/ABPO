#!/bin/bash
set -x

GOLBAL_STEP=120
model_path=./checkpoints/

python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$model_path/global_step_$GOLBAL_STEP/actor" \
    --target_dir "$model_path/global_step_$GOLBAL_STEP/hf_model"