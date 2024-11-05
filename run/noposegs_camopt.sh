#!/bin/bash

RUN_DIR=${RUN_DIR:-runs/noposegs}
DATA_DIR=${2:-data/}
MODEL_DIR=${3:-${RUN_DIR}/camopt_models}
RUN_DIR=${1:-$RUN_DIR/camopt}

python full_eval.py -llff --output_path ${MODEL_DIR}/llff \
    --extra_train_args \
    densify_until_iter=20000 \
    opacity_loss_weight=0.01 \
    anisotropy_max_ratio=10 \
    num_points_limit=256000 \
    images=images_8 \
    white_background \
    --extra_render_args render_depth

python camopt_full_eval.py -llff \
    --data_root=${DATA_DIR} \
    --rundir_root=${RUN_DIR} \
    --model_root=${MODEL_DIR}/llff \
    --extra_args white_background forward_facing cam_lr_init=1e-2 cam_lr_final=1e-4 images=images_8

python print_results.py --root ${RUN_DIR}/camopt -llff --metrics POS ROT rot@5 pos@0.05