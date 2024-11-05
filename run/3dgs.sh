#!/bin/bash
RUN_DIR=${RUN_DIR:-runs/noposegs}
RUN_DIR=${1:-$RUN_DIR}

# 3DGS baseline results, optionally with noise
NOISE=0
PREFIX=3dgs

# using the same split as in the joint reconstruction & estimation experiments
python full_eval.py -llff --output_path ${RUN_DIR}/${PREFIX}_llff \
    --extra_train_args \
    test_hold=0.1 \
    resolution=640 \
    cam_noise=$NOISE

python print_results.py --root ${RUN_DIR}/${PREFIX}_llff -llff --metrics PSNR SSIM LPIPS ROT POS


python full_eval.py -rep --output_path ${RUN_DIR}/${PREFIX}_replica \
    --extra_train_args \
    cam_noise=$NOISE

python print_results.py --root ${RUN_DIR}/${PREFIX}_replica -rep --metrics PSNR SSIM LPIPS RPE_t RPE_r ATE


python full_eval.py -tatnn --output_path ${RUN_DIR}/${PREFIX}_tanks \
    --extra_train_args \
    cam_noise=$NOISE

python print_results.py --root ${RUN_DIR}/${PREFIX}_tanks -tatnn --metrics PSNR SSIM LPIPS RPE_t RPE_r ATE