#!/bin/bash
RUN_DIR=${RUN_DIR:-runs/noposegs}
RUN_DIR=${1:-$RUN_DIR}

# joint reconstruction and pose estimation on llff
python full_eval.py -llff --nopose --output_path ${RUN_DIR}/llff \
    --extra_train_args \
    test_hold=0.1 \
    init_cam_identity \
    cam_lr_init=1e-2 \
    cam_lr_final=1e-4 \
    cam_lr_max_steps=30000 \
    iterations=30000 \
    densify_until_iter=20000 \
    load_depth=dpt \
    position_lr_init=1.6e-2 \
    position_lr_final=1.6e-4 \
    opacity_loss_weight=0.01 \
    anisotropy_max_ratio=10 \
    num_points_limit=256000 \
    resolution=640 \
    --extra_render_args align_cameras=forward test_time_opt_steps=200 render_depth

# joint reconstruction and pose refinement on replica
python full_eval.py -rep --nopose --output_path ${RUN_DIR}/replica \
    --extra_train_args \
    cam_noise=0.05 \
    cam_lr_init=1e-2 \
    cam_lr_final=1e-4 \
    cam_lr_max_steps=60000 \
    iterations=60000 \
    densify_until_iter=20000 \
    load_depth=dpt \
    position_lr_init=1.6e-2 \
    position_lr_final=1.6e-4 \
    opacity_loss_weight=0.01 \
    anisotropy_max_ratio=10 \
    num_points_limit=256000 \
    --extra_render_args align_cameras test_time_opt_steps=200 render_depth

# joint reconstruction and pose estimation on tanks and temples
python full_eval.py -tatnn --nopose --output_path ${RUN_DIR}/tanks \
    --extra_scene_args Family:--test_hold=2 \
    --extra_train_args \
    init_cam_identity \
    cam_lr_init=1e-2 \
    cam_lr_final=1e-4 \
    cam_lr_max_steps=30000 \
    iterations=30000 \
    densify_until_iter=20000 \
    load_depth=dpt \
    num_previous_frames=1 \
    position_lr_init=1.6e-2 \
    position_lr_final=1.6e-4 \
    opacity_loss_weight=0.01 \
    anisotropy_max_ratio=10 \
    num_points_limit=256000 \
    per_frame_cam_lr_init=1e-3 \
    --extra_render_args align_cameras=nearest test_time_opt_steps=200 render_depth \
    --extra_eval_args spherify_poses

python print_results.py --root ${RUN_DIR}/llff -llff --metrics PSNR SSIM LPIPS ROT POS

python print_results.py --root ${RUN_DIR}/replica -rep --metrics PSNR SSIM LPIPS RPE_t RPE_r ATE

python print_results.py --root ${RUN_DIR}/tanks -tatnn --metrics PSNR SSIM LPIPS RPE_t RPE_r ATE
