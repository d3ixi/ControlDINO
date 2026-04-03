#!/bin/bash

ckpt_path=./ckpts/weights_vkitti.pt
video_root_dir="./assets/vkitti_example"

python cli_demo_i2v.py \
    --video_root_dir "$video_root_dir" \
    --controlnet_features_model_path "$ckpt_path" \
    --output_path "out/vkitti" \
    --start_camera_idx 0 \
    --end_camera_idx 1 \
    --controlnet_weights 0.8 \
    --controlnet_guidance_end 0.8 \
    --seed 1 \
    --image "assets/vkitti_example/Scene18_00000_frame000.png"
