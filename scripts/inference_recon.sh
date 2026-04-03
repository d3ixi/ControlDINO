#!/bin/bash

ckpt_path=./ckpts/weights_recon.pt
video_root_dir="./assets/snpp_example"

python cli_demo_i2v.py \
    --video_root_dir $video_root_dir \
    --controlnet_features_model_path "$ckpt_path" \
    --output_path "out/snpp_example" \
    --start_camera_idx 0 \
    --end_camera_idx 1 \
    --controlnet_weights 1.0 \
    --controlnet_guidance_end 0.8 \
    --seed 1
