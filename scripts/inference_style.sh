#!/bin/bash

ckpt_path=./ckpts/weights_style.pt
video_root_dir="./assets/style_example"

python cli_demo_i2v.py \
    --video_root_dir "$video_root_dir" \
    --controlnet_features_model_path "$ckpt_path" \
    --output_path "./out/stylized" \
    --start_camera_idx 0 \
    --end_camera_idx 1 \
    --controlnet_weights 0.8 \
    --controlnet_guidance_end 0.8 \
    --seed 1 \
    --image "assets/style_example/Francis_clip00_frame000_ukiyo.png"
   # --image "assets/style_example/Francis_clip00_frame000_cyberpunk.png"
