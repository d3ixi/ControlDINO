import sys
import os
import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import export_to_video, load_video

from controlnet_pipeline import ControlnetCogVideoXImageToVideoPCDPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet import CogVideoXControlnet
from controlnet_datasets import RealEstate10KPCDRenderDataset
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from torchvision.transforms import Resize
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch.nn.functional as F

def infer_controlnet_config_from_ckpt(ckpt_path):
    """Infer controlnet architecture params from checkpoint state dict."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # inner_dim from patch_embed.proj.bias shape
    inner_dim = state_dict['patch_embed.proj.bias'].shape[0]

    # num_layers from max transformer block index
    max_block_idx = -1
    for key in state_dict:
        if key.startswith('transformer_blocks.'):
            idx = int(key.split('.')[1])
            max_block_idx = max(max_block_idx, idx)
    num_layers = max_block_idx + 1

    # out_proj_dim from out_projectors (if present)
    out_proj_dim = None
    if 'out_projectors.0.weight' in state_dict:
        out_proj_dim = state_dict['out_projectors.0.weight'].shape[0]

    dino_version = "small_2x"  # default
    # Detect causal temporal encoder (CausalConv3d wraps weight as .conv.weight)
    use_causal_temporal = 'temporal_encoder.0.conv.weight' in state_dict

    # Detect dino_input_channels from dino_upscaler or temporal_encoder input channels
    dino_input_channels = None
    if 'dino_upscaler.0.weight' in state_dict:
        detected_ch = state_dict['dino_upscaler.0.weight'].shape[1]
        if detected_ch != 384:
            dino_input_channels = detected_ch
    elif 'temporal_encoder.0.weight' in state_dict:
        # For small_2x: temporal_encoder.0.weight shape is [out, in, D, H, W]
        detected_ch = state_dict['temporal_encoder.0.weight'].shape[1]
        if detected_ch != 384:
            dino_input_channels = detected_ch
    elif 'temporal_encoder.0.conv.weight' in state_dict:
        # Causal variant: temporal_encoder.0.conv.weight shape is [out, in, D, H, W]
        detected_ch = state_dict['temporal_encoder.0.conv.weight'].shape[1]
        if detected_ch != 384:
            dino_input_channels = detected_ch

    return {
        'inner_dim': inner_dim,
        'num_layers': num_layers,
        'out_proj_dim': out_proj_dim,
        'dino_version': dino_version,
        'dino_input_channels': dino_input_channels,
        'use_causal_temporal': use_causal_temporal,
        'state_dict': state_dict,
    }

def stack_images_horizontally(image1: Image.Image, image2: Image.Image) -> Image.Image:
    # Ensure both images have the same height
    height = max(image1.height, image2.height)
    width = image1.width + image2.width

    # Create a new blank image with the combined width and the maximum height
    new_image = Image.new('RGB', (width, height))

    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    return new_image

def pca(x):
    x = x.float()
    B, H, W, C = x.shape
    x_flat = x.reshape(-1, C)                # [B*H*W, C]

    # Center the data
    x_mean = x_flat.mean(0, keepdim=True)
    x_centered = x_flat - x_mean

    # Compute covariance matrix
    cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)  # [C, C]

    # Eigen decomposition (top 3 principal components)
    eigvals, eigvecs = torch.linalg.eigh(cov)                      # ascending order
    pcs = eigvecs[:, -3:]                                          # [C, 3]

    # Project and reshape back
    x_proj = x_centered @ pcs                                      # [B*H*W, 3]
    x_proj = x_proj.reshape(B, H, W, 3)

    # Normalize to [0, 1] for visualization
    x_proj = (x_proj - x_proj.min()) / (x_proj.max() - x_proj.min() + 1e-8)

    return x_proj

@torch.no_grad()
def generate_video(
    prompt,
    image,
    video_root_dir: str,
    base_model_path: str,
    controlnet_model_path: str,
    controlnet_features_model_path: str,
    controlnet_weights: float = 1.0,
    controlnet_guidance_start: float = 0.0,
    controlnet_guidance_end: float = 1.0,
    output_path: str = "./output/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
    start_camera_idx: int = 0,
    end_camera_idx: int = 1,
    controlnet_transformer_attention_head_dim: int = None,
    controlnet_transformer_num_layers: int = 8,
    downscale_coef: int = 8,
    controlnet_input_channels: int = 6,
    pipe_cpu_offload: bool = False,
    dino_version: str = "small_2x",
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_root_dir (str): The path to the camera dataset
    - annotation_json (str): Name of subset (train.json or test.json)
    - base_model_path (str): The path of the pre-trained model to be used.
    - controlnet_model_path (str): The path of the pre-trained conrolnet model to be used.
    - controlnet_weights (float): Strenght of controlnet
    - controlnet_guidance_start (float): The stage when the controlnet starts to be applied
    - controlnet_guidance_end (float): The stage when the controlnet end to be applied
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    os.makedirs(output_path, exist_ok=True)
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer"
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        base_model_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        base_model_path, subfolder="scheduler"
    )
    num_attention_heads_orig = 48 if "5b" in base_model_path.lower() else 30
    controlnet_kwargs = {}

    controlnet_kwargs["num_attention_heads"] = num_attention_heads_orig
    if controlnet_transformer_attention_head_dim is not None:
        controlnet_kwargs["attention_head_dim"] = controlnet_transformer_attention_head_dim

    controlnet = None

    controlnet_features = None
    if controlnet_features_model_path is not None:
        ckpt_config = infer_controlnet_config_from_ckpt(controlnet_features_model_path)
        ckpt_inner_dim = ckpt_config['inner_dim']
        ckpt_num_layers = ckpt_config['num_layers']
        ckpt_out_proj_dim = ckpt_config['out_proj_dim']
        ckpt_dino_version = ckpt_config['dino_version']
        ckpt_dino_input_channels = ckpt_config.get('dino_input_channels')
        ckpt_use_causal_temporal = ckpt_config.get('use_causal_temporal', False)

        feat_attention_head_dim = controlnet_kwargs.get("attention_head_dim", 64)
        feat_num_attn_heads = ckpt_inner_dim // feat_attention_head_dim
        feat_num_layers = ckpt_num_layers

        print(f'[Auto-detected from checkpoint] inner_dim={ckpt_inner_dim}, '
              f'num_attention_heads={feat_num_attn_heads}, num_layers={feat_num_layers}, '
              f'out_proj_dim={ckpt_out_proj_dim}, dino_version={ckpt_dino_version}')

        feat_kwargs = dict(controlnet_kwargs)
        feat_kwargs["num_attention_heads"] = feat_num_attn_heads
        if ckpt_out_proj_dim is not None:
            feat_kwargs["out_proj_dim"] = ckpt_out_proj_dim

        controlnet_features = CogVideoXControlnet(
            num_layers=feat_num_layers,
            downscale_coef=downscale_coef,
            use_causal_temporal=ckpt_use_causal_temporal,
            **feat_kwargs,
        )

        controlnet_state_dict = ckpt_config['state_dict']
        m, u = controlnet_features.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Features - Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

    pipe = ControlnetCogVideoXImageToVideoPCDPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet_features=controlnet_features,
        scheduler=scheduler,
    ).to('cuda')
        
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe = pipe.to(dtype=dtype)
    if pipe_cpu_offload:
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    eval_dataset = RealEstate10KPCDRenderDataset(
        video_root_dir=video_root_dir,
        image_size=(height, width),
        sample_n_frames=num_frames,
    )
    
    camera_indices = list(range(start_camera_idx, end_camera_idx))

    for camera_idx in camera_indices:
        data_dict = eval_dataset[camera_idx]
        reference_video = data_dict['video']
        anchor_video = data_dict['anchor_video']
        features = data_dict['features']

        clip_name = eval_dataset.dataset[camera_idx]

        output_path_file = os.path.join(output_path, f"{clip_name}_{seed}_out.mp4")
        prompt = data_dict['caption']

        if image is None:
            input_images = reference_video[0].unsqueeze(0)
        else:
            input_images = torch.tensor(np.array(Image.open(image))).permute(2,0,1).unsqueeze(0)/255
            pixel_transforms = [transforms.Resize((480, 720)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
            for transform in pixel_transforms:
                input_images = transform(input_images)

        reference_frames = [to_pil_image(frame) for frame in ((reference_video)/2+0.5)]
        
        output_path_file_reference = output_path_file.replace("_out.mp4", "_reference.mp4")
        output_path_file_out_reference = output_path_file.replace(".mp4", "_reference.mp4")
        
        controlnet_output_mask = None

        video_generate_all = pipe(
            image=input_images,
            anchor_video=anchor_video,
            features=features,
            controlnet_output_mask=controlnet_output_mask,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            controlnet_weights=controlnet_weights,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
            latents=None,
        ).frames
        video_generate = video_generate_all[0]

        export_to_video(video_generate, output_path_file, fps=8)
        export_to_video(reference_frames, output_path_file_reference, fps=8)
        out_reference_frames = [
            stack_images_horizontally(frame_reference, frame_out)
            for frame_out, frame_reference in zip(video_generate, reference_frames)
            ]
        
        anchor_video = [to_pil_image(frame) for frame in ((anchor_video)/2+0.5)]

        feature_frames = [to_pil_image(Resize((height, width))(pca(features[f][None].permute(0, 2, 3, 1)).permute((0, 3, 1, 2))[0])) for f in range(features.shape[0])]
        out_reference_frames = [
            stack_images_horizontally(frame_out, frame_reference)
            for frame_out, frame_reference in zip(out_reference_frames, feature_frames)
            ]
        
        export_to_video(out_reference_frames, output_path_file_out_reference, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default=None, help="The description of the video to be generated")
    parser.add_argument("--image", type=str, default=None, help="The reference image of the video to be generated")
    parser.add_argument(
        "--video_root_dir",
        type=str,
        required=True,
        help="The path of the video for controlnet processing.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="ckpts/CogVideoX-5b-I2V", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_features_model_path", type=str, default=None, help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_weights", type=float, default=0.5, help="Strenght of controlnet")
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.5, help="The stage when the controlnet end to be applied")
    parser.add_argument(
        "--output_path", type=str, default="./output", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--start_camera_idx", type=int, default=0)
    parser.add_argument("--end_camera_idx", type=int, default=1)
    parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=64)

    parser.add_argument("--downscale_coef", type=int, default=8)
    parser.add_argument("--vae_channels", type=int, default=16)
    parser.add_argument("--controlnet_input_channels", type=int, default=3)
    parser.add_argument("--controlnet_transformer_num_layers", type=int, default=8)
    parser.add_argument("--dino_version", type=str, default="small_2x", choices=["small_2x"], help="DINO version for controlnet: 'small_2x' (384ch, 60x90), ")
    parser.add_argument("--enable_model_cpu_offload", action="store_true", default=False, help="Enable model CPU offload")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        image=args.image,
        video_root_dir=args.video_root_dir,
        base_model_path=args.base_model_path,
        controlnet_model_path=args.controlnet_model_path,
        controlnet_features_model_path=args.controlnet_features_model_path,
        controlnet_weights=args.controlnet_weights,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_camera_idx=args.start_camera_idx,
        end_camera_idx=args.end_camera_idx,
        controlnet_transformer_attention_head_dim=args.controlnet_transformer_attention_head_dim,
        controlnet_transformer_num_layers=args.controlnet_transformer_num_layers,
        downscale_coef=args.downscale_coef,
        controlnet_input_channels=args.controlnet_input_channels,
        pipe_cpu_offload=args.enable_model_cpu_offload,
        dino_version=args.dino_version,
    )
