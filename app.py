import os
import time
import numpy as np
import gradio as gr

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
)

from diffusers.utils import (
    load_video,
    export_to_video,
)

from controlnet_pipeline import (
    ControlnetCogVideoXImageToVideoPCDPipeline,
)

from cogvideo_transformer import (
    CustomCogVideoXTransformer3DModel,
)

from cogvideo_controlnet import (
    CogVideoXControlnet,
)


@torch.no_grad()
def load_dinov3():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

    return model.to(device="cuda").eval()

@torch.no_grad()
def compute_dinov3_features_2x(videos, model, dtype, chunk_size=16):
    B, Fr, C, H, W = videos.shape

    flat = videos.reshape(B * Fr, C, H, W).to(dtype)
    flat = torch.nn.functional.interpolate(flat, scale_factor=2, mode="bicubic", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=flat.device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=flat.device, dtype=dtype).view(1, 3, 1, 1)
    flat = ((flat + 1) / 2 - mean) / std

    h_p, w_p = flat.shape[-2] // 16, flat.shape[-1] // 16
    num_patches = h_p * w_p

    outs = []
    for i in range(0, flat.shape[0], chunk_size):
        tokens = model(flat[i:i + chunk_size]).last_hidden_state
        outs.append(tokens[:, -num_patches:])  # drop special tokens

    feats = torch.cat(outs, dim=0)
    feats = feats.reshape(B, Fr, h_p, w_p, -1).permute(0, 1, 4, 2, 3).contiguous()
    return feats

def pca(x):
    x = x.float()
    H, W, C = x.shape
    x_flat = x.reshape(-1, C)
    x_mean = x_flat.mean(0, keepdim=True)
    x_centered = x_flat - x_mean
    cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    pcs = eigvecs[:, -3:]
    x_proj = x_centered @ pcs
    x_proj = x_proj.reshape(H, W, 3)
    x_proj = (x_proj - x_proj.min()) / (x_proj.max() - x_proj.min() + 1e-8)
    return x_proj

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL_PATH = "ckpts/CogVideoX-5b-I2V"
CHECKPOINTS = {
    "Reconstruction": "./ckpts/weights_recon.pt",
    "Style": "./ckpts/weights_style.pt",
    "DwD": "./ckpts/weights_vkitti.pt",
}
DEVICE = "cuda"
DTYPE = torch.bfloat16

PIPELINE_CACHE = {}

print("Loading DINOv3...")
dino_model = load_dinov3()
print("DINOv3 loaded.")

os.makedirs("outputs", exist_ok=True)

EXAMPLES = [
    [
        "./assets/snpp_example/videos/09c1414f1b_000.mp4",
        None,
        "Reconstruction",
        "A cozy living room setup featuring a sectional sofa, a leather recliner, and a wooden coffee table.",
    ],
    [
        "./assets/style_example/videos/Francis_clip00.mp4",
        "./assets/style_example/Francis_clip00_frame000_ukiyo.png",
        "Style",
        "The image depicts a statue with a white base and columns, situated in a park-like setting. The statue is surrounded by a paved area with grass and small white flowers. In the background, there is a modern building with a unique, angular design, and a bridge or walkway leading to it. The sky is clear with a few clouds, and the overall atmosphere is serene and peaceful.",
    ],
    [
        "./assets/style_example/videos/Francis_clip00.mp4",
        "./assets/style_example/Francis_clip00_frame000_cyberpunk.png",
        "Style",
        "The image depicts a statue with a white base and columns, situated in a park-like setting. The statue is surrounded by a paved area with grass and small white flowers. In the background, there is a modern building with a unique, angular design, and a bridge or walkway leading to it. The sky is clear with a few clouds, and the overall atmosphere is serene and peaceful.",
    ],
    [
        "./assets/vkitti_example/videos/Scene18_00000.mp4",
        "./assets/vkitti_example/Scene18_00000_frame000.png",
        "DwD",
        "The image depicts a winding road flanked by lush green trees. The road is paved with asphalt, and the trees are adorned with vibrant green leaves, suggesting a spring or summer season. The road curves gently to the right, creating a sense of depth and direction. The overall scene is serene and inviting, with a clear path leading into the distance.",
    ],
]


# ============================================================
# CONTROLNET LOADER
# ============================================================

def infer_controlnet_config_from_ckpt(ckpt_path):

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["state_dict"]

    inner_dim = state_dict[
        "patch_embed.proj.bias"
    ].shape[0]

    max_block_idx = -1

    for key in state_dict:
        if key.startswith("transformer_blocks."):
            idx = int(key.split(".")[1])
            max_block_idx = max(max_block_idx, idx)

    num_layers = max_block_idx + 1

    out_proj_dim = None

    if "out_projectors.0.weight" in state_dict:
        out_proj_dim = state_dict[
            "out_projectors.0.weight"
        ].shape[0]

    use_causal_temporal = (
        "temporal_encoder.0.conv.weight"
        in state_dict
    )

    return {
        "inner_dim": inner_dim,
        "num_layers": num_layers,
        "out_proj_dim": out_proj_dim,
        "use_causal_temporal": use_causal_temporal,
        "state_dict": state_dict,
    }

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")

tokenizer = T5Tokenizer.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="tokenizer",
)

print("Loading text encoder...")

text_encoder = T5EncoderModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="text_encoder",
)

print("Loading transformer...")

transformer = (
    CustomCogVideoXTransformer3DModel.from_pretrained(
        BASE_MODEL_PATH,
        subfolder="transformer",
    )
)

print("Loading VAE...")

vae = AutoencoderKLCogVideoX.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="vae",
)

print("Loading scheduler...")

scheduler = CogVideoXDDIMScheduler.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="scheduler",
)

def get_pipeline(model_name):

    if model_name in PIPELINE_CACHE:
        return PIPELINE_CACHE[model_name]

    ckpt_path = CHECKPOINTS[model_name]

    cfg = infer_controlnet_config_from_ckpt(
        ckpt_path
    )

    controlnet = CogVideoXControlnet(
        num_layers=cfg["num_layers"],
        downscale_coef=8,
        use_causal_temporal=cfg["use_causal_temporal"],
        num_attention_heads=cfg["inner_dim"] // 64,
        attention_head_dim=64,
        out_proj_dim=cfg["out_proj_dim"],
    )

    missing, unexpected = controlnet.load_state_dict(
        cfg["state_dict"],
        strict=False,
    )

    print(
        f"Controlnet loaded. "
        f"Missing={len(missing)} "
        f"Unexpected={len(unexpected)}"
    )

    pipe = ControlnetCogVideoXImageToVideoPCDPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet_features=controlnet,
        scheduler=scheduler,
    )

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    pipe = pipe.to(DEVICE)
    pipe = pipe.to(dtype=DTYPE)

    PIPELINE_CACHE[model_name] = pipe

    return pipe

# ============================================================
# VIDEO PREPROCESS
# ============================================================

def preprocess_anchor_video(video_path):

    frames = load_video(video_path)

    print("Loaded frames:", len(frames))

    frames = frames[:49]

    while len(frames) < 49:
        frames.append(frames[-1])

    anchor_video = (
        torch.tensor(np.array(frames))
        .float()
        .permute(0, 3, 1, 2)
        / 255.0
    )

    anchor_video = F.interpolate(
        anchor_video,
        size=(480, 720),
        mode="bilinear",
        align_corners=False,
    )

    anchor_video = anchor_video * 2.0 - 1.0

    print(
        "anchor_video shape:",
        anchor_video.shape,
    )

    return anchor_video

# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    video_input,
    image_input,
    prompt,
    seed,
    model_name,
    num_inference_steps,
    guidance_scale,
    controlnet_weights,
    controlnet_guidance_start,
    controlnet_guidance_end
):

    print("\n========================")
    print("DEBUG INPUTS")
    print("========================")

    print(
        "video_input:",
        type(video_input),
        video_input,
    )

    print(
        "image_input:",
        type(image_input),
    )

    print(
        "prompt:",
        prompt,
    )

    print(
        "seed:",
        seed,
    )

    if video_input is None:
        raise gr.Error(
            "Please upload a video."
        )

    anchor_video = preprocess_anchor_video(
        video_input
    )

    # ------------------------------------------------
    # CONDITION IMAGE
    # ------------------------------------------------

    if image_input is not None:

        print(
            "Using uploaded conditioning image"
        )

        if isinstance(image_input, np.ndarray):
            cond_image = Image.fromarray(
                image_input
            ).convert("RGB")
        else:
            cond_image = image_input

        cond_image = cond_image.resize(
            (720, 480)
        )

    else:

        print(
            "No image uploaded. Using first frame."
        )

        frame = (
            (anchor_video[0] + 1.0)
            / 2.0
        ).clamp(0, 1)

        cond_image = to_pil_image(frame)

    image_tensor = (
        torch.tensor(np.array(cond_image))
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    image_tensor = image_tensor * 2.0 - 1.0

    print(
        "image_tensor shape:",
        image_tensor.shape,
    )

    # ------------------------------------------------
    # FEATURES
    # ------------------------------------------------

    video_for_dino = (
        anchor_video.unsqueeze(0)
        .to(
            DEVICE,
            dtype=DTYPE,
        )
    )

    features = compute_dinov3_features_2x(
        video_for_dino,
        dino_model,
        DTYPE,
    )[0]

    pca_frames = []

    for f in range(features.shape[0]):

        feat = (
            features[f]
            .permute(1, 2, 0)
            .cpu()
        )

        rgb = pca(feat)

        rgb = (
            rgb * 255
        ).byte()

        pca_frames.append(
            Image.fromarray(
                rgb.numpy()
            )
        )

    print(
        "Extracted DINO features:",
        features.shape,
    )


    # ------------------------------------------------
    # GENERATION
    # ------------------------------------------------
    pipe = get_pipeline(model_name)
    result = pipe(
        image=image_tensor.to(
            DEVICE,
            dtype=DTYPE,
        ),
        anchor_video=anchor_video.to(
            DEVICE,
            dtype=DTYPE,
        ),
        features=features,
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
        generator=torch.Generator(
            device=DEVICE
        ).manual_seed(int(seed)),
        controlnet_weights=controlnet_weights,
        controlnet_guidance_start=controlnet_guidance_start,
        controlnet_guidance_end=controlnet_guidance_end,
        controlnet_output_mask=None,
        latents=None,
    ).frames[0]

    output_path = os.path.join(
        "outputs",
        f"result_{int(time.time())}.mp4",
    )

    export_to_video(
        result,
        output_path,
        fps=8,
    )

    pca_video_path = output_path.replace(
        ".mp4",
        "_pca.mp4",
    )

    export_to_video(
        pca_frames,
        pca_video_path,
        fps=8,
    )

    print(
        "Saved:",
        output_path,
    )

    return (
        output_path,
        pca_video_path,
    )

# ============================================================
# UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("""
<div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
  Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion
</div>
<div style="text-align: center;">
  <a href="https://dedoardo.github.io/projects/control-dino/">🚀 Project Page</a> |
  <a href="https://github.com/d3ixi/ControlDINO.git">🌐 Github</a> |
  <a href="https://arxiv.org/abs/2604.01761">📜 arxiv </a>
</div>
"""
    )

    with gr.Row():

        with gr.Column():

            video_input = gr.Video(
                label="Anchor Video"
            )

            image_input = gr.Image(
                label="Conditioning Image (Optional)"
            )

            prompt_input = gr.Textbox(
                label="Prompt",
                lines=4,
            )

            checkpoint_dropdown = gr.Dropdown(
                label="ControlNet Model",
                choices=[
                    "Reconstruction",
                    "Style",
                    "DwD",
                ],
                value="Reconstruction",
            )

            seed_input = gr.Number(
                value=42,
                label="Seed",
            )

            with gr.Accordion("Generation Settings", open=False):
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference Steps",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=6.0,
                    step=0.5,
                    label="Guidance Scale",
                )
                controlnet_weights = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    label="ControlNet Weight",
                )
                controlnet_guidance_start = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="ControlNet Start",
                )
                controlnet_guidance_end = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="ControlNet End",
                )

            generate_btn = gr.Button(
                "Generate"
            )

        with gr.Column():

            output_video = gr.Video(
                label="Output Video"
            )

            pca_video_output = gr.Video(
                label="DINOv3 PCA Features"
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            video_input,
            image_input,
            checkpoint_dropdown,
            prompt_input,
        ],
    )

    generate_btn.click(
        fn=run_inference,
        inputs=[
            video_input,
            image_input,
            prompt_input,
            seed_input,
            checkpoint_dropdown,
            num_inference_steps,
            guidance_scale,
            controlnet_weights,
            controlnet_guidance_start,
            controlnet_guidance_end
        ],
        outputs=[
            output_video,
            pca_video_output,
        ],
    )

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
