from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput, CogVideoXBlock
from diffusers.loaders import  PeftAdapterMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class TemporalFeatureAdapter(nn.Module):
    def __init__(
        self,
        dino_channels=384,
        latent_channels=256,
        temporal_stride=4,    # Compression ratio (N_frames -> N/4)
        spatial_downscale=1   # Usually 1 if patch size matches latent stride
    ):
        super().__init__()

        self.downscaler = nn.Sequential(
            nn.Conv3d(
                in_channels=dino_channels,
                out_channels=512,
                kernel_size=(3, 3, 3),
                stride=(temporal_stride // 2, 1, 1), # Partial temporal downsample
                padding=(1, 1, 1)
            ),
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv3d(
                in_channels=512,
                out_channels=latent_channels,
                kernel_size=(3, 3, 3),
                stride=(temporal_stride // 2, spatial_downscale, spatial_downscale),
                padding=(1, 1, 1)
            ),
            nn.Conv3d(latent_channels, latent_channels, 1)
        )

    def forward(self, x):
        # Conv3d expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.downscaler(x)
        x = x.permute((0, 2, 1, 3, 4))

        return x

class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding (only pads from past, like CogVideoX VAE)."""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, 1, 1)

        self.time_pad = kernel_size[0] - 1  # causal: pad only from past
        # Spatial padding is symmetric as usual
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=(0, kernel_size[1] // 2, kernel_size[2] // 2),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Causal pad: (W_left, W_right, H_top, H_bot, T_past, T_future)
        x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
        return self.conv(x)


class CausalTemporalEncoder(nn.Module):
    """VAE-style causal temporal encoder: 49 → 25 → 13 frames, preserves channels.

    Uses causal 3D convolutions (past-only temporal padding) matching the CogVideoX
    VAE encoder's temporal compression pattern.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1: 49 → 25 frames
            CausalConv3d(channels, channels, kernel_size=(3, 3, 3), stride=(2, 1, 1)),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            # Stage 2: 25 → 13 frames
            CausalConv3d(channels, channels, kernel_size=(3, 3, 3), stride=(2, 1, 1)),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )
        self.final = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.final(self.encoder(x))


class CogVideoXControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        vae_channels: int = 16,
        downscale_coef: int = 8,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        out_proj_dim: int = None,
        use_causal_temporal: bool = False,  # Use VAE-style causal temporal encoder (keeps channels, no compression)
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        self.vae_channels = vae_channels
        self.unshuffle = nn.PixelUnshuffle(downscale_coef)

        dino_channels = 384
        self.feature_channels = dino_channels
        if use_causal_temporal:

            self.spatial_downsample = nn.Sequential(
                nn.Conv2d(dino_channels, dino_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, dino_channels),
                nn.SiLU(),
            )

            encoder_channels = dino_channels
            self.feature_channels = dino_channels  # stays 384

            causal_enc = CausalTemporalEncoder(channels=encoder_channels)
            self.temporal_encoder = causal_enc.encoder
            self.temporal_final = causal_enc.final
        else:
            self.spatial_downsample = nn.Sequential(
                nn.Conv2d(dino_channels, dino_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, dino_channels),
                nn.SiLU(),
            )

            self.temporal_encoder = nn.Sequential(
                nn.Conv3d(dino_channels, dino_channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(8, dino_channels),
                nn.SiLU(),

                nn.Conv3d(dino_channels, dino_channels, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(8, dino_channels),
                nn.SiLU(),
            )

            self.temporal_final = nn.Conv3d(dino_channels, dino_channels, kernel_size=1)
            nn.init.zeros_(self.temporal_final.weight)
            nn.init.zeros_(self.temporal_final.bias)

        patch_embed_in_channels = vae_channels + self.feature_channels

        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=patch_embed_in_channels,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        
        self.embedding_dropout = nn.Dropout(dropout)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_projectors = None
        if out_proj_dim is not None:
            proj_out_dim = out_proj_dim
            self.out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, proj_out_dim) for _ in range(num_layers)]
            )

        self.gradient_checkpointing = False

        self.adapter = TemporalFeatureAdapter(latent_channels=self.feature_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: Tuple[torch.Tensor, torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        controlnet_output_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        depth_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        controlnet_states = self.temporal_encoder(controlnet_states.permute(0, 2, 1, 3, 4))
        controlnet_states = self.temporal_final(controlnet_states)
        controlnet_states = controlnet_states.permute(0, 2, 1, 3, 4)

        hidden_states = torch.cat([hidden_states, controlnet_states], dim=2)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        controlnet_hidden_states = ()
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

            if self.out_projectors is not None:
                if controlnet_output_mask is not None:
                    controlnet_hidden_states += (self.out_projectors[i](hidden_states) * controlnet_output_mask,)
                else:
                    controlnet_hidden_states += (self.out_projectors[i](hidden_states),)
            else:
                controlnet_hidden_states += (hidden_states,)

        if not return_dict:
            return (controlnet_hidden_states,)
        return Transformer2DModelOutput(sample=controlnet_hidden_states)
