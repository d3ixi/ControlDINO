import os
import torch
import torchvision.transforms as transforms
import numpy as np
from safetensors.torch import load_file
from decord import VideoReader
from torch.utils.data.dataset import Dataset

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

class RealEstate10KPCDRenderDataset(Dataset):
    def __init__(
            self,
            video_root_dir,
            sample_n_frames=49,
            image_size=[480, 720],
    ):
        root_path = video_root_dir
        self.root_path = root_path
        self.sample_n_frames = sample_n_frames
        self.source_video_root = os.path.join(self.root_path, 'videos')
        self.mask_video_root = os.path.join(self.root_path, 'masked_videos')
        self.captions_root = os.path.join(self.root_path, 'captions')
        self.dataset = sorted([n.replace('.mp4','') for n in os.listdir(self.source_video_root)])
        self.length = len(self.dataset)
        sample_size = image_size
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size

        pixel_transforms = [transforms.Resize(sample_size),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = pixel_transforms

        self.dinov3_model = load_dinov3()

    def load_video_reader(self, idx):
        clip_name = self.dataset[idx]

        video_path = os.path.join(self.source_video_root, clip_name + '.mp4')
        video_reader = VideoReader(video_path)
        mask_video_path = os.path.join(self.mask_video_root, clip_name + '.mp4')
        if os.path.exists(mask_video_path):
            mask_video_reader = VideoReader(mask_video_path)
        else:
            mask_video_reader = video_reader  # fallback: anchor = source
        caption_path = os.path.join(self.captions_root, clip_name + '.txt')
        if os.path.exists(caption_path):
            caption = open(caption_path, 'r').read().strip()
        else:
            caption = ''
        return clip_name, video_reader, mask_video_reader, caption

    def get_batch(self, idx):
        clip_name, video_reader, mask_video_reader, video_caption = self.load_video_reader(idx)

        indices = np.minimum(np.arange(self.sample_n_frames), len(video_reader) - 1)
        pixel_values = torch.from_numpy(video_reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        mask_indices = np.minimum(np.arange(self.sample_n_frames), len(mask_video_reader) - 1)
        anchor_pixels = torch.from_numpy(mask_video_reader.get_batch(mask_indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
        anchor_pixels = anchor_pixels / 255.

        frames = (pixel_values) * 2 - 1
        frames = frames.unsqueeze(dim=0).contiguous().cuda()  # [F, 3, H, W] in [-1, 1]
        features = compute_dinov3_features_2x(frames, self.dinov3_model, dtype=frames.dtype).squeeze(dim=0)

        return pixel_values, anchor_pixels, video_caption, clip_name, features

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                video, anchor_video, video_caption, clip_name, features = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
        for transform in self.pixel_transforms:
            video = transform(video)
            anchor_video = transform(anchor_video)
        data = {
            'video': video, 
            'anchor_video': anchor_video,
            'caption': video_caption, 
            'features': features
        }
        return data
    

