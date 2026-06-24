# Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion
<h2 align="center">ECCV 2026</h2>

[![Project Website](https://img.shields.io/badge/Project-ControlDINO-blue)](https://dedoardo.github.io/projects/control-dino)  [![Static Badge](https://img.shields.io/badge/arXiv-2604.01761-d31101)](https://arxiv.org/abs/2604.01761)

[Edoardo A. Dominici](https://dedoardo.github.io/)<sup>&#42;</sup>, 
[Thomas Deixelberger]()<sup>&#42;</sup>,
[Konstantinos Vardis](https://kostasvardis.com/), 
[Markus Steinberger](https://www.markussteinberger.net/)
<br> 
<sup>&#42;</sup> denotes equal contribution

![Teaser](./assets/teaser.gif)

## 📦 Setup
### 1. Clone ControlDINO
```
git clone --recursive https://github.com/d3ixi/ControlDINO.git
cd ControlDINO
```

### 2. Setup environments
tested with:
- **Python**: 3.10
- **Torch**: 2.11.0+cu129
- **Hardware**: H200
```
conda create -n ControlDINO python=3.10 -y
conda activate ControlDINO
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

### 3. Downloading Pretrained Models
Download [CogVideoX-5B-I2V](https://github.com/THUDM/CogVideo) (Base Model)
```
python download.py 
```

Download Control-DINO weights:
```
conda install -c conda-forge git-lfs
git lfs install
git lfs pull
```

### 🚀 Demo Inference
```
./scripts/inference_recon.sh    # reconstruction 
./scripts/inference_style.sh    # styling
./scripts/inference_vkitti.sh   # vkitti
```

### ⭐ Gradio App
```
python app.py
```

## 🙏 Acknowledgements
- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)
- This project also leverages [DINOv3](https://github.com/facebookresearch/dinov3)
- We thank the authors for making their dataset publicly available:
  * [ScanNet++](https://github.com/scannetpp/scannetpp)
  * [DL3DV](https://github.com/DL3DV-10K/Dataset)
  * [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)

## 📖 Citation

```
@article{dominici2026controldinofeaturespaceconditioning,
      title={Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion}, 
      author={Edoardo A. Dominici and Thomas Deixelberger and Konstantinos Vardis and Markus Steinberger},
      year={2026},
      eprint={2604.01761},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2604.01761}, 
}
```
<a href="https://www.tugraz.at/en/home"><img height="100" src="assets/tugraz-logo.jpg"> </a>
<img height="100" src="assets/huawei-logo.jpg">
