# Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion
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
git clone --recursive https://gitlab.tugraz.at/huawei_media/hisilicon/code_release/controldino.git
cd ControlDINO
git lfs install # needed for weights and examples
git lfs pull
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
python download/download.py 

```

### 🚀 Demo Inference
```
./scripts/inference_recon.sh    # reconstruction 
./scripts/inference_style.sh    # styling
./scripts/inference_vkitti.sh   # vkitti

```
## 🙏 Acknowledgements
- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)
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
