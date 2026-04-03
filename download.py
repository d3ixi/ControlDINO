from huggingface_hub import snapshot_download

def download_model():
    snapshot_download(
        repo_id="THUDM/CogVideoX-5b-I2V",
        local_dir="./ckpts/CogVideoX-5b-I2V",
        local_dir_use_symlinks=False,
    )

download_model()
