import os
from huggingface_hub import HfApi, hf_hub_download

# Specify the repository and local directory
repo_id = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"
local_dir = os.path.join("ckpts", "Ghibli-Stable-Diffusion-2.1-Base-finetuning")

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Initialize Hugging Face API
api = HfApi()

# Get list of all files in the repository
repo_files = api.list_repo_files(repo_id=repo_id)

# Download each file
for file in repo_files:
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Downloaded {file} to {local_dir}")
