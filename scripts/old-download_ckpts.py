import os
import argparse
import yaml
from huggingface_hub import snapshot_download

def download_repository(repo_id, local_dir, token=None):
    """
    Download a Hugging Face repository to a local directory using snapshot_download.
    
    Args:
        repo_id (str): Hugging Face repository ID.
        local_dir (str): Local directory to save the downloaded files.
        token (str, optional): Hugging Face API token for private/gated repositories.
    """
    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type="model",
            token=token if token else None,
            allow_patterns=["*.safetensors", "*.ckpt", "*.json", "*.txt"]
        )
        print(f"Successfully downloaded all files from {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading repository {repo_id}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face repositories to local directories.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_ckpts.yaml",
        help="Path to the YAML configuration file containing model IDs and local directories"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token for private or gated repositories (optional)"
    )

    args = parser.parse_args()

    # Load the YAML configuration file
    try:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError("The YAML file is empty or invalid")

        # Extract repo_ids and local_dirs for HuggingFace platform only
        repo_ids = []
        local_dirs = []
        for item in config_data:
            if item.get("platform") == "HuggingFace":
                repo_ids.append(item["model_id"])
                local_dirs.append(item["local_dir"])

        # Validate that at least one valid entry was found
        if not repo_ids or not local_dirs:
            raise ValueError("No valid HuggingFace platform entries found in the YAML file")

        # Validate that the number of repo_ids matches the number of local_dirs
        if len(repo_ids) != len(local_dirs):
            raise ValueError("The number of model_ids must match the number of local_dirs for HuggingFace platform")

        # Download each repository
        for repo_id, local_dir in zip(repo_ids, local_dirs):
            download_repository(repo_id, local_dir, args.token)
            
    except FileNotFoundError:
        print(f"Error: The configuration file '{args.config}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. Details: {e}")
    except KeyError as e:
        print(f"Error: Missing expected key in YAML data. Details: {e}")
    except ValueError as e:
        print(f"Error: {e}")