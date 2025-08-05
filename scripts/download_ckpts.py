import os
import argparse
import yaml
from huggingface_hub import snapshot_download

def download_model_checkpoint(repo_id, local_dir, token=None):
    """
    Download a Hugging Face model checkpoint to a specified local directory.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., 'stabilityai/stable-diffusion-2-1').
        local_dir (str): The local directory to store the downloaded checkpoint files.
        token (str, optional): Hugging Face API token for accessing private or gated repositories.
    """
    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type="model",
            token=token,
            allow_patterns=["*.safetensors", "*.ckpt", "*.json", "*.txt"]
        )
        print(f"Successfully downloaded model checkpoint from {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Failed to download model checkpoint from {repo_id}: {str(e)}")

def load_config(config_path):
    """
    Load and validate the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        list: List of model configurations for HuggingFace platform.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
        ValueError: If the YAML file is empty or contains no valid HuggingFace entries.
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    if not config_data:
        raise ValueError("The YAML configuration file is empty or invalid")

    # Filter for HuggingFace platform entries
    huggingface_configs = [item for item in config_data if item.get("platform") == "HuggingFace"]
    
    if not huggingface_configs:
        raise ValueError("No valid HuggingFace platform entries found in the YAML configuration")

    return huggingface_configs

if __name__ == "__main__":
    """
    Main function to parse arguments and download model checkpoints based on the YAML configuration.
    """
    parser = argparse.ArgumentParser(description="Download Hugging Face model checkpoints specified in a YAML configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_ckpts.yaml",
        help="Path to the YAML configuration file specifying model IDs and local directories."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token for accessing private or gated repositories (optional)."
    )
    args = parser.parse_args()

    try:
        # Load and validate the configuration
        model_configs = load_config(args.config)

        # Download each model checkpoint
        for config in model_configs:
            repo_id = config.get("model_id")
            local_dir = config.get("local_dir")
            if not repo_id or not local_dir:
                print(f"Skipping invalid configuration entry: missing model_id or local_dir")
                continue
            download_model_checkpoint(repo_id, local_dir, args.token)

    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML configuration file. Details: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")