from huggingface_hub import snapshot_download
import os
import argparse

def download_dataset(repo_id, output_dir):
    """Download a dataset from Hugging Face to the specified directory."""
    try:
        print(f"Downloading {repo_id} to {output_dir}...")
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Download the dataset
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            # token=token
        )
        print(f"Successfully downloaded {repo_id} to {output_dir}")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uwunish/ghibli-dataset", "pulnip/ghibli-dataset"],
        help="List of dataset IDs to download (e.g., 'user/dataset-name')"
    )
    parser.add_argument(
        "--local-dir",
        default="data",
        help="Base directory for output (default: 'data')"
    )
    parser.add_argument(
        "--huggingface_token",
        default=None,
        help="Hugging Face API token for private datasets (default: None)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Generate output directories from dataset IDs
    output_dirs = [
        os.path.join(args.local_dir, repo_id.replace("/", "-"))
        for repo_id in args.datasets
    ]

    # Log in to Hugging Face if a token is provided
    if args.huggingface_token:
        from huggingface_hub import login
        login(token=args.huggingface_token)
        print("Logged in to Hugging Face with provided token")

    # Download each dataset
    for repo_id, output_dir in zip(args.datasets, output_dirs):
        download_dataset(repo_id, output_dir)

    print("Download process completed.")