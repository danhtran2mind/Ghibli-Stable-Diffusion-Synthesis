# Scripts Guide

This guide provides an overview of two Python scripts for downloading model checkpoints and datasets from Hugging Face: `download_ckpts.py` and `download_datasets.py`. Below, you'll find descriptions of each script, their purpose, and the command-line arguments they accept.

## 1. `download_ckpts.py`

### Purpose
This script downloads model checkpoints from Hugging Face repositories based on a YAML configuration file. It supports downloading specific file types (e.g., `.safetensors`, `.ckpt`, `.json`, `.txt`) and allows excluding certain file patterns. It also supports authentication for private or gated repositories using a Hugging Face API token.

### Usage
```bash
python scripts/download_ckpts.py --config <path_to_yaml> --token <huggingface_token>
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | `str` | `configs/model_ckpts.yaml` | Path to the YAML configuration file specifying model IDs and local directories. |
| `--token` | `str` | `None` | Hugging Face API token for accessing private or gated repositories (optional). |

### YAML Configuration
The script expects a YAML file (e.g., `model_ckpts.yaml`) with entries for Hugging Face models. Each entry must include:
- `platform`: Must be set to `"HuggingFace"`.
- `model_id`: The Hugging Face repository ID (e.g., `stabilityai/stable-diffusion-2-1`).
- `local_dir`: The local directory where the checkpoint files will be saved.
- `no_download_path` (optional): A list of file patterns to exclude from downloading.

**Example YAML Configuration:**
```yaml
- platform: HuggingFace
  model_id: stabilityai/stable-diffusion-2-1
  local_dir: ./checkpoints/stable-diffusion-2-1
  no_download_path: ["*.bin"]
- platform: HuggingFace
  model_id: runwayml/stable-diffusion-v1-5
  local_dir: ./checkpoints/stable-diffusion-v1-5
```

### Example Command
```bash
python scripts/download_ckpts.py --config configs/model_ckpts.yaml --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Notes
- The script creates the `local_dir` if it doesn't exist.
- Only files matching the allowed patterns (`.safetensors`, `.ckpt`, `.json`, `.txt`) are downloaded unless overridden by `no_download_path`.
- If the YAML file is missing, empty, or lacks valid Hugging Face entries, the script will raise an error.

---

## 2. `download_datasets.py`

### Purpose
This script downloads datasets from Hugging Face to specified local directories. It supports downloading multiple datasets at once and allows authentication for private datasets using a Hugging Face API token.

### Usage
```bash
python scripts/download_datasets.py --datasets <dataset_id_1> <dataset_id_2> --local-dir <output_directory> --huggingface_token <token>
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasets` | `list` | `["uwunish/ghibli-dataset", "pulnip/ghibli-dataset"]` | List of dataset IDs to download (e.g., `user/dataset-name`). |
| `--local-dir` | `str` | `data` | Base directory for output. Each dataset is saved in a subdirectory named after the dataset ID (e.g., `data/user-dataset-name`). |
| `--huggingface_token` | `str` | `None` | Hugging Face API token for accessing private datasets (optional). |

### Example Command
```bash
python scripts/download_datasets.py --datasets uwunish/ghibli-dataset pulnip/ghibli-dataset --local-dir datasets --huggingface_token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### Notes
- The script creates a subdirectory for each dataset in the format `<local-dir>/<user>-<dataset-name>`.
- If a token is provided, the script logs in to Hugging Face to access private datasets.
- The script ensures the output directory exists before downloading.

---

### Common Notes
- Both scripts use the `huggingface_hub` library's `snapshot_download` function to download files.
- Ensure you have the required Python packages installed:
  ```bash
  pip install huggingface_hub pyyaml
  ```
- For private or gated repositories/datasets, a valid Hugging Face API token is required. You can obtain one from your Hugging Face account settings.
- The scripts handle errors such as missing files, invalid YAML, or download failures, printing appropriate error messages.