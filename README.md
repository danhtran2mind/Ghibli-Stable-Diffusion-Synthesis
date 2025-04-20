# Ghibli Fine-Tuned Stable Diffusion 2.1

## Training Notbook

## Dataset
**Source**: [Ghibli Dataset](https://huggingface.co/datasets/uwunish/ghibli-dataset)

## Base Model
**Source**: [HuggingFace Model](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

## Training Hyperparameters

The following hyperparameters were used during training:

- learning_rate: 1e-05
- num_train_epochs: 40
- train_batch_size: 2
- gradient_accumulation_steps: 2
- mixed_precision: "fp16"
- resolution: 512
- max_grad_norm: 1
- lr_scheduler: "constant"
- lr_warmup_steps: 0
- checkpoints_total_limit: 1
- mixed_precision
- use_ema
- use_8bit_adam
- center_crop
- random_flip
- gradient_checkpointing
- 
## Metric
- Loss: 0.0345

## Dependencies Version

## Demonstration
- **Demo URL**: [Ghibli Fine-Tuned SD 2.1](https://huggingface.co/spaces/danhtran2mind/ghibli-fine-tuned-sd-2.1)
- **Preview Image**:  
  ![Demo Image](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/blob/main/visualization/demo_image.png?raw=true)

## Usage

### Encrypt Data in dataset and diffusers Folders

To decrypt the dataset and diffusers folders, which are encrypted using git-crypt, contact me in this repository [Issues tab](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/issues) to obtain the key. This data serves as a backup in case the original links are deleted.

To decrypt the dataset and diffusers folders if they are encrypted using git-crypt:

```bash
git-crypt unlock /path/to/my-repo.asc
```
Ensure the path to the .asc key file is correct. This step assumes that git-crypt is installed and configured for the repository.

### Install Dependencies

```bash
pip install -r requirements.txt
```
### Download Model to Local `ghibli-fine-tuned-sd-2.1` Folder

To download the fine-tuned model weights to the ghibli-fine-tuned-sd-2.1 folder:

```bash
cd ghibli-fine-tuned-sd-2.1
python download_model.py
cd ..
```

The download_model.py script is assumed to handle downloading the model from the HuggingFace repository.

### Extract `dataset` Folder

To download and extract the Ghibli dataset from HuggingFace:

```bash
cd dataset
pip install datasets  # Ensure the HuggingFace datasets library is installed
python extract_files.py
cd ..
```

The extract_files.py script is assumed to handle downloading and extracting the dataset. Alternatively, you can manually download the dataset from the HuggingFace dataset page or use the following Python code:

```python
from datasets import load_dataset
dataset = load_dataset("uwunish/ghibli-dataset")
dataset.save_to_disk("./ghibli_dataset")
```

### Extract `diffusers` Folder

To extract the fine-tuned model weights or related files in the diffusers folder:

```bash
cd diffusers
python extract_files.py
cd ..
```

The extract_files.py script in the diffusers folder is assumed to extract the fine-tuned model weights or related files, possibly downloaded as part of the download_model.py step. If the diffusers folder contains the model weights directly, verify its contents after downloading.

### Run Gradio Demo

```bash
python app.py --local_model True  # Use True for local model, False to download from HuggingFace
```

The app will be available at localhost:7860.

## Environment
- **Python Version**: 3.11.11
- **Dependencies**:
  - huggingface-hub v0.30.2
  - accelerate v1.3.0
  - bitsandbytes v0.45.5
  - torch v2.5.1
  - Pillow v11.1.0
  - numpy v1.26.4
  - transformers v4.51.1
  - torchvision v0.20.1
  - diffusers v0.33.1
  - gradio
