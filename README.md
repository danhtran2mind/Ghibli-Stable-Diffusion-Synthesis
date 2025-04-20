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
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Gradio Demo
```bash
python app.py --local_model True  # Use True for local model, False to download from HuggingFace
```
The app will be available at `localhost:7860`.

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
