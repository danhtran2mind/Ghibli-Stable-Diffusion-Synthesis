# Ghibli Fine-Tuned Stable Diffusion 2.1

## Training Notbook

## Dataset
**Source**: [Ghibli Dataset](https://huggingface.co/datasets/uwunish/ghibli-dataset)

## Base Model


## Training Hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-4
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- weight_decay: 0.01
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- num_epochs: 50

## Metric
- Training loss: 0.052300
- Validation loss: 0.006372
- BLEU Score in Validation Set: 0.9964783232500736

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
