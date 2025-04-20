# Ghibli Fine-Tuned Stable Diffusion 2.1

## Demonstration
- **Demo URL**: [Ghibli Fine-Tuned SD 2.1](https://huggingface.co/spaces/danhtran2mind/ghibli-fine-tuned-sd-2.1)
- **Preview Image**: [Insert Image Here]

## Dataset
- **Source**: [Ghibli Dataset](https://huggingface.co/datasets/uwunish/ghibli-dataset)

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
