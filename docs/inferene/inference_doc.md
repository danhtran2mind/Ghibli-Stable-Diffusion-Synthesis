# Inference Arguments Guide

This guide describes the command-line arguments for the `infer.py` script, which performs Stable Diffusion inference using either full fine-tuning or LoRA methods.

## Arguments

| Argument                | Type   | Default Value                              | Description                                                                 |
|-------------------------|--------|--------------------------------------------|-----------------------------------------------------------------------------|
| `--method`              | String | Required (choices: `full_finetuning`, `lora`) | Specifies the inference method: `full_finetuning` or `lora`.                |
| `--prompt`              | String | `"donald trump in ghibli style"`           | Text prompt for image generation.                                           |
| `--height`              | Integer| `512`                                      | Height of the output image (must be divisible by 8).                        |
| `--width`               | Integer| `512`                                      | Width of the output image (must be divisible by 8).                         |
| `--num_inference_steps` | Integer| `50`                                       | Number of inference steps for the generation process.                       |
| `--guidance_scale`      | Float  | `3.5`                                      | Guidance scale for classifier-free guidance.                                |
| `--batch_size`          | Integer| `1`                                        | Batch size for inference.                                                  |
| `--seed`                | Integer| `42`                                       | Random seed for reproducibility.                                           |
| `--lora_scale`          | Float  | `1.2`                                      | Scaling factor for LoRA weights (applicable when using LoRA method).        |
| `--config_path`         | String | `"configs/model_ckpts.yaml"`               | Path to the model configuration YAML file.                                  |
| `--output_path`         | String | `"test_data/ghibli_style_{method}_output.png"`               | Path to save the output image.                                  |


## Usage Example

- To run inference with full fine-tuning:
```bash
python infer.py --method full_finetuning --prompt "donald trump in ghibli style" --height 512 --width 512 --num_inference_steps 50 --guidance_scale 3.5 --batch_size 1 --seed 42 --config_path configs/model_ckpts.yaml
```
  The output_path is `tests/test_data/ghibli_style_output_full_finetuning.png`.
- To run inference with LoRA:
```bash
python infer.py --method lora --prompt "donald trump in ghibli style" --height 512 --width 512 --num_inference_steps 50 --guidance_scale 3.5 --batch_size 1 --seed 42 --lora_scale 1.2 --config_path configs/model_ckpts.yaml
```
  The output_path is `tests/test_data/ghibli_style_output_lora.png`.
## Notes
- The output image is saved as `test_data/ghibli_style_{method}_output.png`, where `{method}` is either `full_finetuning` or `lora`.
- Ensure the `--height` and `--width` values are divisible by 8 to avoid errors in the Stable Diffusion pipeline.
- The `--lora_scale` argument is only relevant when using the LoRA method.
- The script uses different model IDs based on the method:
  - Full fine-tuning: `danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning`
  - LoRA: `danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA`