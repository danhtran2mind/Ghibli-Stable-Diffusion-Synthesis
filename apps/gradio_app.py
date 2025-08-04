import argparse
import json
from pathlib import Path
import os
import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from tqdm import tqdm
import yaml

def load_model_configs(config_path: str = "configs/model_ckpts.yaml") -> dict:
    """
    Load model configurations from a YAML file.
    Returns a dictionary with model IDs and their details.
    """
    try:
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        return {cfg['model_id']: cfg for cfg in configs}
    except (IOError, yaml.YAMLError) as e:
        raise ValueError(f"Error loading {config_path}: {e}")

def get_examples(examples_dir: str = "apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning") -> list:
    """
    Load example data from the assets/examples directory.
    Each example is a subdirectory containing a config.json and an image file.
    Returns a list of [prompt, height, width, num_inference_steps, guidance_scale, seed, image_path, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale].
    """
    if not os.path.exists(examples_dir) or not os.path.isdir(examples_dir):
        raise ValueError(f"Directory {examples_dir} does not exist or is not a directory")

    all_examples_dir = [os.path.join(examples_dir, d) for d in os.listdir(examples_dir) 
                        if os.path.isdir(os.path.join(examples_dir, d))]

    ans = []
    for example_dir in all_examples_dir:
        config_path = os.path.join(example_dir, "config.json")
        image_path = os.path.join(example_dir, "result.png")
        
        if not os.path.isfile(config_path):
            print(f"Warning: config.json not found in {example_dir}")
            continue
        if not os.path.isfile(image_path):
            print(f"Warning: result.png not found in {example_dir}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                example_dict = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading or parsing {config_path}: {e}")
            continue
        
        required_keys = ["prompt", "height", "width", "num_inference_steps", "guidance_scale", "seed", "image"]
        if not all(key in example_dict for key in required_keys):
            print(f"Warning: Missing required keys in {config_path}")
            continue
        
        if example_dict["image"] != "result.png":
            print(f"Warning: Image key in {config_path} does not match 'result.png'")
            continue
        
        try:
            example_list = [
                example_dict["prompt"],
                example_dict["height"],
                example_dict["width"],
                example_dict["num_inference_steps"],
                example_dict["guidance_scale"],
                example_dict["seed"],
                image_path,
                example_dict.get("use_lora", False),
                example_dict.get("finetune_model_path", "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"),
                example_dict.get("lora_model_path", "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA"),
                example_dict.get("base_model_path", "stabilityai/stable-diffusion-2-1"),
                example_dict.get("lora_rank", 64),
                example_dict.get("lora_scale", 1.2)
            ]
            ans.append(example_list)
        except KeyError as e:
            print(f"Error processing {config_path}: Missing key {e}")
            continue
    
    if not ans:
        model_configs = load_model_configs("configs/model_ckpts.yaml")
        finetune_model_id = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"
        lora_model_id = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA"
        base_model_id = model_configs[lora_model_id]['base_model_id'] if lora_model_id in model_configs else "stabilityai/stable-diffusion-2-1"
        ans = [
            ["a serene landscape in Ghibli style", 512, 512, 50, 3.5, 42, None, False,
             model_configs.get(finetune_model_id, {}).get('local_dir', finetune_model_id),
             model_configs.get(lora_model_id, {}).get('local_dir', lora_model_id),
             base_model_id, 64, 1.2]
        ]
    return ans

def create_demo(
    config_path: str = "configs/model_ckpts.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load model configurations
    model_configs = load_model_configs(config_path)
    finetune_model_id = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"
    lora_model_id = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA"
    finetune_model_path = model_configs[finetune_model_id]['local_dir'] if model_configs[finetune_model_id]['platform'] == "Local" else finetune_model_id
    lora_model_path = model_configs[lora_model_id]['local_dir'] if model_configs[lora_model_id]['platform'] == "Local" else lora_model_id
    base_model_path = model_configs[lora_model_id]['base_model_id']

    # Convert device string to torch.device
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Extract model IDs for dropdown choices based on type
    finetune_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full-finetuning']
    lora_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora']
    base_model_ids = [model_configs[mid]['base_model_id'] for mid in model_configs if 'base_model_id' in model_configs[mid]]

    def update_model_path_visibility(use_lora):
        """
        Update visibility of model path dropdowns based on use_lora checkbox.
        """
        if use_lora:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
    def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale):
        if not prompt:
            return None, "Prompt cannot be empty."
        if height % 8 != 0 or width % 8 != 0:
            return None, "Height and width must be divisible by 8 (e.g., 256, 512, 1024)."
        if num_inference_steps < 1 or num_inference_steps > 100:
            return None, "Number of inference steps must be between 1 and 100."
        if guidance_scale < 1.0 or guidance_scale > 20.0:
            return None, "Guidance scale must be between 1.0 and 20.0."
        if seed < 0 or seed > 4294967295:
            return None, "Seed must be between 0 and 4294967295."
        if use_lora and (not lora_model_path or not os.path.exists(lora_model_path) and not lora_model_path.startswith("danhtran2mind/")):
            return None, f"LoRA model path {lora_model_path} does not exist or is invalid."
        if use_lora and (not base_model_path or not os.path.exists(base_model_path) and not base_model_path.startswith("stabilityai/")):
            return None, f"Base model path {base_model_path} does not exist or is invalid."
        if not use_lora and (not finetune_model_path or not os.path.exists(finetune_model_path) and not finetune_model_path.startswith("danhtran2mind/")):
            return None, f"Fine-tuned model path {finetune_model_path} does not exist or is invalid."
        if use_lora and (lora_rank < 1 or lora_rank > 128):
            return None, "LoRA rank must be between 1 and 128."
        if use_lora and (lora_scale < 0.0 or lora_scale > 2.0):
            return None, "LoRA scale must be between 0.0 and 2.0."

        batch_size = 1
        if random_seed:
            seed = torch.randint(0, 4294967295, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Load models based on use_lora
        if use_lora:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    base_model_path,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
                pipe.load_lora_weights(lora_model_path, adapter_name="ghibli-lora", lora_scale=lora_scale)
                pipe = pipe.to(device)
                vae = pipe.vae
                tokenizer = pipe.tokenizer
                text_encoder = pipe.text_encoder
                unet = pipe.unet
                scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
            except Exception as e:
                return None, f"Error loading LoRA model from {lora_model_path} or base model from {base_model_path}: {e}"
        else:
            try:
                vae = AutoencoderKL.from_pretrained(finetune_model_path, subfolder="vae", torch_dtype=dtype).to(device)
                tokenizer = CLIPTokenizer.from_pretrained(finetune_model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(finetune_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
                unet = UNet2DConditionModel.from_pretrained(finetune_model_path, subfolder="unet", torch_dtype=dtype).to(device)
                scheduler = PNDMScheduler.from_pretrained(finetune_model_path, subfolder="scheduler")
            except Exception as e:
                return None, f"Error loading fine-tuned model from {finetune_model_path}: {e}"

        text_input = tokenizer(
            [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=dtype)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            dtype=dtype,
            device=device
        )

        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma

        for t in tqdm(scheduler.timesteps, desc="Generating image"):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                else:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            latents = latents / vae.config.scaling_factor
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])

        return pil_image, f"Image generated successfully! Seed used: {seed}"

    def load_example_image(prompt, height, width, num_inference_steps, guidance_scale, seed, image_path, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale):
        """
        Load the image for the selected example and update input fields.
        """
        if image_path and Path(image_path).exists():
            try:
                image = Image.open(image_path)
                return (
                    prompt, height, width, num_inference_steps, guidance_scale, seed, image,
                    use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale,
                    f"Loaded image: {image_path}"
                )
            except Exception as e:
                return (
                    prompt, height, width, num_inference_steps, guidance_scale, seed, None,
                    use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale,
                    f"Error loading image: {e}"
                )
        return (
            prompt, height, width, num_inference_steps, guidance_scale, seed, None,
            use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale,
            "No image available"
        )

    badges_text = r"""
    <div style="text-align: left; font-size: 14px; display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; align-items: center; justify-content: left; gap: 8px;">
            You can explore GitHub repository: 
            <a href="https://github.com/danhtran2mind/Ghibli-Stable-Diffusion-Synthesis">
                <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2FGhibli--Stable--Diffusion--Synthesis-blue?style=flat&logo=github" alt="GitHub Repo">
            </a>.
        </div>
        <div style="display: flex; align-items: center; justify-content: left; gap: 8px;">
            And you can explore HuggingFace Model Hub:
            <a href="https://huggingface.co/spaces/danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning">
                <img src="https://img.shields.io/badge/HuggingFace-danhtran2mind%2FGhibli--Stable--Diffusion--2.1--Base--finetuning-yellow?style=flat&logo=huggingface" alt="HuggingFace Space Demo">
            </a>
            and
            <a href="https://huggingface.co/spaces/danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA">
                <img src="https://img.shields.io/badge/HuggingFace-danhtran2mind%2FGhibli--Stable--Diffusion--2.1--LoRA-yellow?style=flat&logo=huggingface" alt="HuggingFace Space Demo">
            </a>
        </div>
    </div>
    """.strip()

    with gr.Blocks() as demo:
        gr.Markdown("# Ghibli-Style Image Generator")
        gr.Markdown(badges_text)
        gr.Markdown("Generate images in Ghibli style using a fine-tuned Stable Diffusion model or Stable Diffusion 2.1 with LoRA weights. Select an example below to load a pre-generated image or enter a prompt to generate a new one.")
        gr.Markdown("""**Note:** For CPU inference, execution time is long (e.g., for resolution 512 × 512 with 50 inference steps, time is approximately 1700 seconds).""")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="e.g., 'a serene landscape in Ghibli style'")
                with gr.Row():
                    width = gr.Slider(32, 4096, 512, step=8, label="Generation Width")
                    height = gr.Slider(32, 4096, 512, step=8, label="Generation Height")
                with gr.Accordion("Advanced Options", open=False):
                    num_inference_steps = gr.Slider(1, 100, 50, step=1, label="Number of Inference Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, 3.5, step=0.5, label="Guidance Scale")
                    seed = gr.Number(42, label="Seed (0 to 4294967295)")
                    random_seed = gr.Checkbox(label="Use Random Seed", value=False)
                    use_lora = gr.Checkbox(label="Use LoRA Weights", value=False)
                    finetune_model_path = gr.Dropdown(
                        label="Fine-tuned Model Path",
                        choices=finetune_model_ids,
                        value=finetune_model_id,
                        visible=not use_lora.value
                    )
                    lora_model_path = gr.Dropdown(
                        label="LoRA Model Path",
                        choices=lora_model_ids,
                        value=lora_model_id,
                        visible=use_lora.value
                    )
                    base_model_path = gr.Dropdown(
                        label="Base Model Path",
                        choices=base_model_ids,
                        value=base_model_path,
                        visible=use_lora.value
                    )
                    lora_rank = gr.Slider(1, 128, 64, step=1, label="LoRA Rank", visible=use_lora.value)
                    lora_scale = gr.Slider(0.0, 2.0, 1.2, step=0.1, label="LoRA Scale", visible=use_lora.value)
                generate_btn = gr.Button("Generate Image")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                output_text = gr.Textbox(label="Status")

        examples = get_examples("assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning")
        gr.Examples(
            examples=examples,
            inputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, output_image, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale],
            outputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, output_image, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale, output_text],
            fn=load_example_image,
            cache_examples=False
        )

        use_lora.change(
            fn=update_model_path_visibility,
            inputs=use_lora,
            outputs=[lora_model_path, base_model_path, finetune_model_path]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed, use_lora, finetune_model_path, lora_model_path, base_model_path, lora_rank, lora_scale],
            outputs=[output_image, output_text]
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghibli-Style Image Generator using a fine-tuned Stable Diffusion model or Stable Diffusion 2.1 with LoRA weights.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_ckpts.yaml",
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app on."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Set to True for public sharing (Hugging Face Spaces)."
    )

    args = parser.parse_args()

    demo = create_demo(args.config_path, args.device)
    demo.launch(server_port=args.port, share=args.share)