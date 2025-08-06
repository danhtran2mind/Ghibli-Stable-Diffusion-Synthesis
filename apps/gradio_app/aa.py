import argparse
import json
from typing import Union, List
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

def get_examples(examples_dir: Union[str, List[str]] = None,
                 use_lora: Union[bool, None] = None) -> List:
    # Convert single string to list
    directories = [examples_dir] if isinstance(examples_dir, str) else examples_dir or []

    # Validate directories
    valid_dirs = [d for d in directories if os.path.isdir(d)]
    if not valid_dirs:
        print("Error: No valid directories found, using provided examples")
        return get_provided_examples(use_lora)

    examples = []
    for dir_path in valid_dirs:
        # Get sorted subdirectories
        subdirs = sorted(
            os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))
        )

        for subdir in subdirs:
            config_path = os.path.join(subdir, "config.json")
            image_path = os.path.join(subdir, "result.png")

            if not (os.path.isfile(config_path) and os.path.isfile(image_path)):
                print(f"Error: Missing config.json or result.png in {subdir}")
                continue

            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {config_path}: {e}")
                continue

            required_keys = ["prompt", "height", "width", "num_inference_steps", "guidance_scale", "seed", "image"]
            if config.get("use_lora", False):
                required_keys.extend(["lora_model_id", "base_model_id", "lora_rank", "lora_scale"])
            else:
                required_keys.append("finetune_model_id")

            if missing_keys := set(required_keys) - set(config.keys()):
                print(f"Error: Missing keys in {config_path}: {', '.join(missing_keys)}")
                continue

            if config["image"] != "result.png":
                print(f"Error: Image key in {config_path} does not match 'result.png'")
                continue

            try:
                Image.open(image_path).verify()
                image = Image.open(image_path)  # Re-open after verify
            except Exception as e:
                print(f"Error: Invalid image {image_path}: {e}")
                continue

            if use_lora is not None and config.get("use_lora", False) != use_lora:
                print(f"DEBUG: Skipping {config_path} due to use_lora mismatch (expected {use_lora}, got {config.get('use_lora', False)})")
                continue

            # Build example list based on use_lora
            example = [
                config["prompt"],
                config["height"],
                config["width"],
                config["num_inference_steps"],
                config["guidance_scale"],
                config["seed"],
                image,
                # config.get("use_lora", False)
            ]
            if config.get("use_lora", False):
                example.extend([
                    config["lora_model_id"],
                    config["base_model_id"],
                    config["lora_rank"],
                    config["lora_scale"]
                ])
            else:
                example.append(config["finetune_model_id"])

            examples.append(example)
            print(f"DEBUG: Loaded example from {config_path}: {example[:6]}")

    return examples or get_provided_examples(use_lora)

def get_provided_examples(use_lora: bool = False) -> list:
    example1_image = None
    example2_image = None
    # Attempt to load example images
    if use_lora:
        try:
            example2_path = "apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-LoRA/1/result.png"
            if os.path.exists(example2_path):
                example2_image = Image.open(example2_path)
        except Exception as e:
            print(f"Failed to load example2 image: {e}")
        output = [list({
            "prompt": "a cat is laying on a sofa in Ghibli style",
            "width": 512,
            "height": 768,
            "steps": 100,
            "cfg_scale": 10.0,
            "seed": 789,
            "image": example2_path, # example2_image,
            # "use_lora": True,
            "model": "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA",
            "base_model": "stabilityai/stable-diffusion-2-1",
            "lora_rank": 64,
            "lora_alpha": 0.9
        }.values())]

    else:
        try:
            example1_path = "apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning/1/result.png"
            if os.path.exists(example1_path):
                example1_image = Image.open(example1_path)
        except Exception as e:
            print(f"Failed to load example1 image: {e}")
        output = [list({
            "prompt": "a serene landscape in Ghibli style",
            "width": 256,
            "height": 512,
            "steps": 50,
            "cfg_scale": 3.5,
            "seed": 42,
            "image": example1_path, # example1_image,
            # "use_lora": False,
            "model": "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"
        }.values())]
    
    return output

def create_demo(
    config_path: str = "configs/model_ckpts.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model_configs = load_model_configs(config_path)

    finetune_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning'), None)
    lora_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora'), None)

    if not finetune_model_id or not lora_model_id:
        raise ValueError("Could not find full_finetuning or lora model IDs in the configuration file.")

    finetune_config = model_configs.get(finetune_model_id, {})
    finetune_local_dir = finetune_config.get('local_dir')
    if finetune_local_dir and os.path.exists(finetune_local_dir) and any(os.path.isfile(os.path.join(finetune_local_dir, f)) for f in os.listdir(finetune_local_dir)):
        finetune_model_path = finetune_local_dir
    else:
        finetune_model_path = finetune_model_id

    lora_config = model_configs.get(lora_model_id, {})
    lora_local_dir = lora_config.get('local_dir')
    if lora_local_dir and os.path.exists(lora_local_dir) and any(os.path.isfile(os.path.join(lora_local_dir, f)) for f in os.listdir(lora_local_dir)):
        lora_model_path = lora_local_dir
    else:
        lora_model_path = lora_model_id

    base_model_id = lora_config.get('base_model_id', 'stabilityai/stable-diffusion-2-1')
    base_model_config = model_configs.get(base_model_id, {})
    base_local_dir = base_model_config.get('local_dir')
    if base_local_dir and os.path.exists(base_local_dir) and any(os.path.isfile(os.path.join(base_local_dir, f)) for f in os.listdir(base_local_dir)):
        base_model_path = base_local_dir
    else:
        base_model_path = base_model_id

    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    finetune_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning']
    lora_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora']
    base_model_ids = [model_configs[mid].get('base_model_id') for mid in model_configs if model_configs[mid].get('base_model_id')]

    def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed, use_lora, finetune_model_id, lora_model_id, base_model_id, lora_rank, lora_scale):
        try:
            model_configs = load_model_configs(config_path)
            finetune_config = model_configs.get(finetune_model_id, {})
            finetune_local_dir = finetune_config.get('local_dir')
            finetune_model_path = finetune_local_dir if finetune_local_dir and os.path.exists(finetune_local_dir) and any(os.path.isfile(os.path.join(finetune_local_dir, f)) for f in os.listdir(finetune_local_dir)) else finetune_model_id

            lora_config = model_configs.get(lora_model_id, {})
            lora_local_dir = lora_config.get('local_dir')
            lora_model_path = lora_local_dir if lora_local_dir and os.path.exists(lora_local_dir) and any(os.path.isfile(os.path.join(lora_local_dir, f)) for f in os.listdir(lora_local_dir)) else lora_model_id

            base_model_config = model_configs.get(base_model_id, {})
            base_local_dir = base_model_config.get('local_dir')
            base_model_path = base_local_dir if base_local_dir and os.path.exists(base_local_dir) and any(os.path.isfile(os.path.join(base_local_dir, f)) for f in os.listdir(base_local_dir)) else base_model_id

            if not prompt:
                return None, "Prompt cannot be empty."
            if height % 8 != 0 or width % 8 != 0:
                return None, "Height and width must be divisible by 8."
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

            if use_lora:
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_model_path, torch_dtype=dtype, use_safetensors=True
                    )
                    pipe.load_lora_weights(lora_model_path, adapter_name="ghibli-lora", lora_scale=lora_scale)
                    pipe = pipe.to(device)
                    vae = pipe.vae
                    tokenizer = pipe.tokenizer
                    text_encoder = pipe.text_encoder
                    unet = pipe.unet
                    scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
                except Exception as e:
                    return None, f"Error loading LoRA model: {e}"
            else:
                try:
                    vae = AutoencoderKL.from_pretrained(finetune_model_path, subfolder="vae", torch_dtype=dtype).to(device)
                    tokenizer = CLIPTokenizer.from_pretrained(finetune_model_path, subfolder="tokenizer")
                    text_encoder = CLIPTextModel.from_pretrained(finetune_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
                    unet = UNet2DConditionModel.from_pretrained(finetune_model_path, subfolder="unet", torch_dtype=dtype).to(device)
                    scheduler = PNDMScheduler.from_pretrained(finetune_model_path, subfolder="scheduler")
                except Exception as e:
                    return None, f"Error loading fine-tuned model: {e}"

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
                generator=generator, dtype=dtype, device=device
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

            if use_lora:
                del pipe
            else:
                del vae, tokenizer, text_encoder, unet, scheduler
            torch.cuda.empty_cache()

            return pil_image, f"Generated image successfully! Seed used: {seed}"
        except Exception as e:
            return None, f"Failed to generate image: {e}"

    def load_example_image_full_finetuning(prompt, height, width, num_inference_steps, guidance_scale,
                                          seed, image, finetune_model_id):
        try:
            status = "Loaded example successfully"
            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                image, finetune_model_id, status
            )
        except Exception as e:
            print(f"DEBUG: Exception in load_example_image: {e}")
            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                None, finetune_model_id,
                f"Error loading example: {e}"
            )

    def load_example_image_lora(prompt, height, width, num_inference_steps, guidance_scale,
                               seed, image, lora_model_id,
                               base_model_id, lora_rank, lora_scale):
        try:
            status = "Loaded example successfully"
            # Ensure base_model_id, lora_rank, and lora_scale have valid values
            base_model_id = base_model_id or "stabilityai/stable-diffusion-2-1"
            lora_rank = lora_rank if lora_rank is not None else 64
            lora_scale = lora_scale if lora_scale is not None else 1.2

            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                image, lora_model_id, base_model_id,
                lora_rank, lora_scale, status
            )
        except Exception as e:
            print(f"DEBUG: Exception in load_example_image_lora: {e}")
            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                None, lora_model_id, base_model_id or "stabilityai/stable-diffusion-2-1",
                lora_rank or 64, lora_scale or 1.2, f"Error loading example: {e}"
            )

    badges_text = r"""
    <div style="text-align: left; font-size: 14px; display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; align-items: center; justify-content: left; gap: 8px;">
            You can explore GitHub repository:
            <a href="https://github.com/danhtran2mind/Ghibli-Stable-Diffusion-Synthesis">
                <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2FGhibli--Stable--Diffusion--Synthesis-blue?style=flat&logo=github" alt="GitHub Repo">
            </a>. And you can explore HuggingFace Model Hub:
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

    try:
        custom_css = open("apps/gradio_app/static/styles.css", "r").read()
    except FileNotFoundError:
        print("Error: styles.css not found, using default styling")
        custom_css = ""

    examples_full_finetuning = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning",
                                             use_lora=False)
    examples_lora = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-LoRA", 
                                 use_lora=True)
    
    with gr.Blocks(css=custom_css, theme="ocean") as demo:
        gr.Markdown("## Ghibli-Style Image Generator")
        with gr.Tabs():
            with gr.Tab(label="Full Finetuning"):
                with gr.Row():
                    with gr.Column(scale.=1):
                        gr.Markdown("### Image Generation Settings")
                        prompt_ft = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., 'a serene landscape in Ghibli style'",
                            lines=2
                        )
                        with gr.Group():
                            gr.Markdown("#### Image Dimensions")
                            with gr.Row():
                                width_ft = gr.Slider(
                                    minimum=32, maximum=4096, value=512, step=8, label="Width"
                                )
                                height_ft = gr.Slider(
                                    minimum=32, maximum=4096, value=512, step=8, label="Height"
                                )
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps_ft = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Inference Steps",
                                info="More steps, better quality, longer wait."
                            )
                            guidance_scale_ft = gr.Slider(
                                minimum=1.0, maximum=20.0, value=3.5, step=0.5, label="Guidance Scale",
                                info="Controls how closely the image follows the prompt."
                            )
                            random_seed_ft = gr.Checkbox(label="Use Random Seed", value=False)
                            seed_ft = gr.Slider(
                                minimum=0, maximum=4294967295, value=42, step=1,
                                label="Seed", info="Use a seed (0-4294967295) for consistent results."
                            )
                        with gr.Group():
                            gr.Markdown("#### Model Configuration")
                            finetune_model_path_ft = gr.Dropdown(
                                label="Fine-tuned Model", choices=finetune_model_ids,
                                value=finetune_model_id
                            )
                        # image_path_ft = gr.Textbox(visible=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Result")
                        output_image_ft = gr.Image(label="Generated Image", interactive=False, height=512)
                        output_text_ft = gr.Textbox(label="Status", interactive=False, lines=3)

                        generate_btn_ft = gr.Button("Generate Image", variant="primary")
                        stop_btn_ft = gr.Button("Stop Generation")

                gr.Markdown("### Examples for Full Finetuning")
                gr.Examples(
                    examples=examples_full_finetuning,
                    inputs=[
                        prompt_ft, height_ft, width_ft, num_inference_steps_ft, 
                        guidance_scale_ft, seed_ft, output_image_ft, finetune_model_path_ft
                    ],
                    outputs=[prompt_ft, height_ft, width_ft, num_inference_steps_ft,
                             guidance_scale_ft, seed_ft, output_image_ft, finetune_model_path_ft,
                             output_text_ft],
                    fn=load_example_image_full_finetuning,
                    # fn=lambda *args: load_example_image_full_finetuning(*args),
                    cache_examples=False,
                    label="Examples for Full Fine-tuning",
                    examples_per_page=4
                )

            with gr.Tab(label="LoRA"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Generation Settings")
                        prompt_lora = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., 'a serene landscape in Ghibli style'",
                            lines=2
                        )
                        with gr.Group():
                            gr.Markdown("#### Image Dimensions")
                            with gr.Row():
                                width_lora = gr.Slider(
                                    minimum=32, maximum=4096, value=512, step=8, label="Width"
                                )
                                height_lora = gr.Slider(
                                    minimum=32, maximum=4096, value=512, step=8, label="Height"
                                )
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps_lora = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Inference Steps",
                                info="More steps, better quality, longer wait."
                            )
                            guidance_scale_lora = gr.Slider(
                                minimum=1.0, maximum=20.0, value=3.5, step=0.5, label="Guidance Scale",
                                info="Controls how closely the image follows the prompt."
                            )
                            lora_rank_lora = gr.Slider(
                                minimum=1, maximum=128, value=64, step=1, label="LoRA Rank",
                                info="Controls model complexity and memory usage."
                            )
                            lora_scale_lora = gr.Slider(
                                minimum=0.0, maximum=2.0, value=1.2, step=0.1, label="LoRA Scale",
                                info="Adjusts the influence of LoRA weights."
                            )
                            random_seed_lora = gr.Checkbox(label="Use Random Seed", value=False)
                            seed_lora = gr.Slider(
                                minimum=0, maximum=4294967295, value=42, step=1,
                                label="Seed", info="Use a seed (0-4294967295) for consistent results."
                            )
                        with gr.Group():
                            gr.Markdown("#### Model Configuration")
                            lora_model_path_lora = gr.Dropdown(
                                label="LoRA Model", choices=lora_model_ids,
                                value=lora_model_id
                            )
                            base_model_path_lora = gr.Dropdown(
                                label="Base Model", choices=base_model_ids,
                                value=base_model_id
                            )
                        # image_path_lora = gr.Textbox(visible=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Result")
                        output_image_lora = gr.Image(label="Generated Image", interactive=False, height=512)
                        output_text_lora = gr.Textbox(label="Status", interactive=False, lines=3)

                        generate_btn_lora = gr.Button("Generate Image", variant="primary")
                        stop_btn_lora = gr.Button("Stop Generation")

                gr.Markdown("### Examples for LoRA")
                gr.Examples(
                    examples=examples_lora,
                    inputs=[
                        prompt_lora, height_lora, width_lora, num_inference_steps_lora, 
                        guidance_scale_lora, seed_lora, output_image_lora,      
                        lora_model_path_lora, base_model_path_lora,
                        lora_rank_lora, lora_scale_lora
                    ],
                    outputs=[
                        prompt_lora, height_lora, width_lora, num_inference_steps_lora,
                        guidance_scale_lora, seed_lora, output_image_lora,
                        lora_model_path_lora, base_model_path_lora,
                        lora_rank_lora, lora_scale_lora,
                        output_text_lora
                    ],
                    fn=load_example_image_lora,
                    # fn=lambda *args: load_example_image_lora(*args),
                    cache_examples=False,
                    label="Examples for LoRA",
                    examples_per_page=4
                )

        gr.Markdown(badges_text)

        generate_event_ft = generate_btn_ft.click(
            fn=generate_image,
            inputs=[
                prompt_ft, height_ft, width_ft, num_inference_steps_ft, guidance_scale_ft, seed_ft,
                random_seed_ft, gr.State(value=False), finetune_model_path_ft, gr.State(value=None),
                gr.State(value=None), gr.State(value=None), gr.State(value=None)
            ],
            outputs=[output_image_ft, output_text_ft]
        )

        generate_event_lora = generate_btn_lora.click(
            fn=generate_image,
            inputs=[
                prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora,
                random_seed_lora, gr.State(value=True), gr.State(value=None), lora_model_path_lora,
                base_model_path_lora, lora_rank_lora, lora_scale_lora
            ],
            outputs=[output_image_lora, output_text_lora]
        )

        stop_btn_ft.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_ft])
        stop_btn_lora.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_lora])

        def cleanup():
            print("DEBUG: Cleaning up resources...")
            torch.cuda.empty_cache()

        demo.unload(cleanup)

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
