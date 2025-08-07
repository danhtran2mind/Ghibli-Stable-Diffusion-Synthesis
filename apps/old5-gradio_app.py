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
    with open(config_path, 'r') as f:
        return {cfg['model_id']: cfg for cfg in yaml.safe_load(f)}

def get_examples(examples_dir: Union[str, List[str]] = None, use_lora: bool = None) -> List:
    directories = [examples_dir] if isinstance(examples_dir, str) else examples_dir or []
    valid_dirs = [d for d in directories if os.path.isdir(d)]
    if not valid_dirs:
        return get_provided_examples(use_lora)

    examples = []
    for dir_path in valid_dirs:
        for subdir in sorted(os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))):
            config_path = os.path.join(subdir, "config.json")
            image_path = os.path.join(subdir, "result.png")
            if not (os.path.isfile(config_path) and os.path.isfile(image_path)):
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            required_keys = ["prompt", "height", "width", "num_inference_steps", "guidance_scale", "seed", "image"]
            if config.get("use_lora", False):
                required_keys.extend(["lora_model_id", "base_model_id", "lora_rank", "lora_scale"])
            else:
                required_keys.append("finetune_model_id")

            if set(required_keys) - set(config.keys()) or config["image"] != "result.png":
                continue

            try:
                image = Image.open(image_path)
            except Exception:
                continue

            if use_lora is not None and config.get("use_lora", False) != use_lora:
                continue

            example = [config["prompt"], config["height"], config["width"], config["num_inference_steps"], 
                       config["guidance_scale"], config["seed"], image]
            example.extend([config["lora_model_id"], config["base_model_id"], config["lora_rank"], config["lora_scale"]] 
                          if config.get("use_lora", False) else [config["finetune_model_id"]])
            examples.append(example)

    return examples or get_provided_examples(use_lora)

def get_provided_examples(use_lora: bool = False) -> list:
    example_path = f"apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-{'LoRA' if use_lora else 'Base-finetuning'}/1/result.png"
    image = Image.open(example_path) if os.path.exists(example_path) else None
    return [[
        "a cat is laying on a sofa in Ghibli style" if use_lora else "a serene landscape in Ghibli style",
        512, 768 if use_lora else 512, 100 if use_lora else 50, 10.0 if use_lora else 3.5, 789 if use_lora else 42,
        image, "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA" if use_lora else "danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning",
        "stabilityai/stable-diffusion-2-1" if use_lora else None, 64 if use_lora else None, 0.9 if use_lora else None
    ]]

def create_demo(config_path: str = "configs/model_ckpts.yaml", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model_configs = load_model_configs(config_path)
    finetune_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning'), None)
    lora_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora'), None)
    
    if not finetune_model_id or not lora_model_id:
        raise ValueError("Missing model IDs in config.")

    finetune_model_path = model_configs[finetune_model_id].get('local_dir', finetune_model_id)
    lora_model_path = model_configs[lora_model_id].get('local_dir', lora_model_id)
    base_model_id = model_configs[lora_model_id].get('base_model_id', 'stabilityai/stable-diffusion-2-1')
    base_model_path = model_configs.get(base_model_id, {}).get('local_dir', base_model_id)

    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed, use_lora, 
                      finetune_model_id, lora_model_id, base_model_id, lora_rank, lora_scale):
        if not prompt or height % 8 != 0 or width % 8 != 0 or num_inference_steps not in range(1, 101) or \
           guidance_scale < 1.0 or guidance_scale > 20.0 or seed < 0 or seed > 4294967295 or \
           (use_lora and (lora_rank < 1 or lora_rank > 128 or lora_scale < 0.0 or lora_scale > 2.0)):
            return None, "Invalid input parameters."

        model_configs = load_model_configs(config_path)
        finetune_model_path = model_configs.get(finetune_model_id, {}).get('local_dir', finetune_model_id)
        lora_model_path = model_configs.get(lora_model_id, {}).get('local_dir', lora_model_id)
        base_model_path = model_configs.get(base_model_id, {}).get('local_dir', base_model_id)

        generator = torch.Generator(device=device).manual_seed(torch.randint(0, 4294967295, (1,)).item() if random_seed else int(seed))

        try:
            if use_lora:
                pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=dtype, use_safetensors=True)
                pipe.load_lora_weights(lora_model_path, adapter_name="ghibli-lora", lora_scale=lora_scale)
                pipe = pipe.to(device)
                vae, tokenizer, text_encoder, unet, scheduler = pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet, PNDMScheduler.from_config(pipe.scheduler.config)
            else:
                vae = AutoencoderKL.from_pretrained(finetune_model_path, subfolder="vae", torch_dtype=dtype).to(device)
                tokenizer = CLIPTokenizer.from_pretrained(finetune_model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(finetune_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
                unet = UNet2DConditionModel.from_pretrained(finetune_model_path, subfolder="unet", torch_dtype=dtype).to(device)
                scheduler = PNDMScheduler.from_pretrained(finetune_model_path, subfolder="scheduler")

            text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)

            uncond_input = tokenizer([""] * 1, padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=dtype)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8), generator=generator, dtype=dtype, device=device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma

            for t in tqdm(scheduler.timesteps, desc="Generating image"):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            image = vae.decode(latents / vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
            pil_image = Image.fromarray((image[0] * 255).round().astype("uint8"))

            if use_lora:
                del pipe
            else:
                del vae, tokenizer, text_encoder, unet, scheduler
            torch.cuda.empty_cache()

            return pil_image, f"Generated image successfully! Seed used: {seed}"
        except Exception as e:
            return None, f"Failed to generate image: {e}"

    def load_example_image_full_finetuning(prompt, height, width, num_inference_steps, guidance_scale, seed, image, finetune_model_id):
        return prompt, height, width, num_inference_steps, guidance_scale, seed, image, finetune_model_id, "Loaded example successfully"

    def load_example_image_lora(prompt, height, width, num_inference_steps, guidance_scale, seed, image, lora_model_id, base_model_id, lora_rank, lora_scale):
        return prompt, height, width, num_inference_steps, guidance_scale, seed, image, lora_model_id, base_model_id or "stabilityai/stable-diffusion-2-1", lora_rank or 64, lora_scale or 1.2, "Loaded example successfully"

    badges_text = """
    <div style="text-align: left; font-size: 14px; display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; align-items: center; justify-content: left; gap: 8px;">
            GitHub: <a href="https://github.com/danhtran2mind/Ghibli-Stable-Diffusion-Synthesis">
                <img src="https://img.shields.io/badge/GitHub-danhtran2mind%2FGhibli--Stable--Diffusion--Synthesis-blue?style=flat&logo=github" alt="GitHub Repo">
            </a> HuggingFace: 
            <a href="https://huggingface.co/spaces/danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning">
                <img src="https://img.shields.io/badge/HuggingFace-danhtran2mind%2FGhibli--Stable--Diffusion--2.1--Base--finetuning-yellow?style=flat&logo=huggingface" alt="HuggingFace Space Demo">
            </a>
            <a href="https://huggingface.co/spaces/danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA">
                <img src="https://img.shields.io/badge/HuggingFace-danhtran2mind%2FGhibli--Stable--Diffusion--2.1--LoRA-yellow?style=flat&logo=huggingface" alt="HuggingFace Space Demo">
            </a>
        </div>
    </div>
    """

    custom_css = open("apps/gradio_app/static/styles.css", "r").read() if os.path.exists("apps/gradio_app/static/styles.css") else ""

    examples_full_finetuning = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning", use_lora=False)
    examples_lora = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-LoRA", use_lora=True)

    with gr.Blocks(css=custom_css, theme="ocean") as demo:
        gr.Markdown("## Ghibli-Style Image Generator")
        with gr.Tabs():
            with gr.Tab(label="Full Finetuning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Generation Settings")
                        prompt_ft = gr.Textbox(label="Prompt", placeholder="e.g., 'a serene landscape in Ghibli style'", lines=2)
                        with gr.Group():
                            gr.Markdown("#### Image Dimensions")
                            with gr.Row():
                                width_ft = gr.Slider(32, 4096, 512, step=8, label="Width")
                                height_ft = gr.Slider(32, 4096, 512, step=8, label="Height")
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps_ft = gr.Slider(1, 100, 50, step=1, label="Inference Steps")
                            guidance_scale_ft = gr.Slider(1.0, 20.0, 3.5, step=0.5, label="Guidance Scale")
                            random_seed_ft = gr.Checkbox(label="Use Random Seed")
                            seed_ft = gr.Slider(0, 4294967295, 42, step=1, label="Seed")
                        gr.Markdown("#### Model Configuration")
                        finetune_model_path_ft = gr.Dropdown(label="Fine-tuned Model", choices=[mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning'], value=finetune_model_id)
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Result")
                        output_image_ft = gr.Image(label="Generated Image", interactive=False, height=512)
                        output_text_ft = gr.Textbox(label="Status", interactive=False, lines=3)
                        generate_btn_ft = gr.Button("Generate Image", variant="primary")
                        stop_btn_ft = gr.Button("Stop Generation")
                gr.Markdown("### Examples for Full Finetuning")
                gr.Examples(examples=examples_full_finetuning, inputs=[prompt_ft, height_ft, width_ft, num_inference_steps_ft, guidance_scale_ft, seed_ft, output_image_ft, finetune_model_path_ft], 
                            outputs=[prompt_ft, height_ft, width_ft, num_inference_steps_ft, guidance_scale_ft, seed_ft, output_image_ft, finetune_model_path_ft, output_text_ft], 
                            fn=load_example_image_full_finetuning, cache_examples=False, examples_per_page=4)

            with gr.Tab(label="LoRA"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Generation Settings")
                        prompt_lora = gr.Textbox(label="Prompt", placeholder="e.g., 'a serene landscape in Ghibli style'", lines=2)
                        with gr.Group():
                            gr.Markdown("#### Image Dimensions")
                            with gr.Row():
                                width_lora = gr.Slider(32, 4096, 512, step=8, label="Width")
                                height_lora = gr.Slider(32, 4096, 512, step=8, label="Height")
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps_lora = gr.Slider(1, 100, 50, step=1, label="Inference Steps")
                            guidance_scale_lora = gr.Slider(1.0, 20.0, 3.5, step=0.5, label="Guidance Scale")
                            lora_rank_lora = gr.Slider(1, 128, 64, step=1, label="LoRA Rank")
                            lora_scale_lora = gr.Slider(0.0, 2.0, 1.2, step=0.1, label="LoRA Scale")
                            random_seed_lora = gr.Checkbox(label="Use Random Seed")
                            seed_lora = gr.Slider(0, 4294967295, 42, step=1, label="Seed")
                        gr.Markdown("#### Model Configuration")
                        lora_model_path_lora = gr.Dropdown(label="LoRA Model", choices=[mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora'], value=lora_model_id)
                        base_model_path_lora = gr.Dropdown(label="Base Model", choices=[model_configs[mid].get('base_model_id') for mid in model_configs if model_configs[mid].get('base_model_id')], value=base_model_id)
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Result")
                        output_image_lora = gr.Image(label="Generated Image", interactive=False, height=512)
                        output_text_lora = gr.Textbox(label="Status", interactive=False, lines=3)
                        generate_btn_lora = gr.Button("Generate Image", variant="primary")
                        stop_btn_lora = gr.Button("Stop Generation")
                gr.Markdown("### Examples for LoRA")
                gr.Examples(examples=examples_lora, inputs=[prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora, output_image_lora, lora_model_path_lora, base_model_path_lora, lora_rank_lora, lora_scale_lora], 
                            outputs=[prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora, output_image_lora, lora_model_path_lora, base_model_path_lora, lora_rank_lora, lora_scale_lora, output_text_lora], 
                            fn=load_example_image_lora, cache_examples=False, examples_per_page=4)

        gr.Markdown(badges_text)

        generate_event_ft = generate_btn_ft.click(fn=generate_image, inputs=[prompt_ft, height_ft, width_ft, num_inference_steps_ft, guidance_scale_ft, seed_ft, random_seed_ft, gr.State(False), finetune_model_path_ft, gr.State(None), gr.State(None), gr.State(None), gr.State(None)], 
                                                 outputs=[output_image_ft, output_text_ft])
        generate_event_lora = generate_btn_lora.click(fn=generate_image, inputs=[prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora, random_seed_lora, gr.State(True), gr.State(None), lora_model_path_lora, base_model_path_lora, lora_rank_lora, lora_scale_lora], 
                                                     outputs=[output_image_lora, output_text_lora])

        stop_btn_ft.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_ft])
        stop_btn_lora.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_lora])

        demo.unload(lambda: torch.cuda.empty_cache())

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghibli-Style Image Generator")
    parser.add_argument("--config_path", type=str, default="configs/model_ckpts.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = create_demo(args.config_path, args.device)
    demo.launch(server_port=args.port, share=args.share)