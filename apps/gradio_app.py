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
    Load examples from the specified directory.
    Returns a list of examples with validated image paths and required fields.
    """
    print(f"Checking examples directory: {examples_dir}")
    if not os.path.exists(examples_dir) or not os.path.isdir(examples_dir):
        print(f"Error: Directory {examples_dir} does not exist or is not a directory")
        fallback_image_path = "apps/gradio_app/assets/examples/default_image.png"
        if not os.path.isfile(fallback_image_path):
            print(f"Fallback image {fallback_image_path} not found")
            fallback_image_path = None
        return [
            ["a serene landscape in Ghibli style", 512, 512, 50, 7.5, 42, fallback_image_path, False,
             "stabilityai/stable-diffusion-2-1-base", None, None, None, None]
        ]

    all_examples_dir = [os.path.join(examples_dir, d) for d in os.listdir(examples_dir) 
                        if os.path.isdir(os.path.join(examples_dir, d))]
    print(f"Found example directories: {all_examples_dir}")
    ans = []

    for example_dir in sorted(all_examples_dir):
        config_path = os.path.join(example_dir, "config.json")
        image_path = os.path.join(example_dir, "result.png")
        print(f"Processing example directory: {example_dir}")
        print(f"Config path: {config_path}, Image path: {image_path}")
        
        if not os.path.isfile(config_path):
            print(f"Error: config.json not found in {example_dir}")
            continue
        if not os.path.isfile(image_path):
            print(f"Error: result.png not found in {example_dir}")
            continue
        
        try:
            with open(config_path, 'r') as f:
                example_dict = json.load(f)
            print(f"Loaded config: {example_dict}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading or parsing {config_path}: {e}")
            continue
        
        required_keys = ["prompt", "height", "width", "num_inference_steps", "guidance_scale", "seed", "image"]
        if example_dict.get("use_lora", False):
            required_keys.extend(["lora_model_id", "base_model_id", "lora_rank", "lora_scale"])
        else:
            required_keys.append("finetune_model_id")
        
        if not all(key in example_dict for key in required_keys):
            print(f"Error: Missing required keys in {config_path}: {', '.join(set(required_keys) - set(example_dict.keys()))}")
            continue
        
        if example_dict["image"] != "result.png":
            print(f"Error: Image key in {config_path} does not match 'result.png'")
            continue
        
        try:
            if not Path(image_path).is_file():
                print(f"Error: Image file {image_path} does not exist")
                continue
                
            try:
                Image.open(image_path).verify()
                print(f"Image verified: {image_path}")
            except Exception as e:
                print(f"Error: Invalid image file {image_path}: {e}")
                continue
                
            example_list = [
                example_dict["prompt"],
                example_dict["height"],
                example_dict["width"],
                example_dict["num_inference_steps"],
                example_dict["guidance_scale"],
                example_dict["seed"],
                image_path,
                example_dict.get("use_lora", False),
                example_dict.get("finetune_model_id", None),
                example_dict.get("lora_model_id", None),
                example_dict.get("base_model_id", None),
                example_dict.get("lora_rank", None),
                example_dict.get("lora_scale", None)
            ]
            ans.append(example_list)
        except KeyError as e:
            print(f"Error processing {config_path}: Missing key {e}")
            continue
    
    if not ans:
        print("No valid examples found, using default example")
        fallback_image_path = "apps/gradio_app/assets/examples/default_image.png"
        if not os.path.isfile(fallback_image_path):
            print(f"Fallback image {fallback_image_path} not found")
            fallback_image_path = None
        ans = [
            ["a serene landscape in Ghibli style", 512, 512, 50, 7.5, 42, fallback_image_path, False,
             "stabilityai/stable-diffusion-2-1-base", None, None, None, None]
        ]

    for an in ans:
        print(f"Final loaded example: {an}")
    return ans

def create_demo(
    config_path: str = "configs/model_ckpts.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model_configs = load_model_configs(config_path)
    print("Loaded model_configs:", model_configs)
    
    finetune_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning'), None)
    lora_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora'), None)
    
    if not finetune_model_id or not lora_model_id:
        raise ValueError("Could not find full_finetuning or lora model IDs in the configuration file.")

    finetune_config = model_configs.get(finetune_model_id, {})
    finetune_local_dir = finetune_config.get('local_dir')
    if finetune_local_dir and os.path.exists(finetune_local_dir) and any(os.path.isfile(os.path.join(finetune_local_dir, f)) for f in os.listdir(finetune_local_dir)):
        finetune_model_path = finetune_local_dir
    else:
        print(f"Local model directory for fine-tuned model '{finetune_model_id}' does not exist or is empty at '{finetune_local_dir}'. Falling back to model ID.")
        finetune_model_path = finetune_model_id

    lora_config = model_configs.get(lora_model_id, {})
    lora_local_dir = lora_config.get('local_dir')
    if lora_local_dir and os.path.exists(lora_local_dir) and any(os.path.isfile(os.path.join(lora_local_dir, f)) for f in os.listdir(lora_local_dir)):
        lora_model_path = lora_local_dir
    else:
        print(f"Local model directory for LoRA model '{lora_model_id}' does not exist or is empty at '{lora_local_dir}'. Falling back to model ID.")
        lora_model_path = lora_model_id

    base_model_id = lora_config.get('base_model_id', 'stabilityai/stable-diffusion-2-1')
    base_model_config = model_configs.get(base_model_id, {})
    base_local_dir = base_model_config.get('local_dir')
    if base_local_dir and os.path.exists(base_local_dir) and any(os.path.isfile(os.path.join(base_local_dir, f)) for f in os.listdir(base_local_dir)):
        base_model_path = base_local_dir
    else:
        print(f"Local model directory for base model '{base_model_id}' does not exist or is empty at '{base_local_dir}'. Falling back to model ID.")
        base_model_path = base_model_id

    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    finetune_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning']
    lora_model_ids = [mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora']
    base_model_ids = [model_configs[mid].get('base_model_id') for mid in model_configs if model_configs[mid].get('base_model_id')]

    def update_model_path_visibility(use_lora):
        if use_lora:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
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
                pipe.to("cpu")
                del pipe
            else:
                del vae, tokenizer, text_encoder, unet, scheduler
            torch.cuda.empty_cache()

            return pil_image, f"Generated image successfully! Seed used: {seed}"
        except Exception as e:
            print(f"Error in generate_image: {e}")
            return None, f"Failed to generate image: {e}"

    def load_example_image(prompt, height, width, num_inference_steps, guidance_scale,
                          seed, image_path, use_lora, finetune_model_id, lora_model_id,
                          base_model_id, lora_rank, lora_scale):
        print("Starting load_example_image function")
        print(f"Received inputs: prompt={prompt}, image_path={image_path}, use_lora={use_lora}")
        try:
            image = None
            status = "No image available"
            if image_path and os.path.isfile(image_path):
                print(f"Attempting to load image: {image_path}")
                try:
                    image = Image.open(image_path)
                    image.load()  # Force load to catch errors
                    status = f"Loaded example image: {image_path}"
                    print(f"Image loaded successfully: {image_path}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    status = f"Error: Invalid image file {image_path}: {e}"
            else:
                print(f"Image path invalid or missing: {image_path}")
                status = f"Error: Image path {image_path} does not exist or is None"
        
            print(f"Returning outputs: image={image is not None}, status={status}")
            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                image, use_lora, finetune_model_id if not use_lora else None,
                lora_model_id if use_lora else None, base_model_id if use_lora else None,
                lora_rank if use_lora else None, lora_scale if use_lora else None, status
            )
        except Exception as e:
            print(f"Error in load_example_image: {e}")
            return (
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                None, use_lora, finetune_model_id if not use_lora else None,
                lora_model_id if use_lora else None, base_model_id if use_lora else None,
                lora_rank if use_lora else None, lora_scale if use_lora else None,
                f"Error loading example: {e}"
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

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Image Generation Settings")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., 'a serene landscape in Ghibli style'",
                    lines=2
                )
                with gr.Group():
                    gr.Markdown("### Image Dimensions")
                    with gr.Row():
                        width = gr.Slider(
                            minimum=32, maximum=4096, value=512, step=8, label="Width"
                        )
                        height = gr.Slider(
                            minimum=32, maximum=4096, value=512, step=8, label="Height"
                        )
                with gr.Accordion("Advanced Settings", open=False):
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1, label="Inference Steps",
                        info="Higher steps improve quality but increase generation time."
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0, maximum=20.0, value=3.5, step=0.5, label="Guidance Scale",
                        info="Controls how closely the image follows the prompt."
                    )
                    lora_rank = gr.Slider(
                        minimum=1, maximum=128, value=64, step=1, visible=False, label="LoRA Rank",
                        info="Controls model complexity and memory usage."
                    )
                    lora_scale = gr.Slider(
                        minimum=0.0, maximum=2.0, value=1.2, step=0.1, visible=False, label="LoRA Scale",
                        info="Adjusts the influence of LoRA weights."
                    )
                    random_seed = gr.Checkbox(label="Use Random Seed", value=False)
                    seed = gr.Slider(
                        minimum=0, maximum=4294967295, value=42, step=1,
                        label="Seed (0–4294967295)", info="Set a specific seed for reproducible results."
                    )
                with gr.Group():
                    gr.Markdown("### Model Configuration")
                    use_lora = gr.Checkbox(
                        label="Use LoRA Weights", value=False,
                        info="Enable to use LoRA weights with a base model."
                    )
                    finetune_model_path = gr.Dropdown(
                        label="Fine-tuned Model", choices=finetune_model_ids,
                        value=finetune_model_id, visible=not use_lora.value
                    )
                    lora_model_path = gr.Dropdown(
                        label="LoRA Model", choices=lora_model_ids,
                        value=lora_model_id, visible=use_lora.value
                    )
                    base_model_path = gr.Dropdown(
                        label="Base Model", choices=base_model_ids,
                        value=base_model_id, visible=use_lora.value
                    )
                image_path = gr.Textbox(visible=False)
                generate_btn = gr.Button("Generate Image", variant="primary")
                stop_btn = gr.Button("Stop Generation")

            with gr.Column(scale=1):
                gr.Markdown("## Generated Result")
                output_image = gr.Image(label="Generated Image", interactive=False, height=512)
                output_text = gr.Textbox(label="Status", interactive=False, lines=3)
                test_btn = gr.Button("Test Example Load")  # Added test button

        gr.Markdown("## Try an Example")
        examples = get_examples()
        gr.Examples(
            examples=examples,
            inputs=[
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                image_path, use_lora, finetune_model_path, lora_model_path, base_model_path,
                lora_rank, lora_scale
            ],
            outputs=[
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                output_image, use_lora, finetune_model_path, lora_model_path,
                base_model_path, lora_rank, lora_scale, output_text
            ],
            fn=load_example_image,
            cache_examples=False,
            label="Example Prompts and Images",
            examples_per_page=5
        )

        gr.Markdown(badges_text)

        use_lora.change(
            fn=update_model_path_visibility,
            inputs=use_lora,
            outputs=[lora_model_path, base_model_path, finetune_model_path, lora_rank, lora_scale]
        )

        generate_event = generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                random_seed, use_lora, finetune_model_path, lora_model_path,
                base_model_path, lora_rank, lora_scale
            ],
            outputs=[output_image, output_text]
        )

        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[generate_event])

        test_btn.click(
            fn=load_example_image,
            inputs=[
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                image_path, use_lora, finetune_model_path, lora_model_path,
                base_model_path, lora_rank, lora_scale
            ],
            outputs=[
                prompt, height, width, num_inference_steps, guidance_scale, seed,
                output_image, use_lora, finetune_model_path, lora_model_path,
                base_model_path, lora_rank, lora_scale, output_text
            ]
        )

        def cleanup():
            print("Cleaning up resources...")
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