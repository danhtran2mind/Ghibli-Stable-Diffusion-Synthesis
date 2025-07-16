import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm import tqdm
from transformers import HfArgumentParser

def get_examples(examples_dir: str = "assets/examples/ghibli-fine-tuned-sd-2.1") -> list:
    """
    Load example data from the assets/examples directory.
    Each example is a subdirectory containing a config.json and an image file.
    Returns a list of [prompt, height, width, num_inference_steps, guidance_scale, seed, image_path].
    """
    examples = Path(examples_dir)
    ans = []
    for example in examples.iterdir():
        if not example.is_dir():
            continue
        with open(example / "config.json") as f:
            example_dict = json.load(f)
        
        required_keys = ["prompt", "height", "width", "num_inference_steps", "guidance_scale", "seed", "image"]
        if not all(key in example_dict for key in required_keys):
            continue

        example_list = [
            example_dict["prompt"],
            example_dict["height"],
            example_dict["width"],
            example_dict["num_inference_steps"],
            example_dict["guidance_scale"],
            example_dict["seed"],
            str(example / example_dict["image"])  # Path to the image file
        ]
        ans.append(example_list)
    
    if not ans:
        ans = [
            ["a serene landscape in Ghibli style", 64, 64, 50, 3.5, 42, None]
        ]
    return ans

def create_demo(
    model_name: str = "danhtran2mind/ghibli-fine-tuned-sd-2.1",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Convert device string to torch.device
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load models with consistent dtype
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=dtype).to(device)
    scheduler = PNDMScheduler.from_pretrained(model_name, subfolder="scheduler")

    def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed):
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

        batch_size = 1
        if random_seed:
            seed = torch.randint(0, 4294967295, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(int(seed))

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

    def load_example_image(prompt, height, width, num_inference_steps, guidance_scale, seed, image_path):
        """
        Load the image for the selected example and update input fields.
        """
        if image_path and Path(image_path).exists():
            try:
                image = Image.open(image_path)
                return prompt, height, width, num_inference_steps, guidance_scale, seed, image, f"Loaded image: {image_path}"
            except Exception as e:
                return prompt, height, width, num_inference_steps, guidance_scale, seed, None, f"Error loading image: {e}"
        return prompt, height, width, num_inference_steps, guidance_scale, seed, None, "No image available"

    badges_text = r"""
    <div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
    <a href="https://huggingface.co/spaces/danhtran2mind/ghibli-fine-tuned-sd-2.1"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Space&color=orange"></a>
    </div>
    """.strip()

    with gr.Blocks() as demo:
        gr.Markdown("# Ghibli-Style Image Generator")
        gr.Markdown(badges_text)
        gr.Markdown("Generate images in Ghibli style using a fine-tuned Stable Diffusion model. Select an example below to load a pre-generated image or enter a prompt to generate a new one.")
        gr.Markdown("""**Note:** For CPU inference, execution time is long (e.g., for resolution 512 × 512) with 50 inference steps, time is approximately 1700 seconds).""")

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
                generate_btn = gr.Button("Generate Image")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                output_text = gr.Textbox(label="Status")

        examples = get_examples("assets/examples")
        gr.Examples(
            examples=examples,
            inputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, output_image],
            outputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, output_image, output_text],
            fn=load_example_image,
            cache_examples=False
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed],
            outputs=[output_image, output_text]
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ghibli-Style Image Generator using a fine-tuned Stable Diffusion model.")
    parser.add_argument(
        "--local_model",
        action="store_true",
        default=True,
        help="Use local model path instead of Hugging Face model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="danhtran2mind/ghibli-fine-tuned-sd-2.1",
        help="Model name or path for the fine-tuned Stable Diffusion model."
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

    # Set model_name based on local_model flag
    if args.local_model:
        args.model_name = "./checkpoints/ghibli-fine-tuned-sd-2.1"

    demo = create_demo(args.model_name, args.device)
    demo.launch(server_port=args.port, share=args.share)
