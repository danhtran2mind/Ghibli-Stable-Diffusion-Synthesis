import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, UNet2DConditionModel, 
    PNDMScheduler, StableDiffusionPipeline
)

from tqdm import tqdm
from .config_loader import load_model_configs

def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, random_seed, use_lora, 
                  finetune_model_id, lora_model_id, base_model_id, lora_scale, config_path, device, dtype):
    if not prompt or height % 8 != 0 or width % 8 != 0 or num_inference_steps not in range(1, 101) or \
       guidance_scale < 1.0 or guidance_scale > 20.0 or seed < 0 or seed > 4294967295 or \
       (use_lora and (lora_scale < 0.0 or lora_scale > 2.0)):
        return None, "Invalid input parameters."

    model_configs = load_model_configs(config_path)
    finetune_model_path = model_configs.get(finetune_model_id, {}).get('local_dir', finetune_model_id)
    base_model_path = model_configs.get(lora_model_id, {}).get('local_dir', lora_model_id)
    lora_model_path = model_configs.get(base_model_id, {}).get('local_dir', base_model_id)
    # base_model_path lora_model_path
    generator = torch.Generator(device=device).manual_seed(torch.randint(0, 4294967295, (1,)).item() if random_seed else int(seed))

    try:
        if use_lora:
            # Load base pipeline
            pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=dtype, use_safetensors=True)
            
            # Add LoRA weights with specified rank and scale
            pipe.load_lora_weights(lora_model_path, adapter_name="ghibli-lora", 
                                   lora_scale=lora_scale)
            
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