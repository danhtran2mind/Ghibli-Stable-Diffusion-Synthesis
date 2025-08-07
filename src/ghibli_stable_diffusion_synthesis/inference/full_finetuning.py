import os
import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm import tqdm
import yaml

def inference_process(prompt, height, width, num_inference_steps,
                      guidance_scale, batch_size, seed,
                      config_path="configs/model_ckpts.yaml",
                      model_id="danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning",
                      device=None, dtype=torch.float16):
    if not device:
        # Set device and dtype
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            
    # Model path
    all_model_config = yaml.safe_load(open(config_path, "r"))
    model_config = next((config for config in all_model_config if config['model_id'] == model_id), None)
    base_model = (
        model_config['local_dir']
        if os.path.exists(model_config['local_dir']) and any(os.scandir(model_config['local_dir']))
        else model_config['model_id']
    )

    # Load models with consistent dtype
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype).to(device)

    # Load scheduler
    scheduler = PNDMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Validate height and width
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("Height and width must be divisible by 8 for Stable Diffusion.")

    # Create device-specific generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Tokenize and encode prompt
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

    # Initialize latents
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=dtype,
        device=device
    )

    # Set scheduler timesteps
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    # Inference loop
    for t in tqdm(scheduler.timesteps):
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

    # Decode latents to image
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample

    # Convert image to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[0])
    # pil_image.save("ghibli_style_output.png")
    return pil_image  # Return PIL image for further processing or saving