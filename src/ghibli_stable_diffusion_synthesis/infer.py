##Full##
import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm

# Set device and dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16  # Use float16 for CUDA
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32  # Use float32 for MPS
else:
    device = torch.device("cpu")
    dtype = torch.float32  # Use float32 for CPU

# Model path
base_model = "ckpts/Ghibli-Stable-Diffusion-2.1-Base-finetuning"

# Load models with consistent dtype
vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=dtype).to(device)

# Load scheduler
scheduler = PNDMScheduler.from_pretrained(base_model, subfolder="scheduler")

# Configurable parameters
prompt = ["donald trump in ghibli style"]
height = 512  # Configurable height (must be divisible by 8)
width = 512   # Configurable width (must be divisible by 8)
num_inference_steps = 50
guidance_scale = 3.5
batch_size = 1
seed = 42

# Validate height and width
if height % 8 != 0 or width % 8 != 0:
    raise ValueError("Height and width must be divisible by 8 for Stable Diffusion.")

# Create device-specific generator
generator = torch.Generator(device=device).manual_seed(seed)

# Tokenize and encode prompt
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
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

# Initialize latents with configurable height and width
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),  # Use unet.config.in_channels
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

# Convert image to PIL for saving/viewing
image = (image / 2 + 0.5).clamp(0, 1)  # Clamp as a PyTorch tensor
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()  # Convert to NumPy
image = (image * 255).round().astype("uint8")  # Scale and convert to uint8
pil_image = Image.fromarray(image[0])  # Convert to PIL Image
pil_image.save("ghibli_style_girl.png")

##lora##

import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from tqdm import tqdm

# Set device and dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16  # Use float16 for CUDA
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype = torch.float32  # Use float32 for MPS
else:
    device = torch.device("cpu")
    dtype = torch.float32  # Use float32 for CPU

# Model paths
base_model = "stabilityai/stable-diffusion-2-1"
lora_model = "danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA"

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=dtype,
    use_safetensors=True
)

# Load LoRA weights
lora_rank = 64  # Example rank
pipe.load_lora_weights(lora_model, adapter_name="ghibli-lora", lora_scale=1.2)

# Move pipeline to device
pipe = pipe.to(device)

# Load individual components for consistency with section 1
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

# Configurable parameters
prompt = ["a girl is on a street on a beach"]
height = 512  # Configurable height (must be divisible by 8)
width = 512   # Configurable width (must be divisible by 8)
num_inference_steps = 25
guidance_scale = 3.5
batch_size = 1
seed = 42

# Validate height and width
if height % 8 != 0 or width % 8 != 0:
    raise ValueError("Height and width must be divisible by 8 for Stable Diffusion.")

# Create device-specific generator
generator = torch.Generator(device=device).manual_seed(seed)

# Tokenize and encode prompt
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
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

# Initialize latents with configurable height and width
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

# Convert image to PIL for saving/viewing
image = (image / 2 + 0.5).clamp(0, 1)  # Clamp as a PyTorch tensor
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()  # Convert to NumPy
image = (image * 255).round().astype("uint8")  # Scale and convert to uint8
pil_image = Image.fromarray(image[0])  # Convert to PIL Image
pil_image.save("ghibli_style_beach_girl.png")
image[0]