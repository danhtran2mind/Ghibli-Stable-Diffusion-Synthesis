import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 
                                             'src', 'ghibli_stable_diffusion_synthesis', 
                                             'inference')))

from inference.full_finetuning import inference_process as full_finetuning_inference
from inference.lora import inference_process as lora_inference

def generate_image(prompt, height, width, num_inference_steps, guidance_scale, seed, 
                   random_seed, use_lora, finetune_model_id, lora_model_id, base_model_id, 
                  lora_scale, config_path, device, dtype):
    batch_size = 1
    if random_seed:
        seed = torch.randint(0, 4294967295, (1,)).item()
    try:
        model_id = finetune_model_id
        if not use_lora:
            pil_image = full_finetuning_inference(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                seed=seed,
                config_path=config_path,
                model_id=model_id,
                device=device,
                dtype=dtype
            )
        else:
            model_id = lora_model_id
            pil_image = lora_inference(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                seed=seed,
                lora_scale=lora_scale,
                config_path=config_path,
                model_id=model_id,
                base_model_id=base_model_id,
                device=device,
                dtype=dtype
            )
        return pil_image, f"Generated image successfully! Seed used: {seed}"
    except Exception as e:
        return None, f"Failed to generate image: {e}"
    