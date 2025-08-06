import os
import json
from typing import Union, List
from PIL import Image

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