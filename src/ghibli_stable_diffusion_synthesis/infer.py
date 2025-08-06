import argparse
from inference.full_finetuning import inference_process as full_finetuning_inference
from inference.lora import inference_process as lora_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion inference with full fine-tuning or LoRA")
    parser.add_argument("--method", type=str, choices=["full_finetuning", "lora"], required=True, 
                        help="Inference method: 'full_finetuning' or 'lora'")
    parser.add_argument("--prompt", type=str, default="donald trump in ghibli style", 
                        help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=512, 
                        help="Height of the output image (must be divisible by 8)")
    parser.add_argument("--width", type=int, default=512, 
                        help="Width of the output image (must be divisible by 8)")
    parser.add_argument("--num_inference_steps", type=int, default=50, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, 
                        help="Guidance scale for classifier-free guidance")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--lora_scale", type=float, default=1.2, 
                        help="Scaling factor for LoRA weights")
    parser.add_argument("--config_path", type=str, default="configs/model_ckpts.yaml", 
                        help="Path to the model configuration YAML file")
    parser.add_argument("--output_path", type=str, default="test_data/ghibli_style_{method}_output.png", 
                        help="Path to save the output image")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.method == "full_finetuning":
        pil_image = full_finetuning_inference(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            lora_scale=args.lora_scale,
            config_path=args.config_path,
            model_id="danhtran2mind/Ghibli-Stable-Diffusion-2.1-Base-finetuning"
        )
    elif args.method == "lora":
        pil_image = lora_inference(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            lora_scale=args.lora_scale,
            config_path=args.config_path,
            model_id="danhtran2mind/Ghibli-Stable-Diffusion-2.1-LoRA"
        )
    pil_image.save(args.output_path.format(method=args.method))