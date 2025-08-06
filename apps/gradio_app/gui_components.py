import gradio as gr
import torch
import os
from .example_handler import get_examples
from .image_generator import generate_image
from .project_info import intro_markdown_1, intro_markdown_2, outro_markdown_1

def load_example_image_full_finetuning(prompt, height, width, num_inference_steps, guidance_scale, seed, image, finetune_model_id):
    return prompt, height, width, num_inference_steps, guidance_scale, seed, image, finetune_model_id, "Loaded example successfully"

def load_example_image_lora(prompt, height, width, num_inference_steps, guidance_scale, seed, image, lora_model_id, base_model_id, lora_scale):
    return prompt, height, width, num_inference_steps, guidance_scale, seed, image, lora_model_id, base_model_id or "stabilityai/stable-diffusion-2-1", lora_scale or 1.2, "Loaded example successfully"

def create_gui(model_configs, device):
    finetune_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'full_finetuning'), None)
    lora_model_id = next((mid for mid, cfg in model_configs.items() if cfg.get('type') == 'lora'), None)
    
    if not finetune_model_id or not lora_model_id:
        raise ValueError("Missing model IDs in config.")

    base_model_id = model_configs[lora_model_id].get('base_model_id', 'stabilityai/stable-diffusion-2-1')
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    config_path = "configs/model_ckpts.yaml"    

    custom_css = open("apps/gradio_app/static/styles.css", "r").read() if os.path.exists("apps/gradio_app/static/styles.css") else ""

    examples_full_finetuning = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-Base-finetuning", use_lora=False)
    examples_lora = get_examples("apps/gradio_app/assets/examples/Ghibli-Stable-Diffusion-2.1-LoRA", use_lora=True)

    with gr.Blocks(css=custom_css, theme="ocean") as demo:
        gr.Markdown("# Ghibli Stable Diffusion Synthesis")
        gr.Markdown(intro_markdown_1)
        gr.Markdown(intro_markdown_2)
        with gr.Tabs():
            with gr.Tab(label="Full Finetuning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Generation Settings")
                        prompt_ft = gr.Textbox(label="Prompt", placeholder="e.g., 'a serene landscape in Ghibli style'", lines=2)
                        with gr.Group():
                            gr.Markdown("#### Image Dimensions")
                            with gr.Row():                                
                                height_ft = gr.Slider(32, 4096, 512, step=8, label="Height")
                                width_ft = gr.Slider(32, 4096, 512, step=8, label="Width")
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
                                height_lora = gr.Slider(32, 4096, 512, step=8, label="Height")
                                width_lora = gr.Slider(32, 4096, 512, step=8, label="Width")
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps_lora = gr.Slider(1, 100, 50, step=1, label="Inference Steps")
                            guidance_scale_lora = gr.Slider(1.0, 20.0, 3.5, step=0.5, label="Guidance Scale")
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
                gr.Examples(examples=examples_lora, inputs=[prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora, output_image_lora, lora_model_path_lora, base_model_path_lora, lora_scale_lora], 
                            outputs=[prompt_lora, height_lora, width_lora, num_inference_steps_lora, guidance_scale_lora, seed_lora, output_image_lora, lora_model_path_lora, base_model_path_lora, lora_scale_lora, output_text_lora], 
                            fn=load_example_image_lora, cache_examples=False, examples_per_page=4)

        gr.Markdown(outro_markdown_1)

        generate_event_ft = generate_btn_ft.click(
            fn=generate_image, 
            inputs=[prompt_ft, height_ft, width_ft, 
                    num_inference_steps_ft, guidance_scale_ft, seed_ft, 
                    random_seed_ft, gr.State(False), finetune_model_path_ft, 
                    gr.State(None), gr.State(None), gr.State(None), 
                    gr.State(config_path), gr.State(device), gr.State(dtype)], 
            outputs=[output_image_ft, output_text_ft]
        )
        generate_event_lora = generate_btn_lora.click(
            fn=generate_image, 
            inputs=[prompt_lora, height_lora, width_lora, 
                    num_inference_steps_lora, guidance_scale_lora, seed_lora, 
                    random_seed_lora, gr.State(True), gr.State(None), 
                    lora_model_path_lora, base_model_path_lora, lora_scale_lora, 
                    gr.State(config_path), gr.State(device), gr.State(dtype)], 
            outputs=[output_image_lora, output_text_lora]
        )

        stop_btn_ft.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_ft])
        stop_btn_lora.click(fn=None, inputs=None, outputs=None, cancels=[generate_event_lora])

        demo.unload(lambda: torch.cuda.empty_cache())

    return demo