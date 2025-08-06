import argparse
import subprocess
import os
import torch
from gradio_app.gui_components import create_gui
from gradio_app.config_loader import load_model_configs

def run_setup_script():
    setup_script = os.path.join(os.path.dirname(__file__),
                                "gradio_app", "setup_scripts.py")
    try:
        result = subprocess.run(["python", setup_script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with error: {e.stderr}")
        return f"Setup script failed: {e.stderr}"

def main():
    parser = argparse.ArgumentParser(description="Ghibli Stable Diffusion Synthesisr")
    parser.add_argument("--config_path", type=str, default="configs/model_ckpts.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    model_configs = load_model_configs(args.config_path)
    demo = create_gui(model_configs, args.device)
    demo.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()