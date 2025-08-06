import argparse
import torch
from gradio_app.gui_components import create_gui
from gradio_app.config_loader import load_model_configs

def main():
    parser = argparse.ArgumentParser(description="Ghibli-Style Image Generator")
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