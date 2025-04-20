if __name__ == "__main__":
    from transformers import HfArgumentParser

    @dataclasses.dataclass
    class AppArgs:
        local_model: bool = dataclasses.field(
            default=True, metadata={"help": "Use local model path instead of Hugging Face model."}
        )
        model_name: str = dataclasses.field(
            default="danhtran2mind/ghibli-fine-tuned-sd-2.1",
            metadata={"help": "Model name or path for the fine-tuned Stable Diffusion model."}
        )
        device: str = dataclasses.field(
            default="cuda" if torch.cuda.is_available() else "cpu",
            metadata={"help": "Device to run the model on (e.g., 'cuda', 'cpu')."}
        )
        port: int = dataclasses.field(
            default=7860, metadata={"help": "Port to run the Gradio app on."}
        )
        share: bool = dataclasses.field(
            default=False, metadata={"help": "Set to True for public sharing (Hugging Face Spaces)."}
        )

    parser = HfArgumentParser([AppArgs])
    args_tuple = parser.parse_args_into_dataclasses()
    args = args_tuple[0]

    # Set model_name based on local_model flag
    if args.local_model:
        args.model_name = "ghibli-fine-tuned-sd-2.1"

    demo = create_demo(args.model_name, args.device)
    demo.launch(server_port=args.port, share=args.share)
