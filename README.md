
# Ghibli Fine-Tuned Stable Diffusion 2.1 [![GitHub Stars](https://img.shields.io/github/stars/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo?label=⭐&style=social)](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/stargazers)
[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-orange.svg?logo=huggingface)](https://huggingface.co/docs/hub)
[![accelerate](https://img.shields.io/badge/accelerate-yellow.svg?logo=pytorch)](https://huggingface.co/docs/accelerate)
[![bitsandbytes](https://img.shields.io/badge/bitsandbytes-%2300A1D6.svg)](https://github.com/TimDettmers/bitsandbytes)
[![torch](https://img.shields.io/badge/torch-yellow.svg?logo=pytorch)](https://pytorch.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%2300A1D6.svg)](https://pypi.org/project/pillow/)
[![numpy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy)](https://numpy.org/)
[![transformers](https://img.shields.io/badge/transformers-orange.svg?logo=huggingface)](https://huggingface.co/docs/transformers)
[![torchvision](https://img.shields.io/badge/torchvision-yellow.svg?logo=pytorch)](https://pytorch.org/vision/stable/index.html)
[![diffusers](https://img.shields.io/badge/diffusers-orange.svg?logo=huggingface)](https://huggingface.co/docs/diffusers)
[![gradio](https://img.shields.io/badge/gradio-yellow.svg?logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
-   [Introduction](./README.md#introduction)
-   [Key Features](./README.md#key-features)
-   [Training Notebook](./README.md#training-notebook)
-   [Dataset](./README.md#dataset)
-   [Base Model](./README.md#base-model)
-   [Installation](./README.md#installation)
-   [Usage](./README.md#usage)
    -   [Running the Training Notebook](./README.md#running-the-training-notebook)
    -   [Running the Gradio Demo](./README.md#running-the-gradio-demo)
-   [Training Hyperparameters](./README.md#training-hyperparameters)
-   [Metrics](./README.md#metrics)
-   [Environment](./README.md#environment)
-   [Demonstration](./README.md#demonstration)
-   [Contact](./README.md#contact)
-   [License](./README.md#license)
-   [Acknowledgements](./README.md#acknowledgements)
-   [Contributing](./README.md#contributing)

## Introduction

The **Ghibli Fine-Tuned Stable Diffusion 2.1** project is a cutting-edge endeavor that harnesses the power of deep learning to generate images in the enchanting and iconic art style of Studio Ghibli. By fine-tuning the Stable Diffusion 2.1 model, this project enables the creation of visually stunning images that capture the vibrant colors, intricate details, and whimsical charm of Ghibli films. The repository includes a meticulously crafted Jupyter notebook for training, an interactive Gradio demo for real-time image generation, and comprehensive instructions for setup and usage. Designed for data scientists, developers, and Ghibli enthusiasts, this project bridges technology and artistry with unparalleled precision.

## Key Features

-   **Fine-Tuned Model**: A Stable Diffusion 2.1 model optimized for Studio Ghibli’s art style, delivering authentic and high-quality image outputs.
-   **Comprehensive Training Notebook**: A detailed Jupyter notebook that guides users through the fine-tuning process, compatible with multiple platforms.
-   **Interactive Gradio Demo**: A user-friendly interface for generating Ghibli-style images, showcasing the model’s capabilities in real-time.
-   **Secure Data Handling**: Encrypted dataset and model files using git-crypt, ensuring data integrity and controlled access.
-   **Cross-Platform Compatibility**: Support for Google Colab, Amazon SageMaker, Deepnote, and JupyterLab, providing flexibility for all users.

## Training Notebook

The cornerstone of this project is the Jupyter notebook located at `notebooks/fine_tuned_sd_2_1_base-notebook.ipynb`. This notebook provides a step-by-step guide to fine-tuning the Stable Diffusion 2.1 model using the Ghibli dataset, complete with code, explanations, and best practices. It is designed to be accessible to both beginners and experienced practitioners, offering flexibility to replicate the training process or experiment with custom modifications. The notebook is compatible with the following platforms:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/blob/main/notebooks/fine_tuned_sd_2_1_base-notebook.ipynb)
[![Open in SageMaker](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/blob/main/notebooks/fine_tuned_sd_2_1_base-notebook.ipynb)
[![Open in Deepnote](https://deepnote.com/buttons/launch-in-deepnote-small.svg)](https://deepnote.com/launch?url=https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/blob/main/notebooks/fine_tuned_sd_2_1_base-notebook.ipynb)
[![JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange?logo=Jupyter)](https://mybinder.org/v2/gh/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/main?filepath=notebooks/fine_tuned_sd_2_1_base-notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/blob/main/notebooks/fine_tuned_sd_2_1_base-notebook.ipynb)

To get started, open the notebook in your preferred platform and follow the instructions to set up the environment and execute the training process.

## Dataset

The project utilizes the [Ghibli Dataset](https://huggingface.co/datasets/uwunish/ghibli-dataset) from Hugging Face, a carefully curated collection of images from Studio Ghibli films. This dataset encapsulates the unique visual style of Ghibli, featuring vibrant colors, intricate landscapes, and whimsical characters, making it ideal for fine-tuning the model.

## Base Model

The fine-tuning process is built upon the [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) model by Stability AI. This robust text-to-image model provides a solid foundation, enabling high-fidelity image generation with targeted fine-tuning to achieve the Ghibli aesthetic.

## Installation

To set up the project, ensure you have Python 3.11 or later installed. The following steps guide you through cloning the repository, installing dependencies, and preparing encrypted data.

### Step 1: Clone the Repository

Clone the repository from [GitHub](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo):

```bash
git clone https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo.git
cd ghibli-fine-tuned-sd-2.1-repo

```

### Step 2: Install Dependencies

Install the required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

### Step 3: Decrypt Encrypted Folders (if necessary)

The `dataset` and `diffusers` folders are encrypted using git-crypt for security. To decrypt them, obtain the decryption key by contacting the maintainer via the [Issues tab](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/issues). Then, run:

```bash
git-crypt unlock /path/to/my-repo.asc

```

Replace `/path/to/my-repo.asc` with the path to your decryption key file. Ensure git-crypt is installed and configured for the repository.

## Usage

The project supports two primary use cases: training the model using the Jupyter notebook and generating images with the Gradio demo.

### Running the Training Notebook

The training notebook (`notebooks/fine_tuned_sd_2_1_base-notebook.ipynb`) is the core component for fine-tuning the model. To run it:

1.  Open the notebook in your preferred platform (Colab, SageMaker, Deepnote, or JupyterLab).
2.  Follow the setup instructions within the notebook to configure the environment.
3.  Execute the cells sequentially to train the model or experiment with custom hyperparameters.

The notebook includes detailed comments and explanations, making it easy to understand and modify the training process.

### Running the Gradio Demo

To generate Ghibli-style images using the Gradio demo, follow these steps:

1.  **Navigate to the Repository Root**:
    
    ```bash
    cd ghibli-fine-tuned-sd-2.1-repo
    
    ```
    
2.  **Download the Fine-Tuned Model**:
    
    Download the model weights to the `ghibli-fine-tuned-sd-2.1` folder:
    
    ```bash
    cd ghibli-fine-tuned-sd-2.1
    python download_model.py
    cd ..
    
    ```
    
    The `download_model.py` script retrieves the model from the Hugging Face repository.
    
3.  **Extract the Dataset**:
    
    Download and extract the Ghibli dataset to the `dataset` folder:
    
    ```bash
    cd dataset
    pip install datasets
    python extract_files.py
    cd ..
    
    ```
        
4.  **Extract the Diffusers Folder**:
    
    Extract the model weights or related files in the `diffusers` folder:
    
    ```bash
    cd diffusers
    python extract_files.py
    cd ..
    
    ```
    
    The `extract_files.py` script handles the extraction process.
    
5.  **Run the Gradio App**:
    
    Launch the Gradio demo to interact with the model:
    
    ```bash
    python app.py --local_model True
    
    ```
    
    The demo will be available at `localhost:7860`. Use `--local_model True` for the local model or `False` to download from Hugging Face.
    

## Training Hyperparameters

The fine-tuning process was optimized with the following hyperparameters:

| Hyperparameter | Value |
| --- | --- |
| `learning_rate` | 1e-05 |
| `num_train_epochs` | 40 |
| `train_batch_size` | 2 |
| `gradient_accumulation_steps` | 2 |
| `mixed_precision` | "fp16" |
| `resolution` | 512 |
| `max_grad_norm` | 1 |
| `lr_scheduler` | "constant" |
| `lr_warmup_steps` | 0 |
| `checkpoints_total_limit` | 1 |
| `use_ema` | True |
| `use_8bit_adam` | True |
| `center_crop` | True |
| `random_flip` | True |
| `gradient_checkpointing` | True |

These parameters were carefully selected to balance training efficiency and model performance, leveraging techniques like mixed precision and gradient checkpointing.

## Metrics

The fine-tuning process achieved a final loss of **0.0345**, indicating excellent convergence and high fidelity to the Ghibli art style.

## Environment

The project was developed and tested in the following environment:

- **Python Version**: 3.11.11
- **Dependencies**:

| Library | Version |
| --- | --- |
| huggingface-hub | 0.30.2 |
| accelerate | 1.3.0 |
| bitsandbytes | 0.45.5 |
| torch | 2.5.1 |
| Pillow | 11.1.0 |
| numpy | 1.26.4 |
| transformers | 4.51.1 |
| torchvision | 0.20.1 |
| diffusers | 0.33.1 |
| gradio | Latest |

Ensure your environment matches these specifications to avoid compatibility issues.

## Demonstration

Explore the model’s capabilities through the interactive demo hosted at [Ghibli Fine-Tuned SD 2.1](https://huggingface.co/spaces/danhtran2mind/ghibli-fine-tuned-sd-2.1). The demo allows users to generate Ghibli-style images effortlessly.

**Preview Image**:

![Demo Image](./visualization/demo_image.png?raw=true)

## Contact

For questions, issues, or to request the git-crypt decryption key, please contact the maintainer via the [Issues tab](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo/issues) on GitHub.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), allowing for flexible use and modification while ensuring proper attribution.

## Acknowledgements

The success of this project is built upon the contributions of several key resources and communities:

-   [Hugging Face](https://huggingface.co/) for providing the dataset and model hubs, enabling seamless access to high-quality resources.
-   [Stability AI](https://stability.ai/) for developing the Stable Diffusion model, a cornerstone of this project.
-   The open-source community for their continuous support and contributions to the tools and libraries used.

## Contributing

Contributions to this project are warmly welcomed! To contribute, please follow these steps:

1.  Fork the repository from [GitHub](https://github.com/danhtran2mind/ghibli-fine-tuned-sd-2.1-repo).
2.  Create a new branch for your feature or bug fix.
3.  Commit your changes with clear and descriptive commit messages.
4.  Push your branch and submit a pull request.

For detailed guidelines, refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file. Your contributions can help enhance the project and bring the Ghibli art style to a wider audience.
