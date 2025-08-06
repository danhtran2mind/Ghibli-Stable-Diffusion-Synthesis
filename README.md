## Usage Guide
### Setup Instructions
#### Step 1: Clone the Repository
Clone the project repository and navigate to the project directory:
```bash
git clone https://github.com/danhtran2mind/Vi-F5-TTS.git
cd Vi-F5-TTS
```

#### Step 2: Install Dependencies
Install the required Python packages:
```bash
pip install -e . 
```
Or Install Dependencies using `requirements.txt`
```bash
pip install -r requirements/requirements.txt
```

#### Step 3: Configure the Environment
Run the following scripts to set up the project:
- **Install Third-Party Dependencies**  
  ```bash
  python scripts/setup_third_party.py
  ```
- **Download Model Checkpoints**
    - Use `SWivid/F5-TTS`:
    ```bash
    python scripts/download_ckpts.py \
        --repo_id "SWivid/F5-TTS" --local_dir "./ckpts" \
        --folder_name "F5TTS_v1_Base_no_zero_init"
    ```
    - Use `danhtran2mind/Vi-F5-TTS`:
    ```bash
    python scripts/download_ckpts.py \
        --repo_id "danhtran2mind/Vi-F5-TTS" \
        --local_dir "./ckpts" --pruning_model
    ```

- **Prepare Dataset (Optional, for Training)**  
  ```bash
  python scripts/process_dataset.py
  ```

### Training
The Training Notebooks, available at [Training Notebook](#training_notebook), offer a comprehensive guide to both the Full Fine-tuning and LoRA training methods.

<!-- #### Config
Configuration of the `accelerate`
```bash
accelerate config default
```
#### Training Bash
To train the model:
```bash
accelerate launch ./src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name vin100h-preprocessed-v2 \
    --finetune \
    --tokenizer pinyin \
    --learning_rate 1e-05 \
    --batch_size_type frame \
    --batch_size_per_gpu 3200 \
    --max_samples 64 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1 \
    --epochs 80 \
    --num_warmup_updates 2761 \
    --save_per_updates 4000 \
    --keep_last_n_checkpoints 1 \
    --last_per_updates 4000 \
    --log_samples \
    --pretrain "<your_pretrain_model>"# such as "./ckpts/F5TTS_v1_Base_no_zero_init/model_1250000.safetensors"
```
#### Training Arguments
Refer to the [Training Documents](docs/training/training_doc.md) for detailed arguments used in fine-tuning the model. ⚙️ -->

### Inference
#### Quick Inference Bash
- To generate iamge using the `Full Fine-tuning` model:
```bash
python src/ghibli_stable_diffusion_synthesis/infer.py \
    --method full_finetuning \
    --prompt "donald trump in ghibli style" \
    --height 512 --width 512 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --batch_size 1 --seed 42 \
    --output_path "tests/test_data/ghibli_style_output_full_finetuning.png"
```
- To run inference with `LoRA`:
```bash
python src/ghibli_stable_diffusion_synthesis/infer.py \
    --method lora \
    --prompt "donald trump in ghibli style" \
    --height 512 --width 512 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --batch_size 1 --seed 42 \
    --lora_scale 1.2 \
    --output_path "tests/test_data/ghibli_style_output_lora.png"
```
#### Inference Arguments
Refer to the [Inference Documents](docs/inference/inference_doc.md) for detailed arguments used in Inference. ⚙️
### Inference Example