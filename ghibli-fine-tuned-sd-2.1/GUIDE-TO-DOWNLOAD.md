
# Hugging Face Model Download Guide

This guide explains how to use the provided Python script to download all files from a specified Hugging Face model repository using the `huggingface_hub` library.

## Prerequisites

-   `huggingface_hub` library installed. Install it using:
    
    ```bash
    pip install huggingface_hub
    
    ```
    
-   An active internet connection to access the Hugging Face Hub.
    

## Code Explanation

The script downloads all files from a specified Hugging Face model repository to a local directory.

### Steps

1.  **Import Libraries**:
    
    -   `HfApi` and `hf_hub_download` from `huggingface_hub` for interacting with the Hugging Face Hub.
    -   `os` for handling file system operations.
2.  **Set Repository and Directory**:
    
    -   `repo_id`: Specifies the Hugging Face repository ID (e.g., `"danhtran2mind/ghibli-fine-tuned-sd-2.1"`).
    -   `local_dir`: Defines the local directory where files will be saved (e.g., `"."` for the current directory).
3.  **Create Local Directory**:
    
    -   Uses `os.makedirs` to create the local directory if it doesn't exist, with `exist_ok=True` to avoid errors if the directory already exists.
4.  **Initialize Hugging Face API**:
    
    -   Creates an `HfApi` instance to interact with the Hugging Face Hub.
5.  **List Repository Files**:
    
    -   Uses `api.list_repo_files` to retrieve a list of all files in the specified repository.
6.  **Download Files**:
    
    -   Iterates through each file in the repository.
    -   Uses `hf_hub_download` to download each file to the specified `local_dir`.
    -   `local_dir_use_symlinks=False` ensures files are physically copied, not symlinked.
    -   Prints a confirmation message for each downloaded file.

## Usage

1.  Save the script to a file (e.g., `download_model.py`).
    
2.  Modify `repo_id` to the desired Hugging Face repository ID.
    
3.  Adjust `local_dir` to your preferred local directory path.
    
4.  Navigate to the directory where you want to store the model files:
    
    ```bash
    cd ghibli-fine-tuned-sd-2.1
    
    ```
    
5.  Run the script:
    
    ```bash
    python download_model.py
    
    ```
    
6.  Return to the parent directory:
    
    ```bash
    cd ..
    
    ```
    

## Example Output

For each file downloaded, the script will print:

```
Downloaded config.json to .
Downloaded model.safetensors to .
...

```

## Notes

-   Ensure you have sufficient disk space for the model files, as some repositories may contain large files (e.g., model weights).
    
-   If you encounter authentication errors, you may need a Hugging Face token for private repositories. Set it up using:
    
    ```bash
    huggingface-cli login
    
    ```
    
-   The script downloads all files in the repository. To download specific files, modify the `repo_files` loop to filter desired filenames.
    

## Troubleshooting

-   **Module Not Found**: Ensure `huggingface_hub` is installed.
-   **Permission Denied**: Check write permissions for `local_dir`.
-   **Network Issues**: Verify your internet connection or try again later.

For more details, refer to the Hugging Face Hub documentation.
