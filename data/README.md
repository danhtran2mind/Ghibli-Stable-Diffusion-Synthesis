# Dataset Extraction

This folder contains a Python script (`extract_files.py`) to extract multi-part ZIP files (`compress_file.zip.*`) into the `dataset` folder. The script combines split ZIP archives, extracts their contents, and cleans up temporary files.

## Extraction Instructions

1.  Navigate to the `dataset` directory:
    
    ```bash
    cd dataset
    
    ```
    
2.  Run the extraction script:
    
    ```bash
    python extract_files.py
    
    ```
    
3.  Return to the parent directory:
    
    ```bash
    cd ..
    
    ```
    

## Script Functionality

The `extract_files.py` script performs the following tasks:

1.  **Combines Multi-Part ZIP Files**:
    
    -   Locates all `compress_file.zip.*` parts in the current directory (e.g., `compress_file.zip.001`, `compress_file.zip.002`).
    -   Combines these parts into a single ZIP file (`combined_file.zip`) by reading and appending their binary content in order.
2.  **Extracts the Combined ZIP**:
    
    -   Creates an extraction directory (if it doesn't exist) in the current folder.
    -   Unzips the combined ZIP file, extracting all contents to the specified directory.
3.  **Cleans Up Temporary Files**:
    
    -   Deletes all `compress_file.zip.*` parts, the combined `combined_file.zip`, and any other `.zip` files in the directory to free up space.

## Notes

-   Ensure all `compress_file.zip.*` parts are present in the `dataset` folder before running the script.
-   The script assumes the ZIP parts are valid and can be combined into a functional ZIP archive.
-   If no `compress_file.zip.*` parts are found, the script will notify you and exit.