import glob
import os
import zipfile
import shutil

def combine_and_extract_zips(source_dir, combined_file, extract_dir):
    # Step 1: Combine multi-part ZIP files
    zip_parts = sorted(glob.glob(os.path.join(source_dir, "compress_file.zip.*")))
    if not zip_parts:
        print("No compress_file.zip.* parts found in the directory")
        return

    with open(combined_file, "wb") as combined:
        for part in zip_parts:
            with open(part, "rb") as f:
                combined.write(f.read())
    print(f"Combined {len(zip_parts)} parts into {combined_file}")

    # Step 2: Unzip the combined file
    os.makedirs(extract_dir, exist_ok=True)  # Create extraction directory if it doesn't exist
    with zipfile.ZipFile(combined_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted files to {extract_dir}")

    # Step 3: Delete *.zip.* and *.zip files
    files_to_delete = glob.glob(os.path.join(source_dir, "compress_file.zip.*")) + glob.glob(os.path.join(source_dir, "*.zip"))
    if os.path.exists(combined_file):
        files_to_delete.append(combined_file)

    for file in files_to_delete:
        os.remove(file)
        print(f"Deleted {file}")

# Define paths
source_dir = "."
combined_file = "combined_file.zip"
extract_dir = "."

combine_and_extract_zips(source_dir, combined_file, extract_dir)
