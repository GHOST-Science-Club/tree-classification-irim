import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

from typing import Dict
from omegaconf import OmegaConf


def print_extracted_files(extract_dir: Path):
    print(f"Successfully extracted to {extract_dir}")

    extracted_files = Path(extract_dir).iterdir()
    print("Extracted files:")
    for extracted_file in list(extracted_files)[:5]:
        print(f"- {extracted_file.stem}")
    if len(list(extracted_files)) > 5:
        print(f"... and {len(list(extracted_files)) - 5} more files")


def extract_files(file_path: str, extract_dir: Path, main_subfolders: Dict):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        # Get list of all files in zip
        image_file_list = zip_ref.namelist()

        # Extract all files, modifying their paths
        for image_file in image_file_list:
            # Extract file with modified path
            source = zip_ref.read(image_file)

            target_path = extract_dir / \
                Path(image_file).relative_to(main_subfolders["aerial_imagery"])

            # Create directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(target_path, "wb") as f:
                f.write(source)

        print_extracted_files(extract_dir)


def download_data(species_folders: Dict, main_subfolders: Dict, dataset_folder: Path):
    """
    Function downloads specified data from HF (PureForest dataset)
    """

    for filename in species_folders:
        print(f"\nProcessing {species_folders[filename]}...")

        extract_dir = Path(dataset_folder) / filename
        # Check if directory already contains files (skip if so)
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(
                f"\nSkipping {species_folders[filename]}: already downloaded and extracted in {extract_dir}. Remove manually for re-download")
            continue

        # Download file only if not already downloaded (checks hash)
        file_path = hf_hub_download(
            repo_id="IGNF/PureForest",
            filename=species_folders[filename],
            repo_type="dataset",
            local_files_only=False,  # Will check cache and hash
            force_download=False     # Only download if hash changed
        )

        # Create a directory for the extracted files
        extract_dir.mkdir(exist_ok=True, parents=True)

        try:
            extract_files(file_path, extract_dir, main_subfolders)
        except zipfile.BadZipFile:
            print(f"Error: {filename} is not a valid zip file")


config = OmegaConf.load("src/config.yaml")

download_data(config.dataset.species_folders,
              config.dataset.main_subfolders, config.dataset.folder)
