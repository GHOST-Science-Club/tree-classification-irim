import zipfile
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_data(species_folders: dict, main_subfolders: dict, dataset_folder: Path):
    """
    Function downloads specified data from HF (PureForest dataset)
    """

    for filename in species_folders:
        print(f"\nProcessing {species_folders[filename]}...")

        # Download file
        file_path = hf_hub_download(
            repo_id="IGNF/PureForest",
            filename=species_folders[filename],
            repo_type="dataset"
        )

        extract_dir = dataset_folder / filename
        extract_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Get list of all files in zip
                image_file_list = zip_ref.namelist()

                # Extract all files, modifying their paths
                for image_file in image_file_list:
                    # Extract file with modified path
                    source = zip_ref.read(image_file)

                    # I assumed we are using aeirla imagery data. However, if needed,
                    # a simple function can be written that chooses either aerial or
                    # LiDAR data
                    target_path = extract_dir / Path(image_file).relative_to(main_subfolders["aerial_imagery"])

                    # Create directories if they don't exist
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(target_path, "wb") as f:
                        f.write(source)

                print(f"Successfully extracted to {extract_dir}")

                extracted_files = Path(extract_dir).iterdir()
                print("Extracted files:")
                for extracted_file in list(extracted_files)[:5]:
                    print(f"- {extracted_file.stem}")
                if len(list(extracted_files)) > 5:
                    print(f"... and {len(extracted_files) - 5} more files")

        except zipfile.BadZipFile:
            print(f"Error: {filename} is not a valid zip file")


def load_dataset(main_dir: dict, species_folders: dict, splits: list=["train", "val", "test"]):

    dataset = {split: {"labels": [], "paths": []} for split in splits} # PLEASE KEEP "paths" KEY!!!!!
    #base_dirs = list(main_dir.glob("*")) 
    base_dirs = [species_folders[filename].\
                 replace("data/imagery-", "").\
                 replace(".zip", "") 
                 for filename in species_folders]

    # Create label mapping
    #label_map = {base_dir.stem: idx for idx, base_dir in enumerate(base_dirs)}
    label_map = {base_dir: idx for idx, base_dir in enumerate(base_dirs)}

    print("Label mapping:", label_map)

    # Load images and create labels
    for base_dir in base_dirs:
        label = label_map[base_dir]

        for split in splits:
            split_dir = main_dir / base_dir / split
            #split_dir = base_dir
            if not split_dir.exists():
                print(f"Warning: {split_dir} does not exist")
                continue

            # Get all TIFF files in the directory
            tiff_files = list(split_dir.glob("*.tiff")) + list(
                split_dir.glob("*.tif")
            )

            print(f"Loading {len(tiff_files)} images from {split_dir}")

            for tiff_path in tiff_files:
                dataset[split]["labels"].append(label)
                dataset[split]["paths"].append(tiff_path)

    # Convert lists to numpy arrays
    for split in splits:
        dataset[split]["labels"] = np.array(dataset[split]["labels"])

    return dataset, label_map


def clip_balanced_dataset(dataset: dict):

    clipped_dataset = {}
    for split in dataset.keys():
        if len(dataset[split]["paths"]) == 0:
            continue

        # Identify minimum class count for this split
        unique_labels, label_counts = np.unique(
            dataset[split]["labels"], return_counts=True
        )
        min_class_count = min(label_counts)

        # Prepare clipped data
        labels_clipped = []
        paths_clipped = []

        for label in unique_labels:
            # Find indices of images with the current label
            indices = np.where(dataset[split]["labels"] == label)[0]

            # Randomly select min_class_count indices from these
            selected_indices = np.random.choice(indices, min_class_count, replace=False)

            # Append selected samples to clipped data lists
            labels_clipped.extend(dataset[split]["labels"][selected_indices])
            paths_clipped.extend([dataset[split]["paths"][i] for i in selected_indices])

        # Convert to numpy arrays
        clipped_dataset[split] = {
            "labels": np.array(labels_clipped),
            "paths": paths_clipped,
        }

    return clipped_dataset

