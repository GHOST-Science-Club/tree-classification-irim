import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


def print_extracted_files(extract_dir: Path):
    print(f"Successfully extracted to {extract_dir}")

    extracted_files = Path(extract_dir).iterdir()
    print("Extracted files:")
    for extracted_file in list(extracted_files)[:5]:
        print(f"- {extracted_file.stem}")
    if len(list(extracted_files)) > 5:
        print(f"... and {len(list(extracted_files)) - 5} more files")


def extract_files(file_path: Path, extract_dir: Path, main_subfolders: str):
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

        print_extracted_files(extract_dir)


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
        extract_dir.mkdir(exist_ok=True, parents=True)

        try:
            extract_files(file_path, extract_dir, main_subfolders)
        except zipfile.BadZipFile:
            print(f"Error: {filename} is not a valid zip file")


def load_dataset(main_dir: dict, species_folders: dict, splits=None):
    if splits is None:
        splits = ["train", "val", "test"]
    dataset = {split: {"labels": [], "paths": []} for split in splits}  # PLEASE KEEP "paths" KEY!!!!!

    merged_labels = {
        "Quercus_petraea": "Deciduous_oak",
        "Quercus_pubescens": "Deciduous_oak",
        "Quercus_robur": "Deciduous_oak",
        "Quercus_rubra": "Deciduous_oak",
        "Quercus_ilex": "Evergreen_oak",
        "Fagus_sylvatica": "Beech",
        "Castanea_sativa": "Chestnut",
        "Robinia_pseudoacacia": "Black_locust",
        "Pinus_pinaster": "Maritime_pine",
        "Pinus_sylvestris": "Scotch_pine",
        "Pinus_nigra_laricio": "Black_pine",
        "Pinus_nigra": "Black_pine",
        "Pinus_halepensis": "Aleppo pine",
        "Abies_alba": "Fir",
        "Abies_nordmanniana": "Fir",
        "Picea_abies": "Spruce",
        "Larix_decidua": "Larch",
        "Pseudotsuga_menziesii": "Douglas"
    }

    # Filtering merged_labels to present classes in config.yaml
    available_labels = {key: merged_labels[key] 
                        for key in species_folders 
                        if key in merged_labels}
    
    unique_labels = sorted(set(available_labels.values()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label mapping:", label_map)

    # base_dirs = list(main_dir.glob("*"))
    base_dirs = [species_folders[filename].
                 replace("data/imagery-", "").
                 replace(".zip", "")
                 for filename in species_folders]
    
    # Load images and create labels
    for base_dir in base_dirs:
        original_label = base_dir
        merged_label = available_labels.get(original_label, None)
        if merged_label is None:
            continue

        label = label_map[merged_label]

        for split in splits:
            split_dir = main_dir / base_dir / split
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
        dataset[split]["labels"] = list(np.array(dataset[split]["labels"]))

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



if __name__ == '__main__':
    import yaml

    with open("src/config.yaml", "r") as c:
        config = yaml.safe_load(c)

    dataset_folder = Path.cwd() / config["dataset"]["folder"]
    dataset_folder.mkdir(exist_ok=True)

    species_folders = config["dataset"]["species_folders"]
    main_subfolders = config["dataset"]["main_subfolders"]


    download_data(species_folders, main_subfolders, dataset_folder)
    dataset, label_map = load_dataset(dataset_folder, species_folders)

    print(dataset.keys())
    print(label_map)