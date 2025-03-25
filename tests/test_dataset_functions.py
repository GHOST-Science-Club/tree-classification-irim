import huggingface_hub.errors
import pytest
import yaml
from pathlib import Path
import zipfile
import huggingface_hub
from src.dataset_functions import download_data, load_dataset, extract_files


@pytest.fixture
def dataset_folder():
    with open("src/config.yaml", "r") as c:
        config = yaml.safe_load(c)
    return Path.cwd() / config["dataset"]["folder"]


@pytest.fixture
def species_folders():
    return {
        "Pinus_nigra": "data/imagery-Pinus_nigra.zip",
        "Castanea_sativa": "data/imagery-Castanea_sativa.zip"
    }


@pytest.fixture
def main_subfolders():
    return {"aerial_imagery": "imagery/"}


@pytest.fixture
def sample_images(dataset_folder):
    exp_path1 = dataset_folder / Path(
        "Castanea_sativa/test/TEST-Castanea_sativa-C3-17_1_42.tiff"
        )

    exp_path2 = dataset_folder / Path(
        "Pinus_nigra/train/TRAIN-Pinus_nigra-C7-100_1_280.tiff"
        )

    return exp_path1, exp_path2


@pytest.mark.dataset_functions
def test_download_data(dataset_folder,
                       species_folders,
                       main_subfolders,
                       sample_images
                       ):
    download_data(species_folders, main_subfolders, dataset_folder)

    assert sample_images[0].exists(), f"File {sample_images[0]} not found"
    assert sample_images[1].exists(), f"File {sample_images[1]} not found"


@pytest.mark.dataset_functions
def test_extract_file_bad_zip_error(tmp_path, dataset_folder, main_subfolders):
    corrupt_zip_path = tmp_path / "corrupt.zip"
    with open(corrupt_zip_path, "wb") as f:
        f.write(b"not a real zip file")  # Writing invalid data

    with pytest.raises(zipfile.BadZipFile):
        extract_files(corrupt_zip_path, dataset_folder, main_subfolders)


@pytest.mark.dataset_functions
def test_hf_download_errors(dataset_folder, main_subfolders):
    invalid_entry = {"invalid": "entry"}
    invalid_folder = {"invalid": 12345}

    with pytest.raises(huggingface_hub.errors.EntryNotFoundError):
        download_data(invalid_entry, main_subfolders, dataset_folder)

    with pytest.raises(AttributeError):
        download_data(invalid_folder, main_subfolders, dataset_folder)


@pytest.mark.dataset_functions
def test_load_dataset(dataset_folder, species_folders, sample_images):
    merged_labels = {
        "Castanea_sativa": "Chestnut",
        "Pinus_nigra": "Black_pine",
    }

    exp_label_map = {
        merged_labels[species]: idx for idx,
        species in enumerate(species_folders)
    }

    dataset, label_map = load_dataset(dataset_folder, species_folders)

    error_msg = {
        "img-1": f"Path {sample_images[0]} not found",
        "img-2": f"Path {sample_images[1]} not found",
        "wrong-label": f"Incorrect label map. Expected: {exp_label_map}"
    }

    assert sample_images[0] in dataset["test"]["paths"], error_msg["img-1"]
    assert sample_images[1] in dataset["train"]["paths"], error_msg["img-2"]
    assert exp_label_map == label_map, error_msg["wrong-label"]
