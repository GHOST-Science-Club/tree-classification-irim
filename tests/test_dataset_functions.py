import pytest
import yaml
from pathlib import Path
import zipfile
from src.dataset_functions import download_data, load_dataset, extract_files


@pytest.fixture
def dataset_folder():
    with open("src/config.yaml", "r") as c:
        config = yaml.safe_load(c)
    return Path.cwd() / config["dataset"]["folder"]


@pytest.fixture
def species_folders():
    return {"Castanea_sativa": "data/imagery-Castanea_sativa.zip", "Pinus_nigra": "data/imagery-Pinus_nigra.zip"}


@pytest.fixture
def main_subfolders():
    return {"aerial_imagery": "imagery/"}


@pytest.mark.dataset_functions
def test_download_data(dataset_folder, species_folders, main_subfolders):
    download_data(species_folders, main_subfolders, dataset_folder)

    exp_file1 = Path(dataset_folder / "Castanea_sativa/train/TRAIN-Castanea_sativa-C3-202_23_48.tiff")
    exp_file2 = Path(dataset_folder / "Castanea_sativa/train/TRAIN-Castanea_sativa-C3-260_1_226.tiff")
    
    assert exp_file1.exists(), f"File {exp_file1} not found"
    assert exp_file2.exists(), f"File {exp_file2} not found"
    
    
@pytest.mark.dataset_functions
def test_extract_file_bad_zip_error(tmp_path, dataset_folder, main_subfolders):
    corrupt_zip_path = tmp_path / "corrupt.zip"
    with open(corrupt_zip_path, "wb") as f:
        f.write(b"not a real zip file")  # Writing invalid data
    
    with pytest.raises(zipfile.BadZipFile):
        extract_files(corrupt_zip_path, dataset_folder, main_subfolders)

    
@pytest.mark.dataset_functions
def test_load_dataset(dataset_folder, species_folders):
    exp_label_map = {species: idx for idx, species in enumerate(species_folders)}
    
    exp_path1 = Path(r"C:\Users\Mateusz\PycharmProjects\GHOST_IRIM\tree-classification-irim\src\data\Castanea_sativa\test\TEST-Castanea_sativa-C3-17_1_42.tiff")
    exp_path2 = Path(r"C:\Users\Mateusz\PycharmProjects\GHOST_IRIM\tree-classification-irim\src\data\Pinus_nigra\train\TRAIN-Pinus_nigra-C7-100_1_280.tiff")
    
    dataset, label_map = load_dataset(dataset_folder, species_folders)
    
    assert exp_path1 in dataset["test"]["paths"], f"Path {exp_path1} not found"
    assert exp_path2 in dataset["train"]["paths"], f"Path {exp_path2} not found"
    assert exp_label_map == label_map, f"Incorrect label map. Expected: {exp_label_map}"