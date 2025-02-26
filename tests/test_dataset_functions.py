import pytest
import yaml
from pathlib import Path
import zipfile
from src.dataset_functions import download_data


@pytest.fixture
def dataset_folder():
    with open("src/config.yaml", "r") as c:
        config = yaml.safe_load(c)
    return Path.cwd() / config["dataset"]["folder"]


@pytest.fixture
def species_folders():
    return {"Castanea_sativa": "data/imagery-Castanea_sativa.zip"}


@pytest.fixture
def main_subfolders():
    return {"aerial_imagery": "imagery/"}


@pytest.mark.dataset_functions
def test_download_data(dataset_folder, species_folders, main_subfolders):
    """Test downloading and extracting dataset files."""

    download_data(species_folders, main_subfolders, dataset_folder)

    exp_file1 = Path(dataset_folder / "Castanea_sativa/train/TRAIN-Castanea_sativa-C3-202_23_48.tiff")
    exp_file2 = Path(dataset_folder / "Castanea_sativa/train/TRAIN-Castanea_sativa-C3-260_1_226.tiff")
    
    assert exp_file1.exists(), "File not found"
    assert exp_file2.exists(), "File not found"

    
    