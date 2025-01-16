from unittest.mock import patch
from src.dataset_functions import download_data


def test_download_data(tmp_path):
    # Mock input
    species_folders = {"Castanea_sativa": "data/imagery-Castanea_sativa.zip"}
    main_subfolders = {"aerial_imagery": "imagery/"}
    dataset_folder = tmp_path / "dataset"

    # Mock hf_hub_download
    mock_zip_path = tmp_path / "mock.zip"
    mock_zip_path.touch()  # Create a mock file
    with patch("huggingface_hub.hf_hub_download", return_value=mock_zip_path):
        # Mock zipfile.ZipFile
        with patch("zipfile.ZipFile") as mock_zip:
            mock_zip.return_value.__enter__.return_value.namelist.return_value = [
                "aerial_imagery/file1.tif",
                "aerial_imagery/file2.tif",
            ]
            mock_zip.return_value.__enter__.return_value.read.side_effect = lambda x: b"mock data"

            # Call the function
            download_data(species_folders, main_subfolders, dataset_folder)

            # Assert the expected files are created
            assert (dataset_folder / "species1/file1.tif").exists()
            assert (dataset_folder / "species1/file2.tif").exists()
