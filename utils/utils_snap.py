# spectral_clustering/utils/utils_snap.py

import requests
import gzip
import shutil
from pathlib import Path
from core.data_preparation import DATA_DIR  # Use your fixed data folder

def fetch_snap(dataset_name: str, data_dir: str = None) -> Path:
    """
    Download a SNAP dataset into the data folder (if not already present) and return the path.

    Supported datasets:
        - ca-GrQc
        - ca-HepTh
        - ca-HepPh

    Parameters
    ----------
    dataset_name : str
        Name of the SNAP dataset.
    data_dir : str or Path, optional
        Directory to store the dataset. Defaults to DATA_DIR.

    Returns
    -------
    Path
        Path to the downloaded and unzipped .txt file.
    """

    SNAP_URLS = {
        "ca-GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        "ca-HepTh": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "ca-HepPh": "https://snap.stanford.edu/data/ca-HepPh.txt.gz",
    }

    if dataset_name not in SNAP_URLS:
        raise ValueError(f"Unknown SNAP dataset: {dataset_name}")

    url = SNAP_URLS[dataset_name]
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    gz_path = data_dir / f"{dataset_name}.txt.gz"
    txt_path = data_dir / f"{dataset_name}.txt"

    # Download if the unzipped file doesn't exist
    if not txt_path.exists():
        print(f"[INFO] Downloading {dataset_name} from SNAP...")
        r = requests.get(url, stream=True)
        r.raise_for_status()  # fail loudly if download fails
        with open(gz_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        # Unzip the .gz file
        with gzip.open(gz_path, 'rb') as f_in:
            with open(txt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()  # remove the .gz after extraction
        print(f"[INFO] Download complete: {txt_path}")

    else:
        print(f"[INFO] Dataset already exists: {txt_path}")

    return txt_path
