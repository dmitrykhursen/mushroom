#%%
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from mushroom.config import settings

URLS = [
    ("extra", "https://a3s.fi/mickusti-2007780-pub/mushroom-extra-info.tar.gz"),
    ("sample", "https://a3s.fi/mickusti-2007780-pub/sample.zip"),
    ("test_unlabeled", "https://a3s.fi/mickusti-2007780-pub/test-unlabeled.zip")
]
PROJECT_ROOT = Path(settings.project_root)


def download_data() -> None:
    output_directory = Path(PROJECT_ROOT) / "data"
    
    Path(output_directory).mkdir(exist_ok=True)
    for directory, url in URLS:
        output_path = output_directory / directory
        if output_path.exists():
            print(f"{directory} already exists. Skipping download.")
            continue
        output_path.mkdir(exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()  # Raise error if download failed
        
        if url.endswith(".tar.gz"):
            with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar_ref:
                tar_ref.extractall(output_path)
        elif url.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_path)

        print(f"{directory} done. Files extracted to: {Path(output_path).absolute()}")

if __name__ == "__main__":
    download_data()