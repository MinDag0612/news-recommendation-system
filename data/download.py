from pathlib import Path
from zipfile import ZipFile
import shutil

from huggingface_hub import hf_hub_download


DATA_DIR = Path(__file__).resolve().parent / "primary"
REPO_ID = "yjw1029/MIND"

FILES = {
    "MINDlarge_train.zip": "train_set",
    "MINDlarge_dev.zip": "dev_set",
    "MINDlarge_test.zip": "test_set",
}

REQUIRED_FILES = {
    "behaviors.tsv",
    "news.tsv",
}


def download_and_extract(filename: str, split_dir: str):
    output_dir = DATA_DIR / split_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {filename}...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
    )

    print(f"Extracting required files from {filename}...")

    with ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            file_name = Path(member.filename).name

            if file_name not in REQUIRED_FILES:
                continue

            target_path = output_dir / file_name

            with zip_file.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

            print(f"  Extracted: {file_name}")

    print(f"Finished: {split_dir}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, split_dir in FILES.items():
        download_and_extract(filename, split_dir)

    print("\nMIND-large download completed.")
    print(f"Dataset location: {DATA_DIR}")


if __name__ == "__main__":
    main()