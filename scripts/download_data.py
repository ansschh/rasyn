"""Download USPTO datasets and RSGPT pretrained weights.

Usage:
    python scripts/download_data.py --datasets uspto50k uspto_full
    python scripts/download_data.py --rsgpt-weights
    python scripts/download_data.py --all
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Dataset download URLs
DATASET_URLS = {
    "uspto50k": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv",
    "uspto_full": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_FULL.csv",
    "uspto_mit": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv",
}

# RSGPT Zenodo URLs
RSGPT_ZENODO_RECORD = "15304009"
RSGPT_FILES = {
    "finetune_50k.pth": f"https://zenodo.org/records/{RSGPT_ZENODO_RECORD}/files/finetune_50k.pth",
    "finetune_full.pth": f"https://zenodo.org/records/{RSGPT_ZENODO_RECORD}/files/finetune_full.pth",
    "USPTO_data.7z": f"https://zenodo.org/records/{RSGPT_ZENODO_RECORD}/files/USPTO_data.7z",
}

RSGPT_REPO = "https://github.com/jogjogee/RSGPT.git"


def download_file(url: str, dest: Path) -> bool:
    """Download a file using wget or curl."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"Already exists: {dest}")
        return True

    logger.info(f"Downloading {url} -> {dest}")

    # Try wget first, then curl
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Python fallback
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


@click.command()
@click.option("--datasets", "-d", multiple=True, type=click.Choice(["uspto50k", "uspto_full", "uspto_mit"]))
@click.option("--rsgpt-weights", is_flag=True, help="Download RSGPT pretrained weights from Zenodo")
@click.option("--rsgpt-repo", is_flag=True, help="Clone RSGPT GitHub repository")
@click.option("--all", "download_all", is_flag=True, help="Download everything")
@click.option("--data-dir", type=click.Path(), default=str(DATA_DIR))
def main(datasets, rsgpt_weights, rsgpt_repo, download_all, data_dir):
    """Download datasets and model weights for Rasyn."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path(data_dir)

    if download_all:
        datasets = ("uspto50k", "uspto_full")
        rsgpt_weights = True
        rsgpt_repo = True

    # Download datasets
    for name in datasets:
        url = DATASET_URLS[name]
        dest = data_dir / f"{name}.csv"
        download_file(url, dest)

    # Download RSGPT weights
    if rsgpt_weights:
        weights_dir = data_dir.parent.parent / "weights" / "rsgpt"
        weights_dir.mkdir(parents=True, exist_ok=True)

        for filename, url in RSGPT_FILES.items():
            dest = weights_dir / filename
            download_file(url, dest)

        logger.info(f"RSGPT weights downloaded to {weights_dir}")

    # Clone RSGPT repo
    if rsgpt_repo:
        repo_dir = data_dir.parent.parent / "external" / "RSGPT"
        if repo_dir.exists():
            logger.info(f"RSGPT repo already exists: {repo_dir}")
        else:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning RSGPT repo to {repo_dir}")
            subprocess.run(["git", "clone", RSGPT_REPO, str(repo_dir)], check=True)

    logger.info("Done!")


if __name__ == "__main__":
    main()
