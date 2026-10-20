"""Download script for Criteo Uplift Prediction Dataset v2.1.

Source: Official Hugging Face mirror maintained by Criteo AI Lab:
https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz

Usage:
    python scripts/download_criteo.py

License Note:
The Criteo dataset is released by Criteo AI Lab for academic and research purposes
(CC BY-NC-SA 4.0). It is strictly not redistributed with code and is gitignored.
Reference publication: Diemert et al., 'A Large Scale Benchmark for Uplift Modeling' (AdKDD 2018).
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from incremental_impact.data.loader import compute_file_sha256

URLS = [
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
]

EXPECTED_SIZE_APPROX = 311_422_618  # bytes (~311 MB)
CHUNK_SIZE = 1 << 20  # 1 MB


def download_criteo_dataset(dest_path: Path | str = "data/raw/criteo-research-uplift-v2.1.csv.gz") -> bool:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size >= EXPECTED_SIZE_APPROX * 0.95:
        print(f"Dataset already downloaded at {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        sha = compute_file_sha256(dest)
        print(f"SHA-256 Checksum: {sha}")
        return True

    print("Initiating download of Criteo Uplift v2.1 (~311 MB compressed)...")
    print("License: Criteo AI Lab Research Release (CC BY-NC-SA 4.0).")

    tmp = dest.with_suffix(".part")
    for attempt, url in enumerate(URLS):
        print(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IncrementalImpact-Research-Benchmark"})
            with urllib.request.urlopen(req, timeout=90) as response, open(tmp, "wb") as f:
                downloaded = 0
                while chunk := response.read(CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (25 * CHUNK_SIZE) < CHUNK_SIZE:
                        print(f"  Downloaded: {downloaded / 1e6:.1f} MB", flush=True)

            tmp.replace(dest)
            size = dest.stat().st_size
            print(f"Successfully downloaded: {dest.name} ({size / 1e6:.1f} MB)")
            sha = compute_file_sha256(dest)
            print(f"Verified SHA-256 Checksum: {sha}")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Download failed from {url}: {e}")
            if tmp.is_file():
                tmp.unlink()
            time.sleep(2**attempt)

    print(
        "\nDownload could not complete automatically. You can manually download the file from:\n"
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz\n"
        f"and place it at: {dest.resolve()}\n"
    )
    return False


if __name__ == "__main__":
    success = download_criteo_dataset()
    sys.exit(0 if success else 1)
