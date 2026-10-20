"""Download the Criteo Uplift Prediction Dataset v2.1.

Usage:
    python scripts/download_data.py

The file (about 311 MB gzip, 13.9M rows) lands at
``$UPLIFT_DATA_DIR/raw/criteo-research-uplift-v2.1.csv.gz`` (default
``./data/raw/``; override ``UPLIFT_DATA_DIR`` in a ``.env`` file). The
download is idempotent: an existing complete file is kept.

License notice: the dataset is released by Criteo AI Lab for research
purposes (see https://ailab.criteo.com/criteo-uplift-prediction-dataset/,
CC BY-NC-SA 4.0 on the Hugging Face dataset card). It is not redistributed
in this repository; ``data/raw/`` is gitignored. When publishing results,
cite Diemert et al., "A Large Scale Benchmark for Uplift Modeling"
(AdKDD 2018).

The historical URL http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
(still linked from the Criteo AI Lab page) returned 404 when this project
was built (2026-07), so the official Hugging Face mirror maintained by
Criteo serves as the primary source; the historical URL is kept as a
fallback in case it comes back.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.data import criteo_path  # noqa: E402

URLS = [
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
]

EXPECTED_SIZE = 311_422_618  # bytes, for a sanity check after download
CHUNK = 1 << 20


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "uplift-modeling-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as fh:
        done = 0
        while chunk := response.read(CHUNK):
            fh.write(chunk)
            done += len(chunk)
            if done % (50 * CHUNK) < CHUNK:
                print(f"  {done / 1e6:.0f} MB", flush=True)
    tmp.replace(dest)


def main() -> int:
    dest = criteo_path()
    if dest.is_file() and dest.stat().st_size >= EXPECTED_SIZE:
        print(f"already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("License: Criteo AI Lab research release, not redistributable from this repository.")
    for attempt, url in enumerate([u for u in URLS for _ in range(2)]):
        try:
            print(f"downloading {url}")
            _fetch(url, dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
            time.sleep(2**attempt)
            continue
        size = dest.stat().st_size
        if size < EXPECTED_SIZE * 0.95:
            print(f"  file too small ({size / 1e6:.0f} MB), retrying")
            continue
        print(f"done: {dest.name} ({size / 1e6:.0f} MB)")
        return 0
    print(
        "Could not download the dataset. Check your network connection, or look for the "
        "current link at https://ailab.criteo.com/criteo-uplift-prediction-dataset/ and "
        "place the file at data/raw/criteo-research-uplift-v2.1.csv.gz manually."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
