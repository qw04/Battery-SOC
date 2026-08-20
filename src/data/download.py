"""Download raw dataset files into data/raw/. Idempotent -- skips files already present."""
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

FILES = {
    "nasa_randomized_battery_usage.zip": (
        "https://phm-datasets.s3.amazonaws.com/NASA/"
        "11.+Randomized+Battery+Usage+Data+Set.zip"
    ),
    "oxford_battery_degradation_dataset_1.mat": (
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/"
        "files/m5ac36a1e2073852e4f1f7dee647909a7"
    ),
    "oxford_example_dc_c1.mat": (
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/"
        "files/me9fc40a60ac98708f1b73f3a836548e9"
    ),
}


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _download_one(url: str, dest: Path, name: str, chunk_size: int = 1 << 20) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                sys.stdout.write(f"\r{name}: {pct}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)")
                sys.stdout.flush()
    tmp.rename(dest)


def download_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            print(f"[skip] {filename} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[download] {filename} <- {url}")
        _download_one(url, dest, filename)
        print(f"\n[done] {filename} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    download_all()
