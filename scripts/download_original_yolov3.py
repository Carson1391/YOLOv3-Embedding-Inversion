from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"


def progress(blocks: int, block_size: int, total_size: int) -> None:
    downloaded = blocks * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\r{percent:6.2f}%  {downloaded / (1024**2):8.1f} / {total_size / (1024**2):.1f} MiB")
    else:
        sys.stdout.write(f"\r{downloaded / (1024**2):8.1f} MiB")
    sys.stdout.flush()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Keeping existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}")
    urllib.request.urlretrieve(url, temporary, reporthook=progress)
    print()
    shutil.move(str(temporary), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Joseph Redmon's original YOLOv3 files.")
    parser.add_argument("--directory", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    directory = Path(args.directory)
    download(CFG_URL, directory / "yolov3.cfg", args.force)
    download(WEIGHTS_URL, directory / "yolov3.weights", args.force)


if __name__ == "__main__":
    main()
