"""Collect the files of the Hugging Face Space into a new folder.

    python tools/build_space.py OUT

OUT gets app.py, model.pth, the modules of src/ that predicting needs, and space/README.md and space/requirements.txt.
Nothing is uploaded: the folder is then put into the Space by hand (docs/deploy.md).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = ("calibration.py", "clip_model.py", "decision.py", "images.py", "labels.py", "networks.py", "predict.py", "report.py")
FILES = {"app.py": "app.py", "model.pth": "model.pth", "space/README.md": "README.md",
         "space/requirements.txt": "requirements.txt", **{f"src/{name}": f"src/{name}" for name in SRC}}


def build(out: Path) -> dict[str, str]:
    """Copy the files; returns copied path -> SHA-256."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: give a new folder (nothing was written)")
    copied = {}
    for source, target in FILES.items():
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / source, out / target)
        copied[target] = hashlib.sha256((out / target).read_bytes()).hexdigest()
    return copied


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    for name, digest in build(Path(sys.argv[1])).items():
        print(digest[:12], name)
