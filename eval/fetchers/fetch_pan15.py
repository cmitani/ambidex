"""Fetch PAN 2015 author verification corpora from Zenodo (record 3737563).

Data stays outside git (data/ is gitignored). Result:
    data/raw/pan15/<language>/<problem>/known*.txt + unknown.txt, truth.txt at root
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

OUTER_URL = (
    "https://zenodo.org/api/records/3737563/files/"
    "pan15-authorship-verification-test-and-training.zip/content"
)
RECORD_URL = "https://zenodo.org/records/3737563"
TRAINING_ZIP = "pan15-authorship-verification-training-dataset-2015-04-19.zip"

INNER_ZIPS = {
    "en": "pan15-authorship-verification-training-dataset-english-2015-04-19.zip",
    "es": "pan15-authorship-verification-training-dataset-spanish-2015-04-19.zip",
    "nl": "pan15-authorship-verification-training-dataset-dutch-2015-04-19.zip",
    "gr": "pan15-authorship-verification-training-dataset-greek-2015-04-19.zip",
}
DIR_NAMES = {"en": "english", "es": "spanish", "nl": "dutch", "gr": "greek"}


def fetch(langs: list[str], out_root: Path) -> None:
    cache = out_root / "pan15.zip"
    if not cache.exists():
        print(f"downloading {OUTER_URL} ...")
        out_root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OUTER_URL, cache)
    outer_bytes = cache.read_bytes()
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        member = [n for n in outer.namelist() if n.endswith(TRAINING_ZIP)]
        if not member:
            raise FileNotFoundError(f"{TRAINING_ZIP} not found in the outer zip")
        training_bytes = outer.read(member[0])
    for lang in langs:
        inner_name = INNER_ZIPS[lang]
        with zipfile.ZipFile(io.BytesIO(training_bytes)) as training:
            member = [n for n in training.namelist() if n.endswith(inner_name)]
            if not member:
                raise FileNotFoundError(f"{inner_name} not found in the training zip")
            inner_bytes = training.read(member[0])
        dest = out_root / "pan15" / DIR_NAMES[lang]
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            inner.extractall(dest)
        children = [p for p in dest.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            inner_root = children[0]
            for p in inner_root.iterdir():
                p.rename(dest / p.name)
            inner_root.rmdir()
        problems = [p for p in dest.iterdir() if p.is_dir() and p.name[:2].isupper() and p.name[2:].isdigit()]
        print(f"[{lang}] extracted {len(problems)} problems -> {dest}")
    print(f"\nsource record: {RECORD_URL} (check license/citation there)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en"], choices=sorted(INNER_ZIPS))
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    fetch(args.langs, Path(args.out))
