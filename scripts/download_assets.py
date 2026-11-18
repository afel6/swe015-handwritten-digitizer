"""Download the public release assets (model weights + cached corpus) and verify every pinned sha256.

Standard library only. Run from the repository root:  python scripts/download_assets.py
Files already present with the right hashes are not downloaded again. Any failure prints one line and exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

RELEASES = "https://github.com/afel6/swe015-handwritten-digitizer/releases/download"
TIMEOUT_S = 300
_USER_AGENT = "swe015-download-assets/1.0"


@dataclass(frozen=True)
class Asset:
    """One release zip: its URL, zip sha256, and the sha256 of every file that must exist after unzipping."""

    name: str
    url: str
    sha256: str
    files: Mapping[str, str] = field(default_factory=dict)


ASSETS: list[Asset] = [
    Asset("assets-v1 (vision weights + corpus cache)", f"{RELEASES}/assets-v1/swe015_assets_v1.zip", "58f5ac70400dd1c3476ce357b275fd630f451d7d6d501c05880f80138fe4cb0e", {
        "weights/dense_mlp.pt": "14a6750036d47d9263a7448f8d3c0bbe46eeeacbc47260cb70a23b3358b858c5",
        "weights/char_cnn_fp32.pt": "80adbd6872dd8f65f5ef0dca719511343b13b70dda1e1f4337691e85d65b0a26",
        "weights/char_cnn_int8.pt": "57938d4ed1d765c6c5c65ad8530ce8ddcc78dd8d340945a7de1b82424b4a205f",
        "data/raw/gutenberg/pg1342.txt": "81300b79e8a8d65ac530a97578417d06137e3bbc90622a10a65e5036183d2500",
    }),
    Asset("assets-v2 (n-gram model + INT8 BERT)", f"{RELEASES}/assets-v2/swe015_assets_v2.zip", "8f6fd20adc56a12be8d76c0911d5d0fe9145e6db8cb82b276afb5b6581d86cf0", {
        "weights/ngram_lm.pkl": "9140216631353901cdc16e375af6b42749b923bad04f0b267b9446b5d1cf1b8f",
        "weights/bert_int8_dynamic.pt": "a27b3a6771e7ddb90605d36f561059b389a663c37a7764d38ef94c5d4139d9b0",
    }),
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def _installed(asset: Asset, root: Path) -> bool:
    return all((root / name).is_file() and _sha256((root / name).read_bytes()) == digest for name, digest in asset.files.items())


def _safe_member(name: str) -> PurePosixPath:
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or ":" in name or not posix.parts:
        raise ValueError(f"unsafe path in archive: {name!r}")
    return posix


def _extract(payload: bytes, asset: Asset, target: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [(info, _safe_member(info.filename)) for info in archive.infolist()]
        for info, posix in members:
            if info.is_dir():
                continue
            out = target / posix
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    for name, digest in asset.files.items():
        path = target / name
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise ValueError(f"{name} is missing or its sha256 does not match the pinned value")


def install(asset: Asset, root: Path) -> bool:
    """Download, verify and unzip one asset into root. Returns False if it was already installed."""
    if _installed(asset, root):
        return False
    payload = _fetch(asset.url)
    if _sha256(payload) != asset.sha256:
        raise ValueError(f"downloaded zip sha256 {_sha256(payload)[:12]}... does not match the pinned {asset.sha256[:12]}...")
    with tempfile.TemporaryDirectory() as staging:
        _extract(payload, asset, Path(staging))
        for path in sorted(Path(staging).rglob("*")):
            if path.is_file():
                out = root / path.relative_to(staging)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the public model/corpus assets.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent, help="Repository root to unzip into.")
    args = parser.parse_args(argv)
    for asset in ASSETS:
        try:
            done = install(asset, args.root)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            sys.stderr.write(f"Could not install {asset.name} from {asset.url}: {' '.join(str(exc).split())}\n")
            return 1
        sys.stdout.write(f"{'Installed' if done else 'Already present'}: {asset.name} ({len(asset.files)} pinned files verified)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
