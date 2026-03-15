from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from app.watermark.models import MediaInput


ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _download_to_tmp(url: str, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix.lower()
    if not suffix:
        suffix = ".bin"

    filename = hashlib.sha1(url.encode("utf-8")).hexdigest() + suffix
    out = tmp_dir / filename
    if out.exists():
        return out

    with urlopen(url) as r, out.open("wb") as f:
        f.write(r.read())
    return out


def resolve_input_to_local(media: MediaInput, tmp_dir: Path) -> Path:
    if media.local_path:
        path = Path(media.local_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input.local_path not found: {path}")
        return path

    if media.url:
        path = _download_to_tmp(media.url, tmp_dir)
        return path.resolve()

    raise ValueError("input requires one of: local_path or url")


def ensure_image_suffix(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_SUFFIX:
        raise ValueError(f"unsupported image type: {path.suffix}")
