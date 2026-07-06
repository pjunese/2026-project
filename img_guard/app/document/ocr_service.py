from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import CLOVA_OCR_INVOKE_URL, CLOVA_OCR_SECRET, DOC_OCR_TIMEOUT_SEC


@dataclass
class OcrPage:
    page_index: int
    image_path: str
    raw: dict[str, Any]


def _image_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        return "jpeg"
    if suffix in {"jpeg", "png", "bmp", "tiff"}:
        return suffix
    return "png"


def call_clova_ocr(image_path: str | Path) -> dict[str, Any]:
    if not CLOVA_OCR_INVOKE_URL or not CLOVA_OCR_SECRET:
        raise RuntimeError("CLOVA_OCR_INVOKE_URL and CLOVA_OCR_SECRET must be set for document OCR.")

    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests is required for CLOVA OCR") from exc

    path = Path(image_path)
    message = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "images": [
            {
                "format": _image_format(path),
                "name": path.stem,
            }
        ],
    }

    with path.open("rb") as f:
        files = {
            "message": (None, json.dumps(message, ensure_ascii=False), "application/json"),
            "file": (path.name, f, f"image/{_image_format(path)}"),
        }
        resp = requests.post(
            CLOVA_OCR_INVOKE_URL,
            headers={"X-OCR-SECRET": CLOVA_OCR_SECRET},
            files=files,
            timeout=DOC_OCR_TIMEOUT_SEC,
        )

    if resp.status_code >= 400:
        body = resp.text[:1000]
        raise RuntimeError(f"CLOVA OCR failed: status={resp.status_code}, body={body}")
    return resp.json()


def run_ocr_on_pages(page_paths: list[Path]) -> list[OcrPage]:
    pages: list[OcrPage] = []
    for idx, path in enumerate(page_paths, start=1):
        raw = call_clova_ocr(path)
        pages.append(OcrPage(page_index=idx, image_path=str(path), raw=raw))
    return pages


def ocr_pages_to_jsonable(pages: list[OcrPage]) -> list[dict[str, Any]]:
    return [
        {
            "page_index": p.page_index,
            "image_path": p.image_path,
            "raw": p.raw,
        }
        for p in pages
    ]
