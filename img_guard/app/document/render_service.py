from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from app.config import DOC_MAX_PAGES, DOC_RENDER_DPI, DOC_TMP_DIR, S3_DEFAULT_BUCKET
from app.contracts_v1 import InputItemV1
from app.source_io import resolve_source_to_local


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class RenderedDocument:
    source_path: Path
    pdf_path: Path | None
    page_paths: list[Path]


def resolve_document_input(item: InputItemV1, job_id: str) -> Path:
    source = item.local_path or item.url or item.s3_uri or item.s3_key
    if not source:
        raise ValueError("document input requires one of: local_path, url, s3_uri, s3_key")
    local_path = resolve_source_to_local(source, DOC_TMP_DIR / job_id / "input", default_s3_bucket=S3_DEFAULT_BUCKET)
    suffix = local_path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"unsupported document type: {suffix}")
    return local_path


def _convert_office_to_pdf(input_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("*.pdf"))
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(input_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("libreoffice is required for DOC/DOCX document rendering") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"libreoffice conversion failed: {err}") from exc

    after = set(out_dir.glob("*.pdf"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_files:
        return new_files[0]

    candidate = out_dir / f"{input_path.stem}.pdf"
    if candidate.exists():
        return candidate
    raise RuntimeError(f"PDF conversion output not found for: {input_path}")


def _render_pdf_to_pages(pdf_path: Path, out_dir: Path, max_pages: int) -> list[Path]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("pymupdf is required for PDF rendering. Install package: pymupdf") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    page_count = min(len(doc), max_pages)
    if page_count <= 0:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")

    zoom = DOC_RENDER_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page_paths: list[Path] = []
    for idx in range(page_count):
        page = doc[idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"page_{idx + 1:03d}.png"
        pix.save(str(out_path))
        page_paths.append(out_path)
    doc.close()
    return page_paths


def _copy_image_as_page(input_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    out_path = out_dir / f"{input_path.stem}.png"
    img.save(out_path)
    return [out_path]


def render_document_to_images(
    item: InputItemV1,
    *,
    job_id: str,
    max_pages: int | None = None,
) -> RenderedDocument:
    source_path = resolve_document_input(item, job_id)
    out_root = DOC_TMP_DIR / job_id
    page_dir = out_root / "pages"
    max_page_count = max(1, int(max_pages or DOC_MAX_PAGES))
    suffix = source_path.suffix.lower()

    pdf_path: Path | None = None
    if suffix == ".pdf":
        pdf_path = source_path
        pages = _render_pdf_to_pages(source_path, page_dir, max_page_count)
    elif suffix in {".doc", ".docx"}:
        pdf_path = _convert_office_to_pdf(source_path, out_root / "pdf")
        pages = _render_pdf_to_pages(pdf_path, page_dir, max_page_count)
    else:
        pages = _copy_image_as_page(source_path, page_dir)

    return RenderedDocument(source_path=source_path, pdf_path=pdf_path, page_paths=pages)


def images_to_pdf(image_paths: list[Path], output_path: Path) -> Path:
    if not image_paths:
        raise ValueError("image_paths is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    try:
        first, rest = images[0], images[1:]
        first.save(output_path, "PDF", save_all=bool(rest), append_images=rest)
    finally:
        for img in images:
            img.close()
    return output_path


def copy_original_document(source_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / source_path.name
    if source_path.resolve() != out.resolve():
        shutil.copy2(source_path, out)
    return out
