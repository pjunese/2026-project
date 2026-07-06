from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from app.config import (
    AWS_REGION,
    S3_DEFAULT_BUCKET,
    S3_ENDPOINT_URL,
    S3_PREFIX_DOC_OCR_RAW,
    S3_PREFIX_DOC_PREVIEW,
    S3_PREFIX_DOC_REGISTER_REQUEST,
    S3_PREFIX_DOC_VERIFY_REQUEST,
    S3_PREFIX_DOC_WATERMARK_RESULT,
)


DOC_KIND_TO_PREFIX = {
    "register_request": S3_PREFIX_DOC_REGISTER_REQUEST,
    "verify_request": S3_PREFIX_DOC_VERIFY_REQUEST,
    "watermarked_result": S3_PREFIX_DOC_WATERMARK_RESULT,
    "preview": S3_PREFIX_DOC_PREVIEW,
    "ocr_raw": S3_PREFIX_DOC_OCR_RAW,
}


def safe_part(v: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (v or "").strip())
    return s.strip("._-") or "unknown"


def safe_filename(name: str, fallback: str = "document.bin") -> str:
    base = Path((name or fallback).strip()).name
    cleaned = safe_part(base)
    if "." not in cleaned and "." in fallback:
        return cleaned + Path(fallback).suffix
    return cleaned or fallback


def build_document_key(
    *,
    kind: str,
    job_id: str,
    meta: dict[str, Any],
    filename: str,
) -> str:
    prefix = DOC_KIND_TO_PREFIX[kind].strip("/")
    user_id = safe_part(str(meta.get("user_id", "anon")))
    content_id = safe_part(str(meta.get("content_id", "none")))
    token = uuid.uuid4().hex[:12]
    return f"{prefix}/{user_id}/{content_id}/{safe_part(job_id)}/{token}_{safe_filename(filename)}"


def build_s3_client():
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for S3 document persistence") from exc

    return boto3.client(
        "s3",
        region_name=AWS_REGION or None,
        endpoint_url=S3_ENDPOINT_URL or None,
    )


def upload_file_to_s3(
    path: str | Path,
    *,
    kind: str,
    job_id: str,
    meta: dict[str, Any],
    bucket: str | None = None,
    filename: str | None = None,
) -> tuple[str, str]:
    resolved_bucket = (bucket or S3_DEFAULT_BUCKET).strip()
    if not resolved_bucket:
        raise RuntimeError("S3 bucket is empty. Set request.bucket or S3_DEFAULT_BUCKET.")

    local_path = Path(path)
    key = build_document_key(
        kind=kind,
        job_id=job_id,
        meta=meta,
        filename=filename or local_path.name,
    )
    build_s3_client().upload_file(str(local_path), resolved_bucket, key)
    return key, f"s3://{resolved_bucket}/{key}"


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
