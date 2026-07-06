from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DOC_TMP_DIR
from app.contracts_v1 import WatermarkOptionsV1
from app.document.render_service import images_to_pdf
from app.watermark.models import (
    MediaInput,
    WatermarkDetectOptions,
    WatermarkDetectRequest,
    WatermarkEmbedOptions,
    WatermarkEmbedRequest,
)
from app.watermark.service import WatermarkService


def _to_embed_options(opt: WatermarkOptionsV1 | None) -> WatermarkEmbedOptions:
    if opt is None:
        return WatermarkEmbedOptions()
    return WatermarkEmbedOptions(
        model=opt.model or "wam",
        nbits=opt.nbits if opt.nbits is not None else 32,
        scaling_w=opt.scaling_w if opt.scaling_w is not None else 2.0,
        proportion_masked=opt.proportion_masked if opt.proportion_masked is not None else 0.65,
    )


def embed_watermark_into_pages(
    *,
    job_id: str,
    page_paths: list[Path],
    meta: dict[str, Any],
    options: WatermarkOptionsV1 | None = None,
    watermark_all_pages: bool = True,
) -> tuple[list[Path], dict[str, Any]]:
    if not page_paths:
        raise ValueError("page_paths is empty")

    service = WatermarkService.create()
    target_pages = page_paths if watermark_all_pages else [page_paths[0]]
    output_pages: list[Path] = []
    page_results: list[dict[str, Any]] = []
    payload_id: str | None = None

    for idx, page_path in enumerate(page_paths, start=1):
        if page_path not in target_pages:
            output_pages.append(page_path)
            page_results.append({"page": idx, "applied": False, "reason": "skipped"})
            continue

        resp = service.embed(
            WatermarkEmbedRequest(
                job_id=f"{job_id}-page-{idx:03d}",
                input=MediaInput(local_path=str(page_path), filename=page_path.name, mime_type="image/png"),
                meta=meta,
                options=_to_embed_options(options),
            )
        )
        if not resp.success or not resp.result.output_path:
            raise RuntimeError(f"watermark embed failed on page {idx}: {resp.reason}")

        output_path = Path(resp.result.output_path)
        output_pages.append(output_path)
        payload_id = payload_id or resp.result.payload_id
        page_results.append(
            {
                "page": idx,
                "applied": bool(resp.result.applied),
                "payload_id": resp.result.payload_id,
                "output_path": resp.result.output_path,
                "model": resp.result.model,
                "model_version": resp.result.model_version,
                "details": resp.result.details,
            }
        )

    out_pdf = DOC_TMP_DIR / job_id / "watermarked" / "document_watermarked.pdf"
    images_to_pdf(output_pages, out_pdf)
    return output_pages, {"output_pdf": str(out_pdf), "payload_id": payload_id, "page_results": page_results}


def detect_watermark_from_pages(
    *,
    job_id: str,
    page_paths: list[Path],
    threshold: float = 0.5,
) -> dict[str, Any]:
    service = WatermarkService.create()
    page_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, page_path in enumerate(page_paths, start=1):
        resp = service.detect(
            WatermarkDetectRequest(
                job_id=f"{job_id}-detect-page-{idx:03d}",
                input=MediaInput(local_path=str(page_path), filename=page_path.name, mime_type="image/png"),
                options=WatermarkDetectOptions(threshold=threshold),
            )
        )
        result = resp.result
        item = {
            "page": idx,
            "success": resp.success,
            "reason": resp.reason,
            "detected": bool(result.detected),
            "confidence": result.confidence,
            "payload_id": result.payload_id,
            "model": result.model,
            "model_version": result.model_version,
            "details": result.details,
        }
        page_results.append(item)
        if best is None or (item.get("confidence") or 0.0) > (best.get("confidence") or 0.0):
            best = item
        if result.detected:
            break

    return {
        "detected": bool(best and best.get("detected")),
        "payload_id": best.get("payload_id") if best else None,
        "best_page": best,
        "page_results": page_results,
    }
