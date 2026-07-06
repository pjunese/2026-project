from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DOC_DEFAULT_TYPE, DOC_TMP_DIR
from app.document.contracts import (
    DocumentAssetsV1,
    DocumentOcrSummaryV1,
    DocumentRegisterWorkflowRequestV1,
    DocumentRegisterWorkflowResponseV1,
    DocumentVerifyWorkflowRequestV1,
    DocumentVerifyWorkflowResponseV1,
    DocumentWatermarkSummaryV1,
)
from app.document.field_extractor import extract_contract_summary
from app.document.ocr_service import ocr_pages_to_jsonable, run_ocr_on_pages
from app.document.render_service import render_document_to_images
from app.document.storage import upload_file_to_s3, write_json
from app.document.watermark_service import detect_watermark_from_pages, embed_watermark_into_pages


def _document_type(value: str | None) -> str:
    return value or DOC_DEFAULT_TYPE


def _try_upload(
    *,
    path: str | Path,
    kind: str,
    job_id: str,
    meta: dict[str, Any],
    bucket: str | None,
    filename: str | None = None,
    warnings: list[str],
) -> str | None:
    try:
        key, _ = upload_file_to_s3(
            path,
            kind=kind,
            job_id=job_id,
            meta=meta,
            bucket=bucket,
            filename=filename,
        )
        return key
    except Exception as exc:
        warnings.append(f"archive({kind}) failed: {exc}")
        return None


def _run_ocr_summary(
    *,
    job_id: str,
    page_paths: list[Path],
    document_type: str,
    meta: dict[str, Any],
    bucket: str | None,
    archive_ocr_raw: bool,
    warnings: list[str],
) -> tuple[DocumentOcrSummaryV1 | None, str | None]:
    try:
        ocr_pages = run_ocr_on_pages(page_paths)
        summary = extract_contract_summary(ocr_pages, document_type=document_type)
        raw_payload = {
            "job_id": job_id,
            "document_type": document_type,
            "summary": summary.model_dump(),
            "pages": ocr_pages_to_jsonable(ocr_pages),
        }
        raw_key = None
        if archive_ocr_raw:
            raw_path = write_json(DOC_TMP_DIR / job_id / "ocr" / "ocr_raw.json", raw_payload)
            raw_key = _try_upload(
                path=raw_path,
                kind="ocr_raw",
                job_id=job_id,
                meta=meta,
                bucket=bucket,
                filename="ocr_raw.json",
                warnings=warnings,
            )
        return summary, raw_key
    except Exception as exc:
        warnings.append(f"ocr failed: {exc}")
        return None, None


def run_document_register_workflow_v1(
    req: DocumentRegisterWorkflowRequestV1 | dict[str, Any],
) -> DocumentRegisterWorkflowResponseV1:
    parsed = (
        req
        if isinstance(req, DocumentRegisterWorkflowRequestV1)
        else DocumentRegisterWorkflowRequestV1.model_validate(req)
    )

    document_type = _document_type(parsed.document_type)
    assets = DocumentAssetsV1()
    watermark = DocumentWatermarkSummaryV1(requested=True)
    warnings: list[str] = []
    pending_actions = ["mint_token_with_existing_image_fields", "save_document_ocr_summary_to_db"]

    try:
        rendered = render_document_to_images(
            parsed.input,
            job_id=parsed.job_id,
            max_pages=parsed.options.max_pages,
        )
        assets.local_original_path = str(rendered.source_path)
        assets.local_page_paths = [str(p) for p in rendered.page_paths]

        if parsed.options.archive_original:
            assets.original_s3_key = _try_upload(
                path=rendered.source_path,
                kind="register_request",
                job_id=parsed.job_id,
                meta=parsed.meta,
                bucket=parsed.bucket,
                filename=parsed.input.filename or rendered.source_path.name,
                warnings=warnings,
            )

        watermarked_pages, wm_info = embed_watermark_into_pages(
            job_id=parsed.job_id,
            page_paths=rendered.page_paths,
            meta=parsed.meta,
            options=parsed.watermark_options,
            watermark_all_pages=parsed.options.watermark_all_pages,
        )
        watermarked_pdf = Path(wm_info["output_pdf"])
        assets.local_watermarked_path = str(watermarked_pdf)
        assets.local_watermarked_page_paths = [str(p) for p in watermarked_pages]
        watermark.applied = True
        watermark.payload_id = wm_info.get("payload_id")
        watermark.output_path = str(watermarked_pdf)
        watermark.page_results = wm_info.get("page_results", [])

        if parsed.options.archive_watermarked:
            assets.watermarked_s3_key = _try_upload(
                path=watermarked_pdf,
                kind="watermarked_result",
                job_id=parsed.job_id,
                meta=parsed.meta,
                bucket=parsed.bucket,
                filename=f"{Path(parsed.input.filename or rendered.source_path.name).stem}_watermarked.pdf",
                warnings=warnings,
            )
            watermark.output_key = assets.watermarked_s3_key

        ocr_summary = None
        if parsed.options.run_ocr:
            ocr_summary, raw_key = _run_ocr_summary(
                job_id=parsed.job_id,
                page_paths=rendered.page_paths,
                document_type=document_type,
                meta=parsed.meta,
                bucket=parsed.bucket,
                archive_ocr_raw=parsed.options.archive_ocr_raw,
                warnings=warnings,
            )
            assets.ocr_raw_s3_key = raw_key

        if ocr_summary is None:
            decision = "review"
            reason = "document watermarked; OCR summary unavailable or skipped"
        elif ocr_summary.status == "verified":
            decision = "verified"
            reason = "document watermarked and OCR summary extracted"
        else:
            decision = "review"
            reason = ocr_summary.reason or "document watermarked; OCR summary requires review"

        return DocumentRegisterWorkflowResponseV1(
            job_id=parsed.job_id,
            success=True,
            decision=decision,
            reason=reason,
            document_type=document_type,
            assets=assets,
            watermark=watermark,
            ocr_summary=ocr_summary,
            pending_actions=pending_actions,
            warnings=warnings,
        )

    except Exception as exc:
        return DocumentRegisterWorkflowResponseV1(
            job_id=parsed.job_id,
            success=False,
            decision="failed",
            reason=str(exc),
            document_type=document_type,
            assets=assets,
            watermark=watermark,
            ocr_summary=None,
            pending_actions=[],
            warnings=warnings,
        )


def run_document_verify_workflow_v1(
    req: DocumentVerifyWorkflowRequestV1 | dict[str, Any],
) -> DocumentVerifyWorkflowResponseV1:
    parsed = (
        req
        if isinstance(req, DocumentVerifyWorkflowRequestV1)
        else DocumentVerifyWorkflowRequestV1.model_validate(req)
    )

    document_type = _document_type(parsed.document_type)
    assets = DocumentAssetsV1()
    watermark = DocumentWatermarkSummaryV1(requested=True)
    warnings: list[str] = []

    try:
        rendered = render_document_to_images(
            parsed.input,
            job_id=parsed.job_id,
            max_pages=parsed.max_pages,
        )
        assets.local_original_path = str(rendered.source_path)
        assets.local_page_paths = [str(p) for p in rendered.page_paths]

        assets.original_s3_key = _try_upload(
            path=rendered.source_path,
            kind="verify_request",
            job_id=parsed.job_id,
            meta=parsed.meta,
            bucket=parsed.bucket,
            filename=parsed.input.filename or rendered.source_path.name,
            warnings=warnings,
        )

        det = detect_watermark_from_pages(job_id=parsed.job_id, page_paths=rendered.page_paths)
        watermark.detected = bool(det.get("detected"))
        watermark.payload_id = det.get("payload_id")
        watermark.page_results = det.get("page_results", [])

        ocr_summary = None
        if parsed.run_ocr:
            ocr_summary, raw_key = _run_ocr_summary(
                job_id=parsed.job_id,
                page_paths=rendered.page_paths,
                document_type=document_type,
                meta=parsed.meta,
                bucket=parsed.bucket,
                archive_ocr_raw=True,
                warnings=warnings,
            )
            assets.ocr_raw_s3_key = raw_key

        if watermark.detected:
            decision = "verified"
            reason = "document watermark detected"
            pending_actions = ["lookup_token_by_payload_id"]
        else:
            decision = "review"
            reason = "document watermark not detected; manual or token check required"
            pending_actions = ["manual_review"]

        return DocumentVerifyWorkflowResponseV1(
            job_id=parsed.job_id,
            success=True,
            decision=decision,
            reason=reason,
            document_type=document_type,
            assets=assets,
            watermark=watermark,
            ocr_summary=ocr_summary,
            pending_actions=pending_actions,
            warnings=warnings,
        )

    except Exception as exc:
        return DocumentVerifyWorkflowResponseV1(
            job_id=parsed.job_id,
            success=False,
            decision="failed",
            reason=str(exc),
            document_type=document_type,
            assets=assets,
            watermark=watermark,
            ocr_summary=None,
            pending_actions=[],
            warnings=warnings,
        )
