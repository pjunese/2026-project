from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.contracts_v1 import InputItemV1, WatermarkOptionsV1
from app.watermark.models import WatermarkDetectResult


DocumentDecisionV1 = Literal["verified", "review", "failed"]


class DocumentWorkflowOptionsV1(BaseModel):
    archive_original: bool = True
    archive_watermarked: bool = True
    archive_ocr_raw: bool = True
    run_ocr: bool = True
    watermark_all_pages: bool = True
    max_pages: int | None = None


class DocumentRegisterWorkflowRequestV1(BaseModel):
    job_id: str
    input: InputItemV1
    meta: dict[str, Any] = Field(default_factory=dict)
    bucket: str | None = None
    document_type: str | None = None
    watermark_options: WatermarkOptionsV1 | None = None
    options: DocumentWorkflowOptionsV1 = Field(default_factory=DocumentWorkflowOptionsV1)


class DocumentVerifyWorkflowRequestV1(BaseModel):
    job_id: str
    input: InputItemV1
    meta: dict[str, Any] = Field(default_factory=dict)
    bucket: str | None = None
    document_type: str | None = None
    run_ocr: bool = False
    max_pages: int | None = None


class DocumentAssetsV1(BaseModel):
    original_s3_key: str | None = None
    watermarked_s3_key: str | None = None
    ocr_raw_s3_key: str | None = None
    preview_s3_keys: list[str] = Field(default_factory=list)
    local_original_path: str | None = None
    local_watermarked_path: str | None = None
    local_page_paths: list[str] = Field(default_factory=list)
    local_watermarked_page_paths: list[str] = Field(default_factory=list)


class DocumentOcrFieldV1(BaseModel):
    value: str | None = None
    confidence: float | None = None
    source: str | None = None


class DocumentOcrSummaryV1(BaseModel):
    document_type: str
    status: DocumentDecisionV1
    representative_name: DocumentOcrFieldV1 = Field(default_factory=DocumentOcrFieldV1)
    worker_name: DocumentOcrFieldV1 = Field(default_factory=DocumentOcrFieldV1)
    written_date: DocumentOcrFieldV1 = Field(default_factory=DocumentOcrFieldV1)
    extracted_count: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    reason: str | None = None


class DocumentWatermarkSummaryV1(BaseModel):
    requested: bool = True
    applied: bool = False
    detected: bool = False
    payload_id: str | None = None
    output_path: str | None = None
    output_key: str | None = None
    page_results: list[dict[str, Any]] = Field(default_factory=list)
    detect_result: WatermarkDetectResult | None = None


class DocumentRegisterWorkflowResponseV1(BaseModel):
    job_id: str
    success: bool
    decision: DocumentDecisionV1
    reason: str
    document_type: str
    assets: DocumentAssetsV1 = Field(default_factory=DocumentAssetsV1)
    watermark: DocumentWatermarkSummaryV1 = Field(default_factory=DocumentWatermarkSummaryV1)
    ocr_summary: DocumentOcrSummaryV1 | None = None
    pending_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentVerifyWorkflowResponseV1(BaseModel):
    job_id: str
    success: bool
    decision: DocumentDecisionV1
    reason: str
    document_type: str
    assets: DocumentAssetsV1 = Field(default_factory=DocumentAssetsV1)
    watermark: DocumentWatermarkSummaryV1 = Field(default_factory=DocumentWatermarkSummaryV1)
    ocr_summary: DocumentOcrSummaryV1 | None = None
    pending_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
