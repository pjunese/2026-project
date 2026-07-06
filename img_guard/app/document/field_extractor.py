from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import DOC_DEFAULT_TYPE
from app.document.contracts import DocumentOcrFieldV1, DocumentOcrSummaryV1
from app.document.ocr_service import OcrPage


STOP_NAME_TOKENS = {
    "근로자",
    "사업주",
    "대표자",
    "서명",
    "성명",
    "주소",
    "주식회사",
    "넥스트플로우",
    "서울특별시",
    "강남구",
    "테헤란로",
    "수원시",
    "영통구",
    "월드컵로",
    "본사",
    "전화",
    "연락처",
}


@dataclass
class OcrToken:
    text: str
    page_index: int
    x: float
    y: float
    confidence: float | None


def _field_vertices(field: dict[str, Any]) -> list[dict[str, Any]]:
    vertices = field.get("boundingPoly", {}).get("vertices") or []
    return vertices if isinstance(vertices, list) else []


def _token_from_field(field: dict[str, Any], page_index: int) -> OcrToken:
    vertices = _field_vertices(field)
    xs = [float(v.get("x", 0)) for v in vertices]
    ys = [float(v.get("y", 0)) for v in vertices]
    return OcrToken(
        text=str(field.get("inferText") or "").strip(),
        page_index=page_index,
        x=sum(xs) / len(xs) if xs else 0.0,
        y=sum(ys) / len(ys) if ys else 0.0,
        confidence=float(field["inferConfidence"]) if field.get("inferConfidence") is not None else None,
    )


def extract_tokens(ocr_pages: list[OcrPage]) -> list[OcrToken]:
    tokens: list[OcrToken] = []
    for page in ocr_pages:
        for image in page.raw.get("images", []):
            for field in image.get("fields", []):
                tok = _token_from_field(field, page.page_index)
                if tok.text:
                    tokens.append(tok)
    return sorted(tokens, key=lambda t: (t.page_index, t.y, t.x))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u3000", " ")).strip()


def _compact(s: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", _norm_text(s)).lower()


def _is_name_candidate(text: str) -> bool:
    clean = re.sub(r"[^가-힣]", "", text or "")
    if not (2 <= len(clean) <= 4):
        return False
    if clean in STOP_NAME_TOKENS:
        return False
    if any(stop in clean for stop in STOP_NAME_TOKENS):
        return False
    return True


def _full_text(tokens: list[OcrToken]) -> str:
    return _norm_text(" ".join(t.text for t in tokens))


def _field_confidence(tokens: list[OcrToken], value: str | None) -> float | None:
    if not value:
        return None
    compact_value = _compact(value)
    hits = [t.confidence for t in tokens if t.confidence is not None and _compact(t.text) in compact_value]
    if not hits:
        hits = [t.confidence for t in tokens if t.confidence is not None and _compact(value) in _compact(t.text)]
    if not hits:
        return None
    return round(sum(hits) / len(hits), 4)


def _extract_written_date(tokens: list[OcrToken]) -> DocumentOcrFieldV1:
    text = _full_text(tokens)
    dates: list[str] = []
    pat = r"(20\d{2})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*[일]?"
    for m in re.finditer(pat, text):
        y, mo, day = m.group(1), int(m.group(2)), int(m.group(3))
        dates.append(f"{y}.{mo:02d}.{day:02d}")

    if not dates:
        return DocumentOcrFieldV1(value=None, confidence=None, source="ocr_date_regex")
    value = dates[-1]
    return DocumentOcrFieldV1(value=value, confidence=_field_confidence(tokens, value), source="ocr_date_regex")


def _tokens_after_label(tokens: list[OcrToken], label_patterns: list[str], *, limit: int = 12) -> list[OcrToken]:
    compact_patterns = [_compact(p) for p in label_patterns]
    for idx, tok in enumerate(tokens):
        tok_c = _compact(tok.text)
        if any(p and p in tok_c for p in compact_patterns):
            return tokens[idx + 1 : idx + 1 + limit]

    joined = [_compact(t.text) for t in tokens]
    for idx in range(len(tokens) - 2):
        window = "".join(joined[idx : idx + 3])
        if any(p and p in window for p in compact_patterns):
            return tokens[idx + 3 : idx + 3 + limit]
    return []


def _extract_name_near_label(tokens: list[OcrToken], labels: list[str]) -> DocumentOcrFieldV1:
    after = _tokens_after_label(tokens, labels, limit=16)
    candidates = [t for t in after if _is_name_candidate(t.text)]
    if not candidates:
        return DocumentOcrFieldV1(value=None, confidence=None, source="ocr_label_window")

    # Signatures often repeat the same name. Prefer the last plausible name near the label.
    chosen = candidates[-1]
    value = re.sub(r"[^가-힣]", "", chosen.text)
    return DocumentOcrFieldV1(value=value, confidence=chosen.confidence, source="ocr_label_window")


def _extract_worker_name(tokens: list[OcrToken]) -> DocumentOcrFieldV1:
    page2 = [t for t in tokens if t.page_index == 2]
    search_tokens = page2 or tokens

    # Standard contract page 2 usually has worker name near "(근로자)" and "성명".
    for labels in (["성명"], ["근로자"]):
        result = _extract_name_near_label(search_tokens, labels)
        if result.value:
            return result

    # Fallback: the last plausible name in lower half tends to be worker signature/name.
    lower = sorted([t for t in search_tokens if _is_name_candidate(t.text)], key=lambda t: (t.y, t.x))
    if lower:
        chosen = lower[-1]
        return DocumentOcrFieldV1(
            value=re.sub(r"[^가-힣]", "", chosen.text),
            confidence=chosen.confidence,
            source="ocr_lower_name_fallback",
        )
    return DocumentOcrFieldV1(value=None, confidence=None, source="ocr_lower_name_fallback")


def _extract_representative_name(tokens: list[OcrToken]) -> DocumentOcrFieldV1:
    page2 = [t for t in tokens if t.page_index == 2]
    search_tokens = page2 or tokens
    result = _extract_name_near_label(search_tokens, ["대표자", "대 표 자"])
    if result.value:
        return result
    return DocumentOcrFieldV1(value=None, confidence=None, source="ocr_label_window")


def _document_type(tokens: list[OcrToken], requested_type: str | None) -> tuple[str, bool]:
    text = _full_text(tokens)
    compact = _compact(text)
    looks_like_labor_contract = "근로계약서" in compact or ("근로계약기간" in compact and "임금" in compact)
    return requested_type or DOC_DEFAULT_TYPE, looks_like_labor_contract


def extract_contract_summary(
    ocr_pages: list[OcrPage],
    *,
    document_type: str | None = None,
) -> DocumentOcrSummaryV1:
    tokens = extract_tokens(ocr_pages)
    doc_type, type_ok = _document_type(tokens, document_type)

    rep = _extract_representative_name(tokens)
    worker = _extract_worker_name(tokens)
    written = _extract_written_date(tokens)

    fields = {
        "representative_name": rep,
        "worker_name": worker,
        "written_date": written,
    }
    missing = [name for name, field in fields.items() if not field.value]
    extracted_count = len(fields) - len(missing)

    if extracted_count >= 2 and type_ok:
        status = "verified"
        reason = "document OCR summary extracted"
    elif extracted_count >= 1:
        status = "review"
        reason = "document OCR summary partially extracted"
    else:
        status = "review"
        reason = "document OCR summary missing required fields"

    return DocumentOcrSummaryV1(
        document_type=doc_type,
        status=status,
        representative_name=rep,
        worker_name=worker,
        written_date=written,
        extracted_count=extracted_count,
        missing_fields=missing,
        reason=reason,
    )
