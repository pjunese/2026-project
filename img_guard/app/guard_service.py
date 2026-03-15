"""
Guard service orchestration shared by:
- REST API route (/v1/guard/image)
- internal function calls (same process integration)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import imagehash

from app.ann_index import ANNIndex
from app.config import TOP_K, TMP_DIR
from app.contracts_v1 import (
    GuardRequestV1,
    GuardResponseV1,
    MatchItemV1,
    ScoreV1,
    TimingV1,
    WatermarkResultV1,
)
from app.embedder import ClipEmbedder
from app.phash import PHashComparator
from app.policy import PolicyEngine
from app.preprocess import load_image_fixed
from app.types import ANNResult, GuardResult

_ENGINE: dict[str, Any] = {}


def _get_engine() -> dict[str, Any]:
    if "engine" not in _ENGINE:
        _ENGINE["engine"] = {
            "embedder": ClipEmbedder(),
            "ann": ANNIndex(),
            "phash": PHashComparator(),
            "policy": PolicyEngine(),
        }
    return _ENGINE["engine"]


def _download_to_tmp(url: str) -> str:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix or ".bin"
    name = hashlib.sha1(url.encode("utf-8")).hexdigest() + suffix
    out_path = TMP_DIR / name
    if out_path.exists():
        return str(out_path)
    with urlopen(url) as r, out_path.open("wb") as f:
        f.write(r.read())
    return str(out_path)


def _phash_to_int(ph: Any) -> int:
    if isinstance(ph, int):
        return ph
    if isinstance(ph, str):
        return int(ph, 16)
    # imagehash.ImageHash
    return int(str(ph), 16)


def _hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _ann_to_contract(a: ANNResult) -> MatchItemV1:
    return MatchItemV1(
        db_key=getattr(a, "db_key", None),
        db_file=a.db_file,
        cosine=round(float(a.cosine), 6),
        phash_dist=None if a.phash_dist is None else int(a.phash_dist),
    )


def _decision_next_action(decision: str) -> str:
    return "start_vote" if decision == "review" else "none"


def _to_guard_request(req: GuardRequestV1 | dict[str, Any]) -> GuardRequestV1:
    if isinstance(req, GuardRequestV1):
        return req
    return GuardRequestV1.model_validate(req)


def run_guard_v1(req: GuardRequestV1 | dict[str, Any]) -> GuardResponseV1:
    """
    Main function-call entrypoint for backend integration.
    - Input: GuardRequestV1 (or dict following same schema)
    - Output: GuardResponseV1
    """
    parsed = _to_guard_request(req)

    if parsed.content_type != "image":
        raise ValueError("content_type must be 'image'")
    if not parsed.input:
        raise ValueError("input is empty")

    item = parsed.input[0]
    if not item.url:
        raise ValueError("input.url is required")

    top_k = (
        parsed.options.search.top_k
        if parsed.options and parsed.options.search and parsed.options.search.top_k
        else TOP_K
    )
    top_phash = parsed.options.search.top_phash if parsed.options and parsed.options.search else None

    engine = _get_engine()
    t0 = time.perf_counter()

    t_download_start = time.perf_counter()
    local_path = _download_to_tmp(item.url)
    t_download = (time.perf_counter() - t_download_start) * 1000

    t_embed_start = time.perf_counter()
    q_vec = engine["embedder"].embed_paths([local_path], batch_size=1)[0]
    t_embed = (time.perf_counter() - t_embed_start) * 1000

    t_ann_start = time.perf_counter()
    candidates = engine["ann"].search(q_vec, k=top_k)
    t_ann = (time.perf_counter() - t_ann_start) * 1000

    t_phash_start = time.perf_counter()
    q_ph = _phash_to_int(imagehash.phash(load_image_fixed(local_path)))

    missing_phash: list[ANNResult] = []
    for cand in candidates:
        db_ph = getattr(cand, "db_phash", None)
        if db_ph is not None:
            cand.phash_dist = _hamming_dist(q_ph, _phash_to_int(db_ph))
        else:
            missing_phash.append(cand)

    if missing_phash:
        engine["phash"].enrich(
            query_path=local_path,
            candidates=missing_phash,
            resolve_path_fn=engine["ann"].get_full_path,
            top_n=top_phash,
        )

    t_phash = (time.perf_counter() - t_phash_start) * 1000

    result: GuardResult = engine["policy"].decide(candidates)
    t_total = (time.perf_counter() - t0) * 1000

    top = result.top_match
    decision = result.decision.value
    watermark_req = parsed.options.watermark if parsed.options else None
    watermark_requested = bool(watermark_req and watermark_req.apply_on_allow)

    return GuardResponseV1(
        job_id=parsed.job_id,
        mode=parsed.mode,
        content_type=parsed.content_type,
        success=True,
        decision=decision,
        reason=result.reason,
        next_action=_decision_next_action(decision),
        scores=ScoreV1(
            top_cosine=None if top is None else round(float(top.cosine), 6),
            top_phash_dist=None if top is None else top.phash_dist,
            policy_version="v1",
        ),
        top_match=None if top is None else _ann_to_contract(top),
        candidates=[_ann_to_contract(c) for c in result.candidates],
        watermark=WatermarkResultV1(
            requested=watermark_requested,
            applied=False,
            output_url=None,
            output_key=None,
            model=watermark_req.model if watermark_req else None,
            model_version=None,
            nbits=watermark_req.nbits if watermark_req else None,
            scaling_w=watermark_req.scaling_w if watermark_req else None,
            proportion_masked=watermark_req.proportion_masked if watermark_req else None,
            payload_id=None,
        ),
        timing_ms=TimingV1(
            download=int(t_download),
            embed=int(t_embed),
            ann_search=int(t_ann),
            phash=int(t_phash),
            total=int(t_total),
        ),
    )
