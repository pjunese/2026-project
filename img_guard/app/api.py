"""
FastAPI layer for img_guard.

Run:
  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional, List
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import imagehash

from app.config import TOP_K, TMP_DIR
from app.embedder import ClipEmbedder
from app.ann_index import ANNIndex
from app.phash import PHashComparator
from app.policy import PolicyEngine
from app.preprocess import load_image_fixed
from app.types import ANNResult, GuardResult


app = FastAPI(title="img_guard API", version="v1")


# --------
# Models
# --------

class InputItem(BaseModel):
    url: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None


class SearchOptions(BaseModel):
    top_k: Optional[int] = None
    top_phash: Optional[int] = None


class WatermarkOptions(BaseModel):
    apply_on_allow: Optional[bool] = None
    model: Optional[str] = None
    nbits: Optional[int] = None
    scaling_w: Optional[float] = None
    proportion_masked: Optional[float] = None


class Options(BaseModel):
    search: Optional[SearchOptions] = None
    watermark: Optional[WatermarkOptions] = None


class GuardRequest(BaseModel):
    job_id: str
    mode: str
    content_type: str
    input: List[InputItem] | InputItem
    meta: Optional[dict] = None
    options: Optional[Options] = None


# --------
# Engine (singleton)
# --------

_ENGINE: dict[str, Any] = {}


def _get_engine():
    if "engine" not in _ENGINE:
        _ENGINE["engine"] = {
            "embedder": ClipEmbedder(),
            "ann": ANNIndex(),
            "phash": PHashComparator(),
            "policy": PolicyEngine(),
        }
    return _ENGINE["engine"]


# --------
# Helpers
# --------

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


def _phash_to_int(ph) -> int:
    if isinstance(ph, int):
        return ph
    if isinstance(ph, str):
        return int(ph, 16)
    # imagehash.ImageHash
    return int(str(ph), 16)


def _hamming_dist(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _ann_to_dict(a: ANNResult) -> dict:
    return {
        "db_key": getattr(a, "db_key", None),
        "db_file": a.db_file,
        "cosine": round(float(a.cosine), 6),
        "phash_dist": None if a.phash_dist is None else int(a.phash_dist),
    }


def _guard_to_dict(r: GuardResult) -> dict:
    return {
        "decision": r.decision.value,
        "reason": r.reason,
        "top_match": None if r.top_match is None else _ann_to_dict(r.top_match),
        "candidates": [_ann_to_dict(c) for c in r.candidates],
    }


# --------
# Routes
# --------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/guard/image")
def guard_image(req: GuardRequest):
    if req.content_type.lower() != "image":
        raise HTTPException(status_code=400, detail="content_type must be 'image'")

    inputs = req.input if isinstance(req.input, list) else [req.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="input is empty")

    item = inputs[0]
    if not item.url:
        raise HTTPException(status_code=400, detail="input.url is required")

    engine = _get_engine()
    top_k = req.options.search.top_k if req.options and req.options.search and req.options.search.top_k else TOP_K
    top_phash = req.options.search.top_phash if req.options and req.options.search else None

    t0 = time.perf_counter()
    # download
    t_download_start = time.perf_counter()
    local_path = _download_to_tmp(item.url)
    t_download = (time.perf_counter() - t_download_start) * 1000

    # embedding
    t_embed_start = time.perf_counter()
    q_vec = engine["embedder"].embed_paths([local_path], batch_size=1)[0]
    t_embed = (time.perf_counter() - t_embed_start) * 1000

    # ANN search
    t_ann_start = time.perf_counter()
    candidates = engine["ann"].search(q_vec, k=top_k)
    t_ann = (time.perf_counter() - t_ann_start) * 1000

    # pHash
    t_phash_start = time.perf_counter()
    q_ph = _phash_to_int(imagehash.phash(load_image_fixed(local_path)))
    # if db_phash present, compute directly
    used_db_phash = False
    for c in candidates:
        db_ph = getattr(c, "db_phash", None)
        if db_ph is not None:
            used_db_phash = True
            c.phash_dist = _hamming_dist(q_ph, _phash_to_int(db_ph))

    if not used_db_phash:
        candidates = engine["phash"].enrich(
            query_path=local_path,
            candidates=candidates,
            resolve_path_fn=engine["ann"].get_full_path,
            top_n=top_phash,
        )
    t_phash = (time.perf_counter() - t_phash_start) * 1000

    # policy
    result = engine["policy"].decide(candidates)
    t_total = (time.perf_counter() - t0) * 1000

    # build response
    payload = _guard_to_dict(result)
    top = result.top_match
    scores = {
        "top_cosine": None if top is None else round(float(top.cosine), 6),
        "top_phash_dist": None if top is None else top.phash_dist,
        "policy_version": "v1",
    }

    watermark_req = req.options.watermark if req.options else None
    watermark_requested = bool(watermark_req and watermark_req.apply_on_allow)

    return {
        "job_id": req.job_id,
        "mode": req.mode,
        "content_type": req.content_type,
        "success": True,
        "decision": payload["decision"],
        "reason": payload["reason"],
        "next_action": "start_vote" if payload["decision"] == "review" else "none",
        "scores": scores,
        "top_match": payload["top_match"],
        "candidates": payload["candidates"],
        "watermark": {
            "requested": watermark_requested,
            "applied": False,
            "output_url": None,
            "output_key": None,
            "model": getattr(watermark_req, "model", None) if watermark_req else None,
            "model_version": None,
            "nbits": getattr(watermark_req, "nbits", None) if watermark_req else None,
            "scaling_w": getattr(watermark_req, "scaling_w", None) if watermark_req else None,
            "proportion_masked": getattr(watermark_req, "proportion_masked", None) if watermark_req else None,
            "payload_id": None,
        },
        "timing_ms": {
            "download": int(t_download),
            "embed": int(t_embed),
            "ann_search": int(t_ann),
            "phash": int(t_phash),
            "total": int(t_total),
        },
    }
