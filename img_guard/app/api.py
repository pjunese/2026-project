"""
FastAPI layer for img_guard.

Run:
  uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.contracts_v1 import GuardRequestV1, GuardResponseV1
from app.guard_service import run_guard_v1
from app.watermark.router import router as watermark_router

app = FastAPI(title="img_guard API", version="v1")
app.include_router(watermark_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/guard/image", response_model=GuardResponseV1)
def guard_image(req: GuardRequestV1):
    try:
        return run_guard_v1(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
