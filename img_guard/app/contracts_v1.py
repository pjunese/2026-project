"""
V1 contract models shared by:
- FastAPI REST layer
- internal function-call integration (Django -> AI module)

If backend calls functions instead of HTTP, keep the same schema here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DecisionV1 = Literal["allow", "review", "block"]
NextActionV1 = Literal["none", "start_vote"]


class InputItemV1(BaseModel):
    url: str
    filename: str | None = None
    mime_type: str | None = None


class SearchOptionsV1(BaseModel):
    top_k: int | None = None
    top_phash: int | None = None


class WatermarkOptionsV1(BaseModel):
    apply_on_allow: bool | None = None
    model: str | None = None
    nbits: int | None = None
    scaling_w: float | None = None
    proportion_masked: float | None = None


class GuardOptionsV1(BaseModel):
    search: SearchOptionsV1 | None = None
    watermark: WatermarkOptionsV1 | None = None


class GuardRequestV1(BaseModel):
    job_id: str
    mode: str = "register"
    content_type: str = "image"
    input: list[InputItemV1] = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)
    options: GuardOptionsV1 | None = None

    @field_validator("mode")
    @classmethod
    def normalize_mode(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("content_type")
    @classmethod
    def normalize_content_type(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("input", mode="before")
    @classmethod
    def normalize_input_list(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [v]
        return v


class MatchItemV1(BaseModel):
    db_key: str | None = None
    db_file: str
    cosine: float
    phash_dist: int | None = None


class ScoreV1(BaseModel):
    top_cosine: float | None = None
    top_phash_dist: int | None = None
    policy_version: str = "v1"


class WatermarkResultV1(BaseModel):
    requested: bool = False
    applied: bool = False
    output_url: str | None = None
    output_key: str | None = None
    model: str | None = None
    model_version: str | None = None
    nbits: int | None = None
    scaling_w: float | None = None
    proportion_masked: float | None = None
    payload_id: str | None = None


class TimingV1(BaseModel):
    download: int
    embed: int
    ann_search: int
    phash: int
    total: int


class GuardResponseV1(BaseModel):
    job_id: str
    mode: str
    content_type: str
    success: bool
    decision: DecisionV1
    reason: str
    next_action: NextActionV1
    scores: ScoreV1
    top_match: MatchItemV1 | None
    candidates: list[MatchItemV1]
    watermark: WatermarkResultV1
    timing_ms: TimingV1
