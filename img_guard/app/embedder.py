# app/embedder.py
"""
embedder.py
CLIP 임베딩(벡터) 생성 담당

원칙:
1) 모델은 한 번만 로드해서 재사용 (속도/메모리)
2) 출력 벡터는 L2 normalize 해서 cosine similarity에 바로 쓰기
3) 반환은 numpy float32 (hnswlib/저장에 최적)
"""

from __future__ import annotations

import numpy as np
import torch
import open_clip

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED
from app.preprocess import load_image_fixed


class ClipEmbedder:
    """
    CLIP 이미지 임베딩 생성기
    - encode_image 결과를 L2 정규화해서 반환
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    @torch.no_grad()
    def embed_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        feats: list[torch.Tensor] = []

        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]

            imgs = [self.preprocess(load_image_fixed(p)) for p in batch]
            x = torch.stack(imgs).to(self.device)

            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)  # L2 normalize
            feats.append(f.cpu())

        out = torch.cat(feats, dim=0).numpy().astype(np.float32)
        return out