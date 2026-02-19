# app/config.py
"""
프로젝트 전체가 공유하는 기준값(경로/모델/검색/정책/인덱스 파라미터)를 모아둔 파일
"""
import os
from pathlib import Path

# 프로젝트 루트 = img_guard/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# __file__ = 지금 실행되는 파일 경로, .../img_guard/app/config.py
# parents[0] = app/,   parents[1] = img_guard/

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
DB_IMAGES_DIR = DATA_DIR / "db_images" / "dataset60"  

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
HNSW_INDEX_PATH = DATA_DIR / "hnsw.index"

DB_MANIFEST_PATH = DATA_DIR / "db_manifest.json"
DB_SIGNATURE_MODE = "mtime_size"

# ANN backend
# - local: 로컬 HNSW 인덱스 사용 (현재 방식)
# - pgvector: PostgreSQL(pgvector) 검색 사용
ANN_BACKEND = os.getenv("ANN_BACKEND", "local").lower()

# pgvector config (service mode)
VECTOR_DSN = os.getenv("VECTOR_DSN", "")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "image_embeddings")
VECTOR_EMBED_COL = os.getenv("VECTOR_EMBED_COL", "embedding")
VECTOR_ID_COL = os.getenv("VECTOR_ID_COL", "id")
VECTOR_FILE_COL = os.getenv("VECTOR_FILE_COL", "file_name")
VECTOR_KEY_COL = os.getenv("VECTOR_KEY_COL", "s3_key")
VECTOR_URL_COL = os.getenv("VECTOR_URL_COL", "asset_url")
VECTOR_PHASH_COL = os.getenv("VECTOR_PHASH_COL", "")

# temp/cache dir for downloaded images
TMP_DIR = Path(os.getenv("IMG_GUARD_TMP_DIR", str(DATA_DIR / "cache" / "tmp")))

# 모델 설정 (CLIP)
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# 검색 설정
TOP_K = 10
TOP_PHASH = 10

# 정책(3상태 룰) threshold v1
COS_BLOCK = 0.97
PHASH_BLOCK = 10

COS_ALLOW_A = 0.90
PHASH_ALLOW_A = 20

COS_ALLOW_B = 0.85

# HNSW 파라미터 (나중에 튜닝 가능)
"""
M: 그래프에서 한 노드가 유지하는 링크 개수, 올리면 정확도 상승, 메모리 증가, 빌드시간 증가
ef_construction: 인덱스 만들 때 탐색 폭, 올리면 인덱스 품질 향상, 빌드시간 증가
ef_search: 검색할 때 탐색 폭, 올리면 recall 증가, latency 증가
"""
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50
