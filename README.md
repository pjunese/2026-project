# 2026 Project

Image and document verification prototype for similarity search, watermarking,
OCR-assisted document processing, and backend integration.

## Repository Layout

- `img_guard/app/`: FastAPI app and core service modules
- `img_guard/app/document/`: document rendering, OCR, watermark workflow
- `img_guard/app/watermark/`: watermark service and WAM/mock backends
- `img_guard/scripts/`: setup and runtime helper scripts
- `img_guard/sql/`: pgvector bootstrap SQL

Runtime data, model weights, secrets, virtual environments, and local notes are
intentionally excluded from Git.

## Local Setup

Create a fresh virtual environment after cloning. Do not reuse or commit a
copied `.venv` directory.

```bash
cd img_guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.cpu.txt
```

Copy the environment template and fill only local values.

```bash
cp .env.example .env
```

## Run

```bash
cd img_guard
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Notes

- `WM_BACKEND=mock` is the default for contract-level local testing.
- Use `WM_BACKEND=wam` only after mounting the WAM repo and checkpoint files.
- Local data under `img_guard/data/` and model weights under
  `img_guard/models/` are not part of this repository.
