from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_DIR = Path.home() / ".vibescholar"

DATA_DIR: Path = _DEFAULT_DATA_DIR
DB_PATH: Path = _DEFAULT_DATA_DIR / "index.sqlite3"
FAISS_INDEX_PATH: Path = _DEFAULT_DATA_DIR / "vectors.faiss"
MODEL_CACHE_DIR: Path = _DEFAULT_DATA_DIR / "model_cache"


def configure(data_dir: Path | None = None) -> None:
    """Set all paths from *data_dir*.  Call before importing other package modules."""
    global DATA_DIR, DB_PATH, FAISS_INDEX_PATH, MODEL_CACHE_DIR

    resolved = (data_dir or _DEFAULT_DATA_DIR).resolve()
    DATA_DIR = resolved
    DB_PATH = resolved / "index.sqlite3"
    FAISS_INDEX_PATH = resolved / "vectors.faiss"
    MODEL_CACHE_DIR = resolved / "model_cache"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_CACHE_DIR / "hf").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "hf"))
