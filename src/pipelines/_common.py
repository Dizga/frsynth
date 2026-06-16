"""Shared scaffolding for pipelines: paths, seeding, per-run output dirs."""
import random
from pathlib import Path

# Resolve paths from the file location, not the cwd, so pipeline outputs land in
# the same place regardless of where the launcher is invoked. (src/pipelines/_common.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / "data" / "local"


def prepare_run(name: str, seed: int = 0) -> Path:
    """Seed the RNG and return a fresh output dir for a pipeline run: data/local/<name>/."""
    random.seed(seed)
    out_dir = LOCAL_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
