"""Top-level launcher so pipelines run from the repo root with src/ on sys.path:

    uv run python src/run_pipeline.py <name> [args]
"""
from pipelines.__main__ import main

if __name__ == "__main__":
    main()
