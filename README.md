# FRSynth

Specialization of a small open model for French industrial-incident classification using only synthetic data.

The pipeline generates schema-valid labeled examples with a frontier LLM, filters them with a generator–critic loop, LoRA-trains a smaller model on the curated set, and measures the result against a hand-verified test set.

## Status

Work in progress. What exists today:

- **Schema & taxonomy** — Pydantic models for the reports, labels, and critic verdicts (`src/schema.py`); the 6-category × 3-severity label space and its definitions live in `data/attribute_grid.yaml`.
- **Generation** — a `Generator` that produces schema-valid reports seeded from an attribute grid (sector, register, length, noise, distractor flag), with each label's rubric injected for fidelity (`src/agents.py`, `src/generate_reports.py`).
- **Critic** — a blind `Critic` that classifies a report from its text alone (no label leakage), used to score and compare judges (`src/agents.py`, `src/compare_critics.py`).
- **Pipelines** — thin orchestrators over the step library; `critic_compare` generates a batch and compares a strong vs. normal critic, reporting agreement and dumping disagreements (`src/pipelines/`).
- **Training** — a first-shot LoRA SFT of `Qwen2.5-1.5B-Instruct` (`src/train.py`). Early / not yet wired into the pipelines.

Not yet built: curated train/val/test splits, the hand-verified test set, the evaluation campaign (F1, judge–human agreement, CIs), and the 4-bit quantization pass.

## Project layout

```
data/
  attribute_grid.yaml     # sampling axes + label definitions (taxonomy)
  raw.jsonl               # generated reports
src/
  schema.py               # Pydantic schemas + label space
  agents.py               # Generator, Critic
  generate_reports.py     # batch generation
  compare_critics.py      # strong vs. normal critic comparison
  train.py                # LoRA SFT (WIP)
  run_pipeline.py         # launcher
  pipelines/              # orchestrators (critic_compare, ...)
```

## Usage

Requires [`uv`](https://docs.astral.sh/uv/) and an `OPENROUTER_API_KEY` for generation/critic calls.

```bash
# generate a batch of labeled reports
uv run python src/generate_reports.py --n 1000 --out data/raw.jsonl

# compare the strong and normal critic on a set of reports
uv run python src/compare_critics.py --in data/raw.jsonl --n 150

# end-to-end: generate N reports then compare critics on them (-> data/local/critic_compare/)
uv run python src/run_pipeline.py critic_compare --n 100
```

Training dependencies (torch, transformers, peft, ...) are heavy and GPU-oriented; install them only where you train via `uv sync --group train`.

## Disclaimer

This is a pet project to get a better grasp of the full pipeline of training a small open model with synthetic data.
