# AGENT Guidelines

This repository is intentionally minimal.

## Core rules

1. Keep a single prompt for all tasks in one selected folder.
2. Keep output schema stable so downstream scripts can parse it.
3. Prefer clear, short functions over abstraction-heavy design.
4. Fail gracefully: if LLM call fails, still write one output JSON with `unknown` statuses.

## Runtime contract

- Run command from the repository root: `python -m models.image_baseline_model.main <tasks_folder_name>`
- Input location: `data/<tasks_folder_name>/*.json`
- Output location: `data/<tasks_folder_name>/<prompt_index>.json`

## Model contract

- API key env var: `GEMMA_API_KEY`
- Optional model override: `GEMMA_MODEL`
- Response must be strict JSON with per-task predictions and brief explanation.
