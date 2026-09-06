# ARC-AGI Gemma Solver (Single Prompt)

Minimal end-to-end Python application that solves ARC-AGI tasks with one shared prompt for all tasks inside a selected folder.

## Architecture

- `../../data/` - repository-level task folders with ARC JSON files
- `../../.env` - repository-level file containing `GEMMA_API_KEY`
- `main.py` - CLI entrypoint and orchestration
- `llm_handler.py` - Gemma API client and JSON parsing
- `prompt.py` - single prompt builder for all tasks in a folder

## Requirements

- Python 3.10+
- Google AI Studio API key for Gemma-compatible model access

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your key in `.env`:

```env
GEMMA_API_KEY=your_api_key_here
# Optional:
# GEMMA_MODEL=gemma-4-31b-it
```

## Data format

Place tasks in `data/<tasks_folder_name>/` as JSON files using standard ARC shape:

```json
{
  "train": [
    {"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}
  ],
  "test": [
    {"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}
  ]
}
```

Notes:
- `task_name` is optional. If missing, filename (without extension) is used.
- If a test case has no `output`, correctness is reported as `unknown`.

## Run

```bash
python -m models.baseline_model.main <tasks_folder_name>
```

Example:

```bash
python -m models.baseline_model.main set_a
```

Run these commands from the repository root. The application locates `data/`
and `.env` relative to its own source file, so those paths remain stable even
if the command is launched from a different working directory.

## Output

A single JSON file is written to `output/`:

- `<tasks_folder_name>_baseline_output.json`

Each output file includes:
- prompt index
- all task names
- total token usage, total processing time, and accuracy percentage in `summary`
- per-task brief logic explanation
- per-task correctness status (`correct`, `incorrect`, or `unknown`)
- predicted test outputs

Each task is attempted once initially, with up to 2 additional retries if the model call or response parsing fails.
