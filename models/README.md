# ARC-AGI Gemma Solver (Single Prompt)

Minimal end-to-end Python application that solves ARC-AGI tasks with one shared prompt for all tasks inside a selected folder.

## Architecture

- `data/` - contains task folders with ARC JSON files
- `.env` - contains `GEMMA_API_KEY`
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
# GEMMA_MODEL=gemma-3-27b-it
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
python main.py <tasks_folder_name>
```

Example:

```bash
python main.py set_a
```

## Output

A single JSON file is written inside the same selected folder:

- `1.json`, `2.json`, `3.json`, ... (sequential prompt index)

Each output file includes:
- prompt index
- all task names
- per-task brief logic explanation
- per-task correctness status (`correct`, `incorrect`, or `unknown`)
- predicted test outputs
