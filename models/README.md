# ARC-AGI Gemma Solver (Single Prompt)

Minimal end-to-end Python application that solves ARC-AGI tasks with one shared prompt for all tasks inside a selected folder.

## Architecture

- `data/` - contains task folders with ARC JSON files
- `.env` - contains `GEMMA_API_KEY`
- `main.py` - CLI entrypoint and orchestration
- `llm_handler.py` - Gemma API client and JSON parsing
- `prompt.py` - single prompt builder for all tasks in a folder
- `task_image_renderer.py` - parallel ARC task-to-image renderer
- `render_settings.py` - easy style configuration for colors and grid

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
python models/main.py <tasks_folder_name>
```

Example:

```bash
python models/main.py set_a
```

Generate only task images (parallel):

```bash
python models/main.py set_a --render-only
```

Generate task images and then run solver:

```bash
python models/main.py set_a --render-images
```

Optional worker count:

```bash
python models/main.py set_a --render-only --render-workers 8
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

## Task Image Rendering

When rendering is enabled:
- Images are saved under `data/<tasks_folder_name>/images/<task_name>/`.
- Each train/test case becomes `*_input.png` and (if present) `*_output.png`.
- `0` values use background color.
- Non-zero values use distinct colors.
- Grid lines are drawn between squares.

Default style:
- Background: white
- Grid lines: black
- Zero value color: same as background

To quickly test different visual strategies, edit values in `render_settings.py`:
- `BACKGROUND_COLOR`
- `GRID_COLOR`
- `ZERO_COLOR`
- `COLOR_BY_VALUE`
- `CELL_SIZE`
- `GRID_LINE_WIDTH`
