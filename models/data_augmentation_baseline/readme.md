# ARC-AGI Baseline with Modular Test-Time Augmentation

This directory contains the original baseline behavior with a reusable geometric augmentation strategy added around it.

## Original baseline

The baseline runner:

1. Loads every `*.json` task from `data/<tasks_folder_name>/`.
2. Normalizes the task name, preserving the file name as a fallback.
3. Builds one prompt for each task using the baseline JSON contract.
4. Sends the prompt to `GemmaHandler`.
5. Extracts the matching task from the model response.
6. Compares `predicted_test_outputs` with available test labels.
7. Writes one normalized result file.

The model contract is provider-specific only at the handler boundary:

```text
prompt: str -> {"tasks": [{"task_name", "logic_explanation", "predicted_test_outputs"}]}
```

The baseline makes one model call per task and uses that response directly. It does not change the model weights, transform grids, generate alternative views, or vote between predictions.

The original runner is:

```powershell
python -m models.baseline_model.main <tasks_folder_name>
```

It writes to `data/baseline_output/output.json`.

## Current augmented runner

The augmented runner keeps the same input format, model response schema, correctness evaluation, and graceful error behavior. The difference is that each task is solved through several reversible geometric views:

```text
task
  -> transform every train/test grid
  -> build an ordinary baseline prompt
  -> call the injected model solver
  -> inverse-transform predicted outputs
  -> exact-vote normalized predictions
  -> keep the winning prediction
```

The default views are:

- `identity`
- `flip_horizontal`
- `flip_vertical`
- `transpose`

For every view, the transform is applied to both known train inputs/outputs and test inputs. The returned test outputs are then mapped back with the inverse transform. This is important: comparing outputs before inverse transformation would compare grids in different coordinate systems.

The winning output is selected by:

1. highest exact vote count;
2. identity-view preference when vote counts tie;
3. fixed transform order as the final deterministic tie-breaker.

This is an inference-time ensemble, not full Test-Time Training. It does not train LoRA adapters, perform leave-one-out training, use color permutations, or use model log-probabilities. Those extensions can be added behind the strategy boundary later.

## Configuration

Create or edit the `.env` file at the repository root to configure your Google Gemini credentials and model:

```env
GEMMA_API_KEY=your_api_key_here
GEMMA_MODEL=gemini-3.1-flash-lite
```

Recommended models:
- `gemini-3.1-flash-lite` (High daily quota: 500 RPD, 15 RPM)
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

## Run

From the repository root:

```powershell
python -m models.data_augmentation_baseline.main <tasks_folder_name>
```

### Example: Running both Baseline and Data Augmentation on `20_training`

To run both modes and generate the comparison report on `data/20_training`:

```powershell
python -m models.data_augmentation_baseline.main 20_training --run-both
```

Choose a smaller set of views for a cheaper/faster run:

```powershell
python -m models.data_augmentation_baseline.main 20_training --transforms identity transpose --run-both
```

The runner defaults to only running the **Augmented Phase (With Data Augmentation)** when `--run-both` is omitted.

When using the `--run-both` flag, it automatically executes two sequential phases:

1. **Baseline Phase (No Data Augmentation)**: Runs using only the `identity` transform.
2. **Augmented Phase (With Data Augmentation)**: Runs using the specified `--transforms` (or the default four views: `identity`, `flip_horizontal`, `flip_vertical`, `transpose`).

The results are saved into the root `output/` directory (created if it doesn't exist). For `20_training`, running with `--run-both` writes:

```text
output/20_training_baseline_output.json
output/20_training_augmented_output.json
output/20_training_comparison_report.json
```
(If `--run-both` is omitted, only the `20_training_augmented_output.json` file is produced).

Additionally, when `--run-both` finishes, a concise comparison scorecard is printed directly to the terminal showing:
- Accuracy Delta ($+X\%$) and Net Gained tasks.
- Average Pixel Accuracy Delta and Shape Match Rate Delta.
- List of tasks fixed by data augmentation (Baseline failed $\to$ Augmented succeeded).
- List of regressed tasks (Baseline succeeded $\to$ Augmented failed).
- Counts of `Both Correct` and `Both Incorrect`.

With `--run-both` and the default four views, the run makes a total of five model calls per task (1 baseline + 4 augmented), so API cost and latency are higher.

## Module layout

### `transforms.py`

Contains only ARC grid/task transformations:

- `TransformSpec`: name, forward function, and inverse function;
- `transform_task`: transforms every relevant grid in a task;
- `transform_outputs`: maps model outputs back to the original orientation;
- `get_transformations`: selects registered transformations by name.

This module has no dependency on Gemma, prompts, or the CLI. It can be copied into another model or imported by another solver.

### `strategy.py`

Contains `AugmentedInference`, which owns the augmentation workflow and voting. It receives two callables:

```python
AugmentedInference(
    solve_prompt=model_callable,
    build_prompt=prompt_builder,
    transformations=selected_transforms,
)
```

`solve_prompt` can be any function with the baseline signature `str -> dict`. Therefore, another model only needs an adapter around its own API; the augmentation code does not need to know which model is being used.

### `main.py`

Contains task loading, CLI arguments, model construction, result evaluation, and output writing. It is deliberately not responsible for geometric operations or vote logic.

### `llm_handler.py` and `prompt.py`

These preserve the local baseline-compatible model and prompt contracts. The augmentation layer uses them through dependency injection rather than importing a model inside `strategy.py`.

## Reusing the augmentation with another model

A different solver can reuse the strategy as follows:

```python
from models.data_augmentation_baseline.strategy import AugmentedInference
from models.data_augmentation_baseline.transforms import get_transformations

augmented_solver = AugmentedInference(
    solve_prompt=other_model.solve,
    build_prompt=other_model.build_prompt,
    transformations=get_transformations(["identity", "transpose"]),
)

result = augmented_solver.solve_task(task, prompt_index=1)
```

The replacement model must return a dictionary containing a `tasks` list. Each task should contain `task_name` and `predicted_test_outputs`, matching the baseline response contract.

## Execution Output & Ensemble Telemetry

The output JSON file includes top-level run statistics as well as granular ensemble, telemetry, and ARC-specific metrics for every evaluated task.

### Top-Level `summary`
Aggregates overall task accuracy, sub-pixel correctness, shape matching, latency, token usage, and per-transformation effectiveness:

```json
"summary": {
  "total_tasks": 10,
  "correct_tasks": 7,
  "incorrect_tasks": 3,
  "unknown_tasks": 0,
  "accuracy_percentage": 70.0,
  "average_pixel_accuracy_percentage": 89.45,
  "shape_match_percentage": 90.0,
  "color_preservation_percentage": 100.0,
  "total_duration_seconds": 24.8,
  "token_usage": {
    "prompt_tokens": 12450,
    "candidates_tokens": 3120,
    "total_tokens": 15570
  },
  "consensus_distribution": {
    "unanimous": 4,
    "majority": 2,
    "plurality": 1
  },
  "transforms_performance": {
    "identity": {
      "attempted": 10,
      "correct": 6,
      "incorrect": 4,
      "accuracy_percentage": 60.0,
      "average_pixel_accuracy_percentage": 82.1,
      "times_voted_winner": 7
    },
    "transpose": {
      "attempted": 10,
      "correct": 7,
      "incorrect": 3,
      "accuracy_percentage": 70.0,
      "average_pixel_accuracy_percentage": 89.5,
      "times_voted_winner": 8
    }
  }
}
```

### Per-Task Metrics & `arc_metrics`
Each task entry includes:
- **`arc_metrics`**:
  - `shape_match`: Boolean indicating whether predicted grid dimensions match ground truth for all test cases.
  - `pixel_accuracy_percentage`: Percentage of cells matching ground truth across all test cases.
  - `test_case_details`: Per-test-case breakdown with shapes, total pixels, matching pixels, and exact match flags.
  - `color_evaluation`: Palette validation comparing predicted colors against expected ground truth and context colors (flags hallucinated/unseen colors).
- **`ensemble_details`**:
  - **`summary`**: Consensus type, vote counts/percentages, winning transforms, and unique candidate count.
  - **`telemetry`**: Total task latency in seconds and prompt/candidate token consumption.
  - **`voting_distribution`**: Ranked distribution of unique predictions, their vote counts/percentages, and which transforms voted for each.
  - **`individual_views`**: The full log of each transform view, including its latency, token usage, converted output, raw model output, explanation, and individual `arc_metrics`.

## Design limitations

- Exact voting only recognizes byte-for-byte equivalent JSON grid structures.
- If every view fails, the task receives `unknown` status and the per-view errors are retained in the runner's error path.
- If the same mistake appears in all views, augmentation cannot correct it.
- The current implementation uses one call per transform and does not batch requests.
- Only geometric transformations are included. D8 rotations/reflections, color permutations, example-order permutations, hierarchical voting, and confidence scoring are possible future modules.
