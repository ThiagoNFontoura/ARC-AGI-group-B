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

## Run

From the repository root:

```powershell
python -m models.data_augmentation_baseline.main <tasks_folder_name>
```

Choose a smaller set of views for a cheaper run:

```powershell
python -m models.data_augmentation_baseline.main <tasks_folder_name> --transforms identity transpose
```

Available transformations are `identity`, `flip_horizontal`, `flip_vertical`, and `transpose`.

The augmented runner writes to:

```text
data/data_augmentation_output/output.json
```

The baseline output is not overwritten. With the default four views, the run makes up to four model calls per task, so API cost and latency are also approximately multiplied by four.

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

## Design limitations

- Exact voting only recognizes byte-for-byte equivalent JSON grid structures.
- If every view fails, the task receives `unknown` status and the per-view errors are retained in the runner's error path.
- If the same mistake appears in all views, augmentation cannot correct it.
- The current implementation uses one call per transform and does not batch requests.
- Only geometric transformations are included. D8 rotations/reflections, color permutations, example-order permutations, hierarchical voting, and confidence scoring are possible future modules.
