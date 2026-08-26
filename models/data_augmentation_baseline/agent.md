# Agent Guide: Data Augmentation Baseline

## Purpose

This package is the modular augmented version of `models/baseline_model`. It performs geometric test-time augmentation around a baseline-compatible LLM interface.

## Runtime contract

Run from the repository root:

```powershell
python -m models.data_augmentation_baseline.main <tasks_folder_name>
```

Optional view selection:

```powershell
python -m models.data_augmentation_baseline.main <tasks_folder_name> --transforms identity flip_horizontal
```

Inputs are read from `data/<tasks_folder_name>/*.json`.

Results are written to `data/data_augmentation_output/output.json`. Do not change this to `baseline_output` unless overwriting the baseline is explicitly required.

## Architecture rules

- Keep model/provider code out of `transforms.py` and `strategy.py`.
- Add new geometric operations to `TRANSFORM_REGISTRY` in `transforms.py` as `TransformSpec` values.
- Every nontrivial transform must provide a correct inverse.
- Transform train inputs, train outputs, and test inputs consistently.
- Inverse-transform predicted test outputs before voting.
- Keep the baseline JSON response schema stable.
- Prefer dependency injection through `AugmentedInference.solve_prompt` and `build_prompt`.
- Do not make `strategy.py` import `GemmaHandler`.
- Keep CLI and filesystem concerns in `main.py`.

## Model adapter contract

`AugmentedInference` expects:

```python
solve_prompt(prompt: str) -> dict
build_prompt(tasks: list[dict], prompt_index: int) -> str
```

The solver response must contain:

```json
{
  "tasks": [
    {
      "task_name": "string",
      "logic_explanation": "string",
      "predicted_test_outputs": [[[0]]]
    }
  ]
}
```

A different model can be used by adapting its request/response format to this contract. Do not couple the augmentation implementation to a new provider.

## Change workflow

1. Read the target module and preserve existing user changes.
2. Add or update transformation unit checks, especially for rectangular grids.
3. Verify `T^-1(T(grid)) == grid` for every transform.
4. Verify that all train/test grids are transformed, including known train outputs.
5. Verify that normalized equivalent predictions receive the same vote.
6. Run syntax compilation:

```powershell
python -m compileall -q models/data_augmentation_baseline
```

7. Run the API-free strategy checks before using a real API key.
8. Run a real task only when credentials and dependencies are configured.

## Extension points

### Add rotations or D8

Implement the operation and inverse in `transforms.py`, register a named `TransformSpec`, and select it through `--transforms`. Keep the default list conservative unless cost changes are documented.

### Add multiple orderings

Create a separate task-level augmentation module rather than mixing prompt-order changes into grid transforms. The current `TransformSpec` contract is for grid transformations.

### Add confidence scoring

Extend candidate metadata and ranking in `strategy.py`. Preserve exact voting as the default fallback when log-probabilities are unavailable.

### Add TTT

Implement task-specific training as a separate injected adapter or strategy. Do not put training state in the pure transformation functions.

## Common failure modes

- **Wrong orientation after voting**: the inverse was not applied to model outputs.
- **Broken rectangular grids**: an implementation assumed height equals width.
- **Train/output mismatch**: only inputs were transformed.
- **Unexpected output overwrite**: output path was changed to `data/baseline_output`.
- **Provider coupling**: the reusable strategy imports or constructs a specific LLM client.
- **Unstable ties**: ranking does not use the documented identity and registry-order tie-breakers.
