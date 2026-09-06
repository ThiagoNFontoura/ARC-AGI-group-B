# ARC-AGI Research Framework (Group B)

An exploratory framework for solving the **Abstraction and Reasoning Corpus (ARC-AGI)** using Large Language Models (LLMs) and Vision-Language Models (VLMs). This repository investigates multiple reasoning strategies including text-based prompting, multimodal grid rasterization, test-time geometric augmentation (TTA), and few-shot exemplar synthesis.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Components & Methodologies](#key-components--methodologies)
  - [1. Textual Baseline Solver](#1-textual-baseline-solver)
  - [2. Multimodal Image Solver](#2-multimodal-image-solver)
  - [3. Test-Time Data Augmentation (TTA)](#3-test-time-data-augmentation-tta)
  - [4. Exemplar Synthesis Pipeline](#4-exemplar-synthesis-pipeline)
- [Pipeline Architectures](#-pipeline-architectures)
  - [1. Data Augmentation Pipeline (Test-Time Augmentation)](#1-data-augmentation-pipeline-test-time-augmentation)
  - [2. Exemplar Synthesis Pipeline (Synthetic Data Generation)](#2-exemplar-synthesis-pipeline-synthetic-data-generation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Environment Setup](#installation--environment-setup)
- [Usage Guide](#usage-guide)
  - [Running the Text Baseline](#running-the-text-baseline)
  - [Running the Image Multimodal Baseline](#running-the-image-multimodal-baseline)
  - [Running Geometric Augmentation & Comparison](#running-geometric-augmentation--comparison)
  - [Running Exemplar Synthesis](#running-exemplar-synthesis)
- [Evaluation & Telemetry](#evaluation--telemetry)
- [Project Context & Research Notes](#project-context--research-notes)

---

## 🔍 Overview

The **ARC-AGI** benchmark tests inductive reasoning and general intelligence through visual/spatial grid transformations. This project (PCI - Group B) evaluates:

1. **Text vs. Vision modalities**: Comparing model performance on raw 2D JSON matrix inputs versus rendered color images.
2. **Inference-Time Ensembling**: Testing whether reversible geometric transformations (rotations, reflections, transpositions) improve prediction stability and accuracy.
3. **Exemplar Synthesis (Data Augmentation)**: Generating verified synthetic demonstration pairs using a stronger reasoning model (`gemini-3.7-flash` / `gemini-3.5-flash-lite`) to enrich the context for downstream solvers.

---

## 🚀 Key Components & Methodologies

### 1. Textual Baseline Solver
*Path: [`models/baseline_model/`](models/baseline_model/)*

- Directly consumes ARC task JSONs containing `train` demonstrations and `test` inputs.
- Employs structured system prompts to solicit both task logic explanations and predicted test grids.
- Supported backends: Google GenAI API (`gemma-4-31b-it`, `gemini-2.5-flash`, etc.).
- Evaluates exact grid equality against known test labels.

### 2. Multimodal Image Solver
*Path: [`models/image_baseline_model/`](models/image_baseline_model/)*

- Rasterizes ARC numeric grids into high-contrast color PNG images with configurable cell sizes, grid lines, and palette mappings (`render_settings.py`).
- Supports parallel batch rendering of task inputs and outputs.
- Solves tasks through visual prompts sent to multimodal LLMs, comparing image-driven spatial reasoning against textual representations.

### 3. Test-Time Data Augmentation (TTA)
*Path: [`models/data_augmentation_baseline/`](models/data_augmentation_baseline/)*

- Augments ARC tasks at inference time across reversible geometric views:
  - `identity`
  - `flip_horizontal`
  - `flip_vertical`
  - `transpose`
- For each view:
  1. Forward-transforms train and test grids.
  2. Queries the solver.
  3. Inversely transforms the predicted test grids back to the canonical orientation.
  4. Conducts exact-match consensus voting to pick the winning prediction.
- Includes a `--run-both` comparison mode that executes both baseline (identity) and augmented phases, outputting comparative scorecards (accuracy deltas, sub-pixel accuracy, shape match rates, net tasks gained/lost).

### 4. Exemplar Synthesis Pipeline
*Path: [`models/example_gen/`](models/example_gen/)*

- Synthesizes additional supervised training examples for ARC tasks before solver execution without fine-tuning weights.
- **Invariant Analysis** (`invariants.py`): Separately analyzes dimensions, color counts, estimated background color, connected components (4-way), and symmetries across original grids.
- **Rule & Program Generation**: Prompts a reasoning LLM to infer the transformation rule, generate a Python `transform(grid)` function, and produce $N$ synthetic input/output training pairs.
- **Restricted Sandbox Execution**: Dynamically compiles and executes the generated Python function against both original and synthetic examples.
- **Strict Quality Gating**:
  - Rejects synthetic sets violating identified invariants.
  - If $>20\%$ of synthetic pairs fail functional validation, the entire synthetic set is rejected (`valid: false`).
- Generates enriched `<task_id>-plus.json` files ready for downstream solver consumption.

---

## 🏗️ Pipeline Architectures

### 1. Data Augmentation Pipeline (Test-Time Augmentation)
*Reference documentation: [`simple_geometric_augmentation_strategy.md`](models/data_augmentation_baseline/simple_geometric_augmentation_strategy.md)*

Language models serialize 2D grids line-by-line, creating directional reading biases that can obscure certain spatial rules. The data augmentation pipeline mitigates this by presenting the exact same task under multiple reversible geometric orientations, solving each view, mapping the candidate predictions back to the original orientation, and taking a consensus vote.

```
                          +-------------------------+
                          |   Original Task (ARC)   |
                          |  Train & Test Grids (C) |
                          +------------+------------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
  +--------------+              +--------------+              +--------------+
  |  Identity T0 |              | Flip Horiz T1|              |  Transpose T3| ... (Flip Vert T2)
  +-------+------+              +-------+------+              +-------+------+
          |                             |                             |
    Transform Task                Transform Task                Transform Task
     (All Grids)                   (All Grids)                   (All Grids)
          |                             |                             |
          v                             v                             v
  +--------------+              +--------------+              +--------------+
  |  LLM Solver  |              |  LLM Solver  |              |  LLM Solver  |
  | (Greedy Dec) |              | (Greedy Dec) |              | (Greedy Dec) |
  +-------+------+              +-------+------+              +-------+------+
          |                             |                             |
    Output y_aug_0                Output y_aug_1                Output y_aug_3
          |                             |                             |
  +-------v------+              +-------v------+              +-------v------+
  | Apply T0^-1  |              | Apply T1^-1  |              | Apply T3^-1  |
  +-------+------+              +-------+------+              +-------+------+
          |                             |                             |
   Normalized y_0                Normalized y_1                Normalized y_3
          |                             |                             |
          +-----------------------------+-----------------------------+
                                        |
                                        v
                       +---------------------------------+
                       | Exact-Match Majority Voting     |
                       |  1. Highest exact vote count    |
                       |  2. Tie-break: Identity view    |
                       |  3. Tie-break: Transform order  |
                       +----------------+----------------+
                                        |
                                        v
                       +---------------------------------+
                       | Final Winning Prediction (ŷ)    |
                       | + Granular Telemetry Scorecard  |
                       |   (Pixel/Shape/Palette Metrics) |
                       +---------------------------------+
```

#### Pipeline Steps:

1. **Consistent Full-Task Transformation**:
   The transformation $T$ is applied consistently to **every** grid in the task: all training inputs, all training outputs, and all test inputs. Transforming only test inputs would corrupt the spatial relationships learned from demonstrations.
   - Standard 4-view set: `identity`, `flip_horizontal`, `flip_vertical`, and `transpose` (all are involutions: $T^{-1} = T$).
   - Can optionally extend to the full 8 symmetries of the dihedral group $D_8$.
2. **Independent Greedy Inference**:
   The LLM generates a greedy prediction for each transformed view independently.
3. **Inverse Transformation ($T^{-1}$)**:
   Every predicted output $y_{\text{aug}}$ is brought back into the original canonical orientation: $\hat{y} = T^{-1}(y_{\text{aug}})$.
4. **Exact-Match Consensus Voting**:
   Normalized grid predictions are grouped by byte-for-byte exact equality. The winner is determined by:
   - Highest frequency of exact match;
   - Preference for the `identity` view in the event of ties;
   - Fixed transform order as deterministic final tie-breaker.
5. **Comparative Telemetry (`--run-both`)**:
   Compares the baseline (identity only) against the ensemble across overall accuracy, pixel accuracy, shape match rate, and per-transform winning rates.

---

### 2. Exemplar Synthesis Pipeline (Synthetic Data Generation)
*Reference documentation: [`abstract_example_generation.md`](models/example_gen/abstract_example_generation.md)*

Rather than fine-tuning model parameters, Exemplar Synthesis uses a high-capacity reasoning model (`gemini-3.7-flash` with High Thinking) to infer the inductive rule behind an ARC task, formulate an executable Python `transform(grid)` function, and generate additional verified input/output training pairs to enrich the demonstration context for downstream solvers.

```
                          +-------------------------+
                          |   Original Task (ARC)   |
                          | Train: input + output   |
                          | Test: input ONLY        |
                          +------------+------------+
                                       |
                                       v
                     +-----------------------------------+
                     |    Invariant Feature Extractor    |
                     |  - Dimensions (H, W constant?)    |
                     |  - Color sets & per-color counts  |
                     |  - Background color detection     |
                     |  - Connected components (4-way)   |
                     |  - Horizontal & vertical symmetry |
                     +-----------------+-----------------+
                                       |
                   Prompt with Demonstrations + Invariants
                                       |
                                       v
                     +-----------------------------------+
                     |     Reasoning LLM Synthesis      |
                     | (Gemini 3.7 Flash, High Thinking) |
                     | Produces structured JSON:         |
                     |  - logic_explanation (r)          |
                     |  - transformation_function (Py)   |
                     |  - generated_train (N pairs)      |
                     |  - predicted_test_outputs         |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Restricted Sandbox Verification  |
                     |   Executes transform(grid) on:    |
                     |   - Original training examples    |
                     |   - Generated synthetic pairs     |
                     |   Checks: transform(inp) == out   |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     Multi-Tier Quality Gating     |
                     |  1. Invariant consistency check   |
                     |     (Any violation -> reject all) |
                     |  2. Failure rate check:           |
                     |     - If >20% fail -> reject all  |
                     |     - If <=20% fail -> keep valid |
                     +-----------------+-----------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
             [Pass Validation]                   [Fail Validation]
                     |                                   |
                     v                                   v
          +---------------------+             +---------------------+
          |  Write -plus.json   |             |  Flag valid: false  |
          |  C+ = C ∪ G_valid   |             |  Audit in report    |
          +----------+----------+             +----------+----------+
                     |                                   |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Downstream Solvers (Text/Vision)  |
                     | (Excludes pairs with valid: false)|
                     +-----------------------------------+
```

#### Pipeline Steps:

1. **Context Construction & Information Containment**:
   All original training demonstrations $(C)$ are provided to give complete task context. Test outputs are strictly omitted ($X_{\text{test}}$ only) to prevent leakage during synthesis.
2. **Invariant Feature Extraction (`invariants.py`)**:
   Analyzes train inputs and outputs separately for:
   - Fixed heights and widths;
   - Color palettes and exact counts per color;
   - Dominant background color;
   - Number of 4-connected components;
   - Exact horizontal and vertical symmetries.
   Properties found to be strictly constant are injected into the prompt as hard generation constraints.
3. **Single-Call Structured Generation**:
   The reasoning model is queried in a single call to generate:
   - `logic_explanation`: Clear deduction of the grid transformation rule.
   - `transformation_function`: Complete executable Python function `def transform(grid): ...`.
   - `generated_train`: Exactly $N$ new synthetic input/output pairs ($G$).
   - `predicted_test_outputs`: Model prediction for the withheld test input.
4. **Sandboxed Program Execution**:
   The Python function is compiled and executed in a restricted sandbox (no network/file/import access). The code runs `transform(inp)` on every original pair and every generated pair, requiring exact grid equality `transform(inp) == out`.
5. **Multi-Tier Quality Gating**:
   - *Invariant Gate*: If any generated pair violates a known constant property (e.g. dimensions, allowed colors), all generated examples are flagged `valid: false`.
   - *20% Error Margin Gate*: If $>20\%$ of generated pairs fail functional execution, the entire synthetic batch is invalidated (`valid: false`). If $\le 20\%$ fail, only individual invalid examples are dropped.
6. **Dataset Augmentation & Audit Reporting**:
   - Saves augmented dataset as `output/example-gen/<task_id>-plus.json` containing $C^+ = C \cup G_{\text{valid}}$.
   - Saves `output/example-gen/relatorio[ID].json` logging token usage, rule explanations, validation statuses, and error traces.
   - Solvers automatically skip any training pairs flagged with `valid: false`.

---

## 📂 Repository Structure

```text
ARC-AGI-group-B/
├── .env                                # API keys and model configurations (gitignored)
├── HOWTORUN.md                         # Detailed step-by-step Portuguese execution guide
├── README.md                           # Project documentation
├── lab-notebook.md                     # Chronological development and experiment log
├── requirements.txt                    # Project dependencies
├── example-generation-pipeline.jpg     # Pipeline diagram
│
├── data/                               # ARC task datasets
│   ├── 10_evaluation/                  # Evaluation sample sets
│   ├── 20_training/                    # Training sample sets
│   ├── evaluation/                     # ARC evaluation split
│   └── training/                       # ARC training split
│
├── models/
│   ├── baseline_model/                 # Text-based single-prompt solver
│   ├── image_baseline_model/           # Image rendering + multimodal vision solver
│   ├── data_augmentation_baseline/     # Test-time geometric augmentation & ensembling
│   └── example_gen/                    # Exemplar synthesis, invariant extractor & validator
│
├── output/                             # Generated predictions, comparison reports & -plus.json
└── presentations/                      # Project presentation slides (Quarto Reveal.js)
```

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini / AI Studio API key

### Installation & Environment Setup

1. **Clone the repository and set up a virtual environment:**

   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**

   Create a `.env` file in the root directory:

   ```env
   GEMMA_API_KEY=your_google_ai_studio_api_key_here

   # Optional model configuration:
   GEMMA_MODEL=gemma-4-31b-it
   GEMMA_VALIDATOR_MODEL=gemma-4-31b-it
   EXAMPLE_GEN_MODEL=gemini-3.7-flash
   GEMMA_THINKING_LEVEL=high
   ```

---

## 💻 Usage Guide

### Running the Text Baseline

Executes the model on all JSON tasks within a specified folder in `data/`:

```powershell
python -m models.baseline_model.main 20_training
```
Output results are written to `output/<folder>_baseline_output.json`.

### Running the Image Multimodal Baseline

Render tasks to visual images and evaluate:

```powershell
# Only render images to data/<folder>/images/ without calling the LLM:
python -m models.image_baseline_model.main 20_training --render-only

# Render images and run multimodal solver:
python -m models.image_baseline_model.main 20_training --render-images

# Run a single task with custom workers:
python -m models.image_baseline_model.main training --task-file 007bbfb7.json --no-strong-validate
```

### Running Geometric Augmentation & Comparison

Run test-time geometric ensembling, or compare directly against the unaugmented baseline:

```powershell
# Run augmentation only (default: identity, flip_h, flip_v, transpose):
python -m models.data_augmentation_baseline.main 20_training

# Run both Baseline and Augmented phases to generate comparison scorecards:
python -m models.data_augmentation_baseline.main 20_training --run-both

# Run with specific transformations:
python -m models.data_augmentation_baseline.main 20_training --transforms identity transpose --run-both
```

### Running Exemplar Synthesis

Generate new synthetic training demonstrations for tasks in a folder or for a single task:

```powershell
# Generate extra examples for an entire task folder:
python -m models.example_gen.main data/20_training

# Generate extra examples for a single task:
python -m models.example_gen.main data/20_training/3aa6fb7a.json

# Override the number of synthetic examples requested:
python -m models.example_gen.main data/20_training --generated-examples 5
```

Enriched tasks are written to `output/example-gen/<task_name>-plus.json` alongside detailed execution and validation reports (`output/example-gen/relatorio[ID].json`).

---

## 📊 Evaluation & Telemetry

The runners collect rich telemetry across ARC-specific criteria:

- **Exact Match Correctness**: Strict equality between predicted grid and ground-truth output.
- **Pixel Accuracy**: Sub-pixel match percentage across test cases.
- **Shape Match Rate**: Validation that output dimensions $(H \times W)$ match expected targets.
- **Color Preservation & Palettes**: Detection of unseen or hallucinated color values.
- **Ensemble Telemetry**: Vote distributions (unanimous, majority, plurality), per-view latency, and token consumption.

---

## 📖 Project Context & Research Notes

- **Mini-ARC Baseline Reference**: Early benchmarks evaluated a minimal transformer baseline (e.g., 4.5% with TTT on 114 ARC tasks) as a reference comparison against published literature (such as the Mini-ARC study).
- For an experiment log and progress history, consult [`lab-notebook.md`](lab-notebook.md).
- For slide decks and presentation materials, see [`presentations/slides_10-03/`](presentations/slides_10-03/).
- For detailed execution notes in Portuguese, see [`HOWTORUN.md`](HOWTORUN.md).
