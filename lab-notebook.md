08-14-2026
- Group-B's GitHub has been created
- First solver's architecture created using Copilot
- First step: we chose to compare the gemma's performance on tasks using images and only text


08-16-2026
- Baseline model with gemma was continued
- First image generator created

08-17-2026
- Baseline model with gemma compleated
- Output Generated

08-26-2026
- Briefly studied how example generation works and planned an example generator with TTT
- Implemented the initial version of example-gen, but it was not extensively tested; testing was limited to two tasks
- Switched to Gemini 3.7 Flash, corrected the grid handling, and preserved source dimensions.

08-27-2026
- Updated the project setup and execution instructions in HOWTORUN.md, including example-gen
- Updated the example-gen output directory to output/example-gen

08-28-2026
- 

09-03-2026
- Configured exemplar synthesis to use Gemini 3.7 Flash with high thinking and no automatic retries by default.
- Updated generation to use all available training examples and analyze input/output invariants independently.
- Added rule-based validation for original and generated examples, with per-example validity flags and automatic rejection when over 20% of generated examples fail.
- Updated both solvers to ignore invalid examples and added per-run generation reports with validation summaries and errors.
