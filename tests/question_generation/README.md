# Question Generation

This directory contains scripts and data assets for generating pharmaceutical industry benchmark test questions. The pipeline takes hand-curated reference questions, generates natural language variations, then substitutes chemical placeholders with real data to produce a diverse test dataset.

## Workflow Overview

```
reference_test_questions.csv (manually curated)
        │
        ├──→ [generate_test_question_variations.ipynb] → variations CSV
        │                                                       │
        └────────────────────────────────────────────────────→ [generate_test_question.py] → TOML Test Dataset
                                                                        ↑
                                               molecule_names.csv, smiles.csv, reactions.csv, property_constraints.csv
```

### Step 1: Generate Question Variations (one-time)

Open `generate_test_question_variations.ipynb` and run all cells. This reads `reference_test_questions.csv` (the manually curated base questions) and calls Gemini 2.5 Pro (via Vertex AI) to produce 4 paraphrased variations of each question at different formality levels (formal → informal). Output is a CSV with 5 rows per reference question (the original + 4 variations).

Requires: Google Cloud credentials and a configured Vertex AI project.

### Step 2: Generate Test Questions

Run `generate_test_question.py` to substitute chemical placeholders in the variation templates with randomly sampled values from the data CSVs.

```bash
python generate_test_question.py \
  --references reference_test_questions.csv \
  --molecules molecule_names.csv \
  --smiles smiles.csv \
  --reactions reactions.csv \
  --property-constraints property_constraints.csv \
  --num-samples 5 \
  --n-question-variations 4 \
  --seed 42 \
  --output test_questions.toml
```

Key flags:
- `--num-samples` — number of randomized samples to generate per question template
- `--n-question-variations` — how many variation levels (1–4) to include
- `--seed` — random seed for reproducibility
- `--output` — output TOML file path

Output is a TOML file organized by `task_type` and `task_label`, where each entry contains the question text and the expected tool call sequences for evaluation.

---

## File Descriptions

| File | Description |
|------|-------------|
| `generate_test_question.py` | CLI script that substitutes `{{molecule_name}}`, `{{smiles}}`, `{{reaction}}`, and `{{property_constraint}}` placeholders in question templates with random samples from the data CSVs. Produces the final TOML benchmark dataset. |
| `generate_test_question_variations.ipynb` | Jupyter notebook that reads `reference_test_questions.csv` and uses Gemini 2.5 Pro to paraphrase each reference question into 4 variations at different formality levels. Outputs a CSV with reference + variation rows consumed by the generation script. |
| `reference_test_questions.csv` | Manually curated CSV of base questions. Contains one row per reference question with columns: `reference_question`, `variation_label`, `variation_question`, `task_type`, `task_label`, `tool_calls_complete`, `tool_calls_accepted`, `note`. Serves as the starting point for both the variation notebook and `generate_test_question.py` (via `--references`). |
| `molecule_names.csv` | List of pharmaceutical molecule names (e.g., Ibuprofen, Aspirin, Gleevec) used to fill `{{molecule_name}}` placeholders. |
| `smiles.csv` | SMILES strings for pharmaceutically relevant compounds used to fill `{{smiles}}` placeholders. |
| `reactions.csv` | Reaction SMILES entries (reactant>>product) covering common synthetic transformations used to fill `{{reaction}}` placeholders. |
| `property_constraints.csv` | Molecular property constraint phrases (e.g., "higher molecular weight", "lower HBD") used to fill `{{property_constraint}}` placeholders in drug design questions. |
