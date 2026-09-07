# langdmta_eval

Evaluation framework for the LangDMTA multi-agent drug discovery system. Provides an LLM-as-a-Judge pipeline, human annotation tooling, alignment analysis, and automated judge optimization for LangDMTA.

## What It Is

An evaluation framework for **LangDMTA** -- a multi-agent AI system for drug discovery (He et al., 2026). The framework provides an **LLM-as-a-Judge** pipeline to automatically assess agent outputs, validates it against **human expert annotations**, and iteratively optimizes the judge using **DSPy** to maximize human-LLM alignment.

## System Context

LangDMTA is a multi-agent architecture for democratising real-world drug discovery:

- **Design Agent** -- Generates novel compounds and scores them against property targets
- **Synthesis Agent** -- Plans chemical synthesis routes
- **Analyzer Agent** -- Analyzes CSV output files for statistics and plots
- **Utility Agent** -- Converts compound names to chemical structures (SMILES)

---

## Evaluation Pipeline

The full pipeline runs in six steps:

### Step 1: Question Generation

See **[`langdmta-lab/tests/question_generation/README.md`](../../tests/question_generation/README.md)** for full details. The two-step process:

1. Run `generate_test_question_variations.ipynb` to paraphrase reference questions into variation levels using Gemini 2.5 Pro (one-time, requires Google Cloud credentials).
2. Run `generate_test_question.py` to substitute `{{placeholders}}` with chemical entities sampled from the data CSVs, producing the final TOML benchmark dataset.

```bash
python tests/question_generation/generate_test_question.py \
  --references reference_test_questions.csv \
  --num-samples 5 \
  --n-question-variations 4 \
  --seed 42 \
  --output test_questions.toml
```

Each generated question includes expected tool-call sequences for automated correctness checking.

### Step 2: Run LangDMTA and Summarize

Run the LangDMTA agents on the generated test questions, then extract the results into a summary CSV:

```bash
# Run agents on test questions
python tests/test_questions.py

# Extract Langfuse traces into a summary CSV
python evaluation/langfuse_summary.py
```

The summary CSV is the input to the LLM judge.

### Step 3: Run the LLM Judge

Score agent outputs across evaluation metrics using the CLI:

```bash
python -m langdmta_eval.run_judge results/chatinvent_results/test_questions_trial1_summary.csv \ # replace path with your summary CSV file
    --model azure/gpt-5 \
    --provider azure \
    --judge-type capability \
    --force-rerun \
    --save-path results/judge_results/evaluation_results.json # replace with appropriate path
```

### Step 4: Human Annotation

Extract a stratified sample from the judge results and use the annotation web app to collect annotations from experts:

```bash
# Extract stratified sample for annotation
python -m langdmta_eval.human_annotation.extract_annotation_sample \
    results/judge_results/evaluation_results.json \
    --n-samples 30 \
    --n-duplicates 5 \


# Launch annotation app
python -m langdmta_eval.human_annotation.human_annotation_app \
    annotator_sheet.csv
```

Then analyze LLM-human alignment and inter-annotator agreement using the notebook:

**[`human_annotation/human_alignment_analysis_multimodels.ipynb`](human_annotation/human_alignment_analysis_multimodels.ipynb)**

This notebook covers intra-LLM alignment, intra-human alignment, and human-LLM alignment across multiple models, with formal agreement metrics (Cohen's κ, Fleiss' κ, Spearman ρ).

### Step 5: Optimize the Judge Signature

If alignment is insufficient, optimize the judge signatures using the optimization cells in:

**[`human_annotation/human_alignment_analysis_multimodels.ipynb`](human_annotation/human_alignment_analysis_multimodels.ipynb)**

Optimization methods available: `LabeledFewShot`, `BootstrapFewShot`, `MIPROv2`. The optimized signature is saved to a JSON file to be passed back into the judge.

### Step 6: Re-run Judge and Analyze Results

Re-run the judge with the optimized signature, then analyze results using the evaluation notebook:

```bash
python -m langdmta_eval.run_judge <summary.csv> \
    --model azure/gpt-5 \
    --judge-type comprehensive \
    --optimized-path optimized_judge.json \
    --output evaluation_results_optimized.json
```

Then explore results with:

**[`evaluate_judge.ipynb`](evaluate_judge.ipynb)**

This notebook covers score distributions, per-category breakdowns, model comparisons, and before/after optimization comparisons.

---

## Package Structure

```
langdmta_eval/
├── run_judge.py                  # CLI entry point for batch evaluation
├── evaluate_judge.ipynb          # Analysis notebook: score distributions, model comparisons
├── schemas.py                    # TestCase, EvaluationResult, BatchEvaluationResult; JSON load/save
├── evaluators/
│   ├── base.py                   # BaseEvaluator abstract class
│   └── dspy_evaluator.py         # DSPyEvaluator with async batch + provider routing
├── judges/
│   ├── signatures.py             # Judge signatures
│   └── score_mappings.py         # Categorical <-> numeric score conversion
├── metrics/
│   └── tool_validation.py        # Agent tool-call sequence validation
├── alignment/
│   ├── data_loader.py            # Load/merge human + LLM scores
│   ├── metrics.py                # Agreement statistics
│   └── visualization.py          # Publication-quality plots
├── optimization/
│   ├── alignment_metric.py       # DSPy-compatible metric with partial credit
│   ├── signature_optimizer.py    # Optimization pipeline orchestrator
│   └── versioning.py             # Judge version tracking
├── human_annotation/
│   ├── human_annotation_app.py           # Flask annotation web app
│   ├── extract_annotation_sample.py      # Stratified sample extraction
│   └── human_alignment_analysis_multimodels.ipynb  # Alignment analysis + optimization
├── visualization/
│   └── visualize.py              # General evaluation result plots
```

---

## Part 1: LLM Judge

**Core idea**: Use Chain-of-Thought prompting via DSPy to evaluate agent responses on multiple orthogonal dimensions, with automated tool-call sequence validation.

> See the [run_judge.py](run_judge.py) CLI (`python -m langdmta_eval.run_judge --help`) for all options.

### Evaluation Metrics

| Metric | Scale | What it measures |
|--------|-------|-----------------|
| Correctness | INCORRECT / PARTIALLY CORRECT / CORRECT | Are tool calls and factual claims accurate? |
| Completeness | INCOMPLETE / PARTIALLY COMPLETE / COMPLETE | Recall: is all required information present? |
| Relevancy | NOT RELEVANT / PARTIALLY RELEVANT / RELEVANT | Precision: is all included content relevant? |
| Structural Clarity | NOT CLEAR / PARTIALLY CLEAR / CLEAR | Is the output well-structured and readable? |
| Scope Adherence | BELOW TARGET / ON TARGET / ABOVE TARGET | Does the agent stay on-target? |
| Ethical Awareness | NOT AWARE / PARTIALLY AWARE / AWARE / NOT APPLICABLE | Are ethical concerns appropriately handled? |


### Judge Signature Types

- `basic` -- Completeness, Relevancy, Structural Clarity, Scope Adherence
- `ethical` -- Basic + Ethical Awareness
- `capability` -- Basic + Scope Adherence
- `comprehensive` -- All metrics


### Key Modules

| Module | Purpose |
|--------|---------|
| `judges/signatures.py` | Defines evaluation criteria as DSPy signatures |
| `evaluators/dspy_evaluator.py` | Runs the judge with async batch support and multi-provider routing |
| `judges/score_mappings.py` | Normalizes categorical scores (CORRECT → 1.0, PARTIALLY → 0.5, etc.) |
| `metrics/tool_validation.py` | Validates agent tool-call sequences against expected sequences |
| `run_judge.py` | CLI entry point for batch evaluation |
| `schemas.py` | Core schemas: TestCase, EvaluationResult, BatchEvaluationResult; JSON load/save |

---

## Part 2: Human Annotation and Alignment

**Core idea**: Validate the LLM judge against human expert annotations, measure inter-annotator reliability to contextualize disagreements, then optimize the judge to improve agreement.

> See **[`human_annotation/README.md`](human_annotation/README.md)** for the annotation guide, scoring rubrics, web app usage, and sample extraction options.
>
> See **[`alignment/README.md`](alignment/README.md)** for details on the alignment modules and available metric functions.
>
> See **[`optimization/README.md`](optimization/README.md)** for details on optimization strategies and versioning.

### Human Annotation System

| Component | Purpose |
|-----------|---------|
| `human_annotation/human_annotation_app.py` | Flask-based web app for annotators to score agent outputs |
| `human_annotation/extract_annotation_sample.py` | Stratified sampling: calibration anchors + complex multi-tool workflows |

### Alignment Analysis & Optimization Notebook

**[`human_annotation/human_alignment_analysis_multimodels.ipynb`](human_annotation/human_alignment_analysis_multimodels.ipynb)**

The primary notebook for alignment work. Covers:

- Intra-LLM alignment (self-consistency via duplicate questions)
- Intra-human alignment (inter-annotator agreement)
- Human-LLM alignment across multiple models
- Judge signature optimization (LabeledFewShot, BootstrapFewShot, MIPROv2)

### Alignment Modules

| Component | Purpose |
|-----------|---------|
| `alignment/data_loader.py` | Merges human + LLM scores; supports multiple annotators and models |
| `alignment/metrics.py` | Cohen's κ (weighted + unweighted), Fleiss' κ, Spearman ρ, exact match |
| `alignment/visualization.py` | Confusion matrices, bar charts, before/after comparisons |

### Optimization Modules

| Component | Purpose |
|-----------|---------|
| `optimization/alignment_metric.py` | DSPy-compatible metric with partial credit scoring |
| `optimization/signature_optimizer.py` | Orchestrates optimization runs |
| `optimization/versioning.py` | Tracks judge versions with per-metric kappa scores over time |

---

## Part 3: Evaluation Analysis

**[`evaluate_judge.ipynb`](evaluate_judge.ipynb)**

The primary notebook for exploring judge output. Covers score distributions, per-category breakdowns, cross-model comparisons, and before/after optimization analysis.

---

## Part 4: Test Question Generation

**Core idea**: Programmatically generate diverse test questions from templates, ensuring systematic coverage of question phrasings and chemical entities.

> See **[`tests/question_generation/README.md`](../../tests/question_generation/README.md)** for the full workflow and CLI flags.

Located in `tests/question_generation/`:

| Component | Purpose |
|-----------|---------|
| `generate_test_question.py` | Parses templates with `{{placeholders}}`, substitutes from lookup tables |
| `variations_test_questions.csv` | Template bank with variation levels (0=reference, 1–10=informal/rephrased) |
| `molecule_names.csv` | Compound names (Ibuprofen, Morpholine, etc.) |
| `smiles.csv` | SMILES chemical structure strings |
| `reactions.csv` | Reaction SMARTS strings |
| `property_constraints.csv` | Property specification templates |

---

## Multi-Provider Support

| Provider | Model string |
|----------|-------------|
| Azure OpenAI | `azure/gpt-5` |
| AWS Bedrock | `bedrock/us.anthropic.claude-opus-4-7` |
| AWS Bedrock | `bedrock/us.meta.llama3-1-70b-instruct-v1:0` |
| Google Gemini | `gemini/gemini-2.5-pro` |
| Google Vertex AI | `vertex_ai/gemini-2.5-pro` |

---

## Dependencies

- `dspy-ai` -- Chain-of-Thought reasoning and optimization
- `scikit-learn` -- Agreement metrics
- `scipy` -- Statistical tests
- `pandas` / `numpy` -- Data manipulation
- `matplotlib` / `seaborn` -- Visualization
- `flask` -- Human annotation web app
