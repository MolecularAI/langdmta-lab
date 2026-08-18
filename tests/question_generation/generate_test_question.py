"""
Generate test questions from templates by substituting molecule/reaction variables.

Usage:
    python tests/generate_test_questions.py \
        --variations variations_test_questions_generated.csv \
        --molecules molecule_names.csv \
        --smiles smiles.csv \
        --reactions reactions.csv \
        --property-constraints property_constraints.csv \
        --output test_questions_generated.toml \
        --num-samples 3
"""

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_sequences(seq_str: str) -> List[List[str]]:
    """Parse sequence string with delimiters into nested list format.

    Examples:
        "molformer|molformer;Analyzer" -> [['molformer'], ['molformer', 'Analyzer']]
        "synonyms2smiles;aizynthfinder" -> [['synonyms2smiles', 'aizynthfinder']]
        "aizynthfinder" -> [['aizynthfinder']]
        "" -> []

    Args:
        seq_str: Sequence string using | for outer list, ; for inner list

    Returns:
        Nested list structure for TOML sequences
    """
    if pd.isna(seq_str) or seq_str.strip() == "":
        return []

    # Split by | for multiple sequence options
    outer_sequences = seq_str.split("|")

    # Split each sequence by ; for tool call chains
    return [[tool.strip() for tool in seq.split(";")] for seq in outer_sequences]


def extract_placeholders(text: str) -> List[str]:
    """Extract all {{placeholder}} variables from text."""
    return re.findall(r"\{\{\s*(\w+)\s*\}\}", text)


def count_placeholder_occurrences(text: str) -> Dict[str, int]:
    """Count how many times each placeholder appears in text."""
    placeholders = extract_placeholders(text)
    return {ph: placeholders.count(ph) for ph in set(placeholders)}


def substitute_variables(
    question_template: str,
    molecules: List[str],
    smiles: List[str],
    reactions: List[str],
    property_constraint: List[str],
) -> str:
    """Substitute placeholders in question template with random values.

    Handles multiple occurrences of same placeholder (e.g., two {{smiles}}).

    Args:
        question_template: Template string with {{placeholders}}
        molecules: List of molecule names to sample from
        smiles: List of SMILES strings to sample from
        reactions: List of reaction SMILES to sample from
        property_constraint: List of property constraints to sample from

    Returns:
        Question with all placeholders substituted
    """
    # Count occurrences of each placeholder type
    placeholder_counts = count_placeholder_occurrences(question_template)

    # Sample required number of unique values for each placeholder type
    samples = {}
    if "molecule_name" in placeholder_counts:
        count = min(placeholder_counts["molecule_name"], len(molecules))
        samples["molecule_name"] = random.sample(molecules, count)

    if "smiles" in placeholder_counts:
        count = min(placeholder_counts["smiles"], len(smiles))
        samples["smiles"] = random.sample(smiles, count)

    if "reaction" in placeholder_counts:
        count = min(placeholder_counts["reaction"], len(reactions))
        samples["reaction"] = random.sample(reactions, count)

    if "property_constraint" in placeholder_counts:
        count = min(placeholder_counts["property_constraint"], len(property_constraint))
        samples["property_constraint"] = random.sample(property_constraint, count)

    # Substitute placeholders sequentially (first occurrence, second occurrence, etc.)
    question = question_template
    for placeholder_type, values in samples.items():
        for value in values:
            # Replace first occurrence of {{placeholder_type}}
            placeholder = f"{{{{{placeholder_type}}}}}"
            question = question.replace(placeholder, value, 1)

    return question


def generate_test_questions(
    variations_csv: str,
    molecules_csv: str,
    smiles_csv: str,
    reactions_csv: str,
    property_constraint_csv: str,
    output_toml: str,
    num_samples: int = 1,
    n_question_variations: int = None,
    seed: int = 42,
) -> None:
    """Generate test questions from reference and variation templates.

    Generates num_samples questions from EACH variation (including reference with label 0).

    Args:
        variations_csv: CSV with columns: reference_question, variation_label,
                       variation_question, task_type, task_label,
                       tool_calls_complete, tool_calls_accepted, note
        molecules_csv: CSV with molecule_name column
        smiles_csv: CSV with smiles column
        reactions_csv: CSV with reaction column (Reaction SMILES)
        property_constraint_csv: CSV with property_constraint column
        output_toml: Output path for generated test questions TOML
        num_samples: Number of questions to generate per template (default: 1)
        n_question_variations: Number of variations to sample per reference (None = all)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load data
    variations_df = pd.read_csv(variations_csv)
    molecules_df = pd.read_csv(molecules_csv)
    smiles_df = pd.read_csv(smiles_csv)
    reactions_df = pd.read_csv(reactions_csv)
    property_constraint_df = pd.read_csv(property_constraint_csv)

    # Extract lists and remove NaN values
    molecules = molecules_df["molecule_name"].dropna().tolist()
    smiles = smiles_df["smiles"].dropna().tolist()
    reactions = reactions_df["reaction"].dropna().tolist()
    property_constraint = (
        property_constraint_df["property_constraint"].dropna().tolist()
    )

    print(f"📊 Loaded data:")
    print(f"   - {len(molecules)} molecule names")
    print(f"   - {len(smiles)} SMILES strings")
    print(f"   - {len(reactions)} reaction SMILES")
    print(f"   - {len(property_constraint)} property constraints")
    print(f"   - {len(variations_df)} question variations")

    # Group variations by reference_question
    variation_groups = variations_df.groupby("reference_question")

    # Generate questions
    generated_by_task_label = {}

    for reference_question, group_df in variation_groups:
        # Get task metadata from first row (same for all variations)
        first_row = group_df.iloc[0]
        task_type = first_row["task_type"]
        task_label = first_row["task_label"]
        tool_calls_complete = first_row["tool_calls_complete"]
        tool_calls_accepted = first_row.get("tool_calls_accepted", "")

        # Initialize task_label entry
        if task_label not in generated_by_task_label:
            generated_by_task_label[task_label] = []

        # Generate questions from each variation (including reference with label 0)
        # Sample variations if n_question_variations is specified (independent sampling per group)
        variations_to_use = group_df
        if n_question_variations is not None and len(group_df) > n_question_variations:
            variations_to_use = group_df.sample(
                n=n_question_variations
            )  # , random_state=42) #include random state if same samples for each question

        for _, variation_row in variations_to_use.iterrows():
            variation_question = variation_row["variation_question"]
            variation_label = variation_row["variation_label"]
            note = variation_row.get("note", "")

            for _ in range(num_samples):
                question_text = substitute_variables(
                    variation_question,
                    molecules,
                    smiles,
                    reactions,
                    property_constraint,
                )

                test_entry = {
                    "question": question_text,
                    "question_variation": variation_label,
                    "task_type": task_type,
                    "note": note if pd.notna(note) else "",
                    "sequences": {
                        "complete": parse_sequences(tool_calls_complete),
                        "accepted": parse_sequences(tool_calls_accepted),
                    },
                }

                generated_by_task_label[task_label].append(test_entry)

    # Write TOML file manually to get proper inline table format for sequences
    output_path = Path(output_toml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for task_label, questions in generated_by_task_label.items():
            for question_entry in questions:
                task_type = question_entry["task_type"]
                f.write(f"[[{task_type}.{task_label}]]\n")
                f.write(f'question = "{question_entry["question"]}"\n')
                f.write('answer = ""\n')
                f.write(
                    f'question_variation = {question_entry["question_variation"]}\n'
                )
                f.write(f'note = "{question_entry["note"]}"\n')

                # Write sequences as inline tables
                complete_seq = question_entry["sequences"]["complete"]
                accepted_seq = question_entry["sequences"]["accepted"]

                f.write(f"sequences.complete = {complete_seq}\n")
                f.write(f"sequences.accepted = {accepted_seq}\n")
                f.write("\n")

    # Print summary
    total_questions = sum(len(q) for q in generated_by_task_label.values())
    total_variations = sum(len(group) for _, group in variation_groups)

    print(f"\n✅ Generated {total_questions} test questions")
    print(f"   - {len(variation_groups)} unique reference questions")
    print(f"   - {total_variations} variations")
    print(f"   - {num_samples} samples per template")
    print(f"📝 Output written to: {output_path}")

    print("\n📋 Breakdown by task:")
    for task_label, questions in sorted(generated_by_task_label.items()):
        print(f"   - {task_label}: {len(questions)} questions")

    # Print sample questions
    print("\n🔍 Sample generated questions:")
    for task_label, questions in list(generated_by_task_label.items())[:3]:
        sample_q = questions[0]["question"]
        task_type = questions[0]["task_type"]
        sequences = questions[0]["sequences"]["complete"]
        accepted_sequences = questions[0]["sequences"]["accepted"]
        print(f"\n[[{task_type}.{task_label}]]")
        print(f'question = "{sample_q}"')
        print(f"sequences.complete = {sequences}")
        print(f"sequences.accepted = {accepted_sequences}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LangDMTA test questions from variation templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 question per template (reference + all variations)
  python tests/generate_test_questions.py

  # Generate 5 samples per template with different molecule substitutions
  python tests/generate_test_questions.py --num-samples 5

  # Use custom seed for different random sampling
  python tests/generate_test_questions.py --num-samples 3 --seed 123
        """,
    )

    parser.add_argument(
        "--variations",
        default="variations_test_questions_generated.csv",
        help="Path to variations CSV file",
    )
    parser.add_argument(
        "--molecules", default="molecule_names.csv", help="Path to molecule names CSV"
    )
    parser.add_argument("--smiles", default="smiles.csv", help="Path to SMILES CSV")
    parser.add_argument(
        "--reactions",
        default="reactions.csv",
        help="Path to reactions CSV (Reaction SMILES format)",
    )
    parser.add_argument(
        "--property-constraints",
        default="property_constraints.csv",
        help="Path to property constraints CSV",
    )
    parser.add_argument(
        "--output", default="test_questions_generated.toml", help="Output TOML path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of questions to generate per template (default: 1)",
    )
    parser.add_argument(
        "--n-question-variations",
        type=int,
        default=None,
        help="Number of question variations to sample per reference question (default: None, use all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    generate_test_questions(
        variations_csv=args.variations,
        molecules_csv=args.molecules,
        smiles_csv=args.smiles,
        reactions_csv=args.reactions,
        property_constraint_csv=args.property_constraints,
        output_toml=args.output,
        num_samples=args.num_samples,
        n_question_variations=args.n_question_variations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
