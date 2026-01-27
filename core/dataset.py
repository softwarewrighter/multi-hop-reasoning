"""
Dataset generation from KG paths.

See spec/schemas.md for train.jsonl format.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from .kg import (
    load_kg,
    sample_path,
    get_entity_vocab,
    get_entity_types,
    get_neighbors,
    get_entity_by_type
)


# Question templates based on relation types
QUESTION_TEMPLATES = {
    "caused_by": [
        "Your system shows {start}. What is the most likely underlying cause?",
        "You're seeing {start} in production. What typically causes this issue?",
        "A user reports {start}. What is the root cause?",
        "{start} has been detected. What usually causes this problem?",
    ],
    "fixed_by": [
        "{start} needs to be resolved. What is the recommended fix?",
        "How should you fix {start}?",
        "What action resolves {start}?",
        "Your team encounters {start}. What's the solution?",
    ],
    "diagnosed_by": [
        "To investigate {start}, what diagnostic should you run first?",
        "What's the best way to diagnose {start}?",
        "You need to troubleshoot {start}. What diagnostic step comes first?",
        "How do you identify the cause of {start}?",
    ],
    "uses_tool": [
        "What tool should you use for {start}?",
        "Which tool is most appropriate for {start}?",
        "{start} requires what tool?",
    ],
    "leads_to": [
        "If {start} is not addressed, what problem will it likely cause?",
        "What issue does {start} typically lead to?",
        "{start} can escalate to what symptom?",
    ],
    "related_to": [
        "{start} is closely related to which other issue?",
        "What problem is most similar to {start}?",
    ],
}


def generate_distractors(
    correct_answer: str,
    kg: Dict[str, Any],
    path_entities: List[str],
    n: int = 3,
    seed: Optional[int] = None
) -> List[str]:
    """
    Generate n distractor options from KG entities.

    Prefers "near-miss" distractors that are:
    1. Same type as the correct answer
    2. Adjacent in the graph (1-2 hops away)
    3. Not in the path
    """
    if seed is not None:
        random.seed(seed)

    entity_types = get_entity_types(kg)
    correct_type = entity_types.get(correct_answer, "unknown")
    path_set = set(path_entities)

    # Get entities of the same type (excluding path entities)
    same_type = [
        e["id"] for e in kg["entities"]
        if e.get("type") == correct_type
        and e["id"] not in path_set
        and e["id"] != correct_answer
    ]

    # Get neighbors of the correct answer (near-miss distractors)
    neighbors = get_neighbors(kg, correct_answer, max_hops=2)
    neighbor_distractors = [
        n for n in neighbors
        if n not in path_set and n != correct_answer
    ]

    # Prioritize: same-type neighbors > same-type > neighbors > any
    distractors = []

    # 1. Same type AND neighbor (best distractors)
    same_type_neighbors = [d for d in neighbor_distractors if entity_types.get(d) == correct_type]
    random.shuffle(same_type_neighbors)
    distractors.extend(same_type_neighbors)

    # 2. Same type (not neighbor)
    same_type_only = [d for d in same_type if d not in neighbor_distractors]
    random.shuffle(same_type_only)
    distractors.extend(same_type_only)

    # 3. Neighbors of different type
    other_neighbors = [d for d in neighbor_distractors if d not in same_type_neighbors]
    random.shuffle(other_neighbors)
    distractors.extend(other_neighbors)

    # 4. Any remaining entities
    all_entities = [
        e["id"] for e in kg["entities"]
        if e["id"] not in path_set
        and e["id"] != correct_answer
        and e["id"] not in distractors
    ]
    random.shuffle(all_entities)
    distractors.extend(all_entities)

    # Take the first n unique distractors
    return distractors[:n]


def generate_reference_trace(path: Dict[str, Any]) -> str:
    """Generate reference trace mentioning all path entities."""
    entities = path["entities"]
    edges = path.get("edges", [])

    # Build a natural language trace
    parts = []
    for edge in edges:
        rel = edge["rel"].replace("_", " ")
        parts.append(f"{edge['src']} is {rel} {edge['dst']}")

    trace = ", and ".join(parts) if parts else f"The answer involves {entities[-1]}"
    return f"TRACE: {trace}"


def build_prompt(question: str, options: Dict[str, str]) -> str:
    """Build the full prompt with output format instructions."""
    prompt = """You must follow the exact output format.

OUTPUT FORMAT:
TRACE: <one or two sentences explaining your reasoning>
ANSWER: <A|B|C|D>

Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
"""
    return prompt.format(question=question, **options)


def generate_mcq(
    path: Dict[str, Any],
    kg: Dict[str, Any],
    example_id: str,
    split: str = "train",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate an MCQ from a KG path.

    Returns example in train.jsonl format.
    """
    if seed is not None:
        random.seed(seed)

    entities = path["entities"]
    edges = path["edges"]

    if not edges:
        raise ValueError("Path must have at least one edge")

    # The question is about the path: start -> ... -> answer
    # We ask about the final destination or an intermediate node
    start_entity = entities[0]
    answer_entity = entities[-1]

    # Determine question type from the first edge's relation
    first_rel = edges[0]["rel"]

    # Get appropriate question template
    templates = QUESTION_TEMPLATES.get(first_rel, QUESTION_TEMPLATES["caused_by"])
    question_template = random.choice(templates)
    question = question_template.format(start=start_entity)

    # Generate distractors
    distractors = generate_distractors(
        correct_answer=answer_entity,
        kg=kg,
        path_entities=entities,
        n=3,
        seed=seed
    )

    # Randomize option positions
    all_options = [answer_entity] + distractors
    random.shuffle(all_options)

    # Find correct answer position
    option_labels = ["A", "B", "C", "D"]
    options = {label: opt for label, opt in zip(option_labels, all_options)}
    answer_star = option_labels[all_options.index(answer_entity)]

    # Build prompt
    prompt = build_prompt(question, options)

    # Generate reference trace and answer
    ref_trace = generate_reference_trace(path)
    ref_full = f"{ref_trace}\nANSWER: {answer_star}"

    return {
        "id": example_id,
        "split": split,
        "hop_len": len(edges),
        "question": question,
        "options": options,
        "answer_star": answer_star,
        "path_star": {
            "entities": entities,
            "edges": edges
        },
        "prompt": prompt,
        "ref": {
            "trace": ref_full,
            "style_rules": [
                "Mention at least 2 path entities.",
                "Do not list all options.",
                "Do not repeat an entity more than twice."
            ]
        },
        "meta": {
            "topic": first_rel,
            "difficulty": "easy" if len(edges) <= 2 else "medium" if len(edges) <= 3 else "hard",
            "distractor_type": "near_miss"
        }
    }


def generate_dataset(
    kg_path: Path,
    output_path: Path,
    n_examples: int,
    hop_lengths: List[int],
    split: str = "train",
    seed: int = 42
) -> None:
    """Generate a dataset file from KG."""
    random.seed(seed)

    kg = load_kg(kg_path)

    # Distribute examples across hop lengths
    examples_per_hop = n_examples // len(hop_lengths)
    remainder = n_examples % len(hop_lengths)

    examples = []
    example_idx = 0

    for i, hop_len in enumerate(hop_lengths):
        # Add remainder to first hop lengths
        n_for_hop = examples_per_hop + (1 if i < remainder else 0)

        for j in range(n_for_hop):
            # Use different seeds for each path
            path_seed = seed + example_idx * 1000

            try:
                # Sample path starting from symptoms for more natural questions
                path = sample_path(
                    kg=kg,
                    length=hop_len,
                    seed=path_seed,
                    start_types=["symptom", "cause"],
                    preferred_relations=["caused_by", "fixed_by", "diagnosed_by", "leads_to"]
                )

                # Generate MCQ
                example_id = f"ex_{split}_{example_idx:06d}"
                mcq = generate_mcq(
                    path=path,
                    kg=kg,
                    example_id=example_id,
                    split=split,
                    seed=path_seed + 1
                )

                examples.append(mcq)
                example_idx += 1

            except ValueError as e:
                # Try with any start type
                try:
                    path = sample_path(
                        kg=kg,
                        length=hop_len,
                        seed=path_seed + 500
                    )
                    example_id = f"ex_{split}_{example_idx:06d}"
                    mcq = generate_mcq(
                        path=path,
                        kg=kg,
                        example_id=example_id,
                        split=split,
                        seed=path_seed + 501
                    )
                    examples.append(mcq)
                    example_idx += 1
                except ValueError:
                    print(f"Warning: Could not generate path of length {hop_len}")
                    continue

    # Shuffle examples
    random.shuffle(examples)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(examples)} examples -> {output_path}")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate MCQ datasets from KG")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate train and eval datasets")
    gen_parser.add_argument("--kg", type=Path, required=True, help="Path to kg.json")
    gen_parser.add_argument("--train", type=Path, required=True, help="Output path for train.jsonl")
    gen_parser.add_argument("--eval", type=Path, required=True, help="Output path for eval.jsonl")
    gen_parser.add_argument("--n-train", type=int, default=500, help="Number of training examples")
    gen_parser.add_argument("--n-eval", type=int, default=100, help="Number of eval examples")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "generate":
        # Generate training set (1-3 hops)
        print("Generating training dataset (1-3 hops)...")
        generate_dataset(
            kg_path=args.kg,
            output_path=args.train,
            n_examples=args.n_train,
            hop_lengths=[1, 2, 3],
            split="train",
            seed=args.seed
        )

        # Generate eval set (4-5 hops)
        print("Generating eval dataset (4-5 hops)...")
        generate_dataset(
            kg_path=args.kg,
            output_path=args.eval,
            n_examples=args.n_eval,
            hop_lengths=[4, 5],
            split="eval",
            seed=args.seed + 10000
        )

        print("Done!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
