"""
Rejection Sampling Fine-Tuning (RSFT) - RL-lite approach.

For each prompt:
1. Sample K completions
2. Score with reward function
3. Keep top-1 (or top-N) as pseudo-labels
4. Fine-tune LoRA on winners
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from .reward import compute_reward
from .kg import load_kg, get_entity_vocab
from .infer import load_mlx_model, run_inference
from .mlx_sft import prepare_sft_data, train_lora, load_model_with_lora


# Default configuration
DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"

DEFAULT_RSFT_CONFIG = {
    "k_samples": 8,
    "keep_top": 1,
    "temperature": 0.7,
    "max_tokens": 256,
}

DEFAULT_TRAIN_CONFIG = {
    "num_iters": 500,
    "batch_size": 4,
    "learning_rate": 5e-5,  # Lower LR for RSFT (continuing from SFT)
    "lora_rank": 16,
    "lora_layers": 16,
}


def sample_completions(
    model,
    tokenizer,
    prompt: str,
    k: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 256
) -> List[str]:
    """
    Sample K completions for a prompt.

    Uses temperature sampling to get diverse outputs.
    """
    completions = []

    for i in range(k):
        try:
            completion = run_inference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False
            )
            completions.append(completion)
        except Exception as e:
            print(f"Warning: Failed to generate completion {i}: {e}")
            continue

    return completions


def select_winners(
    completions: List[str],
    example: Dict[str, Any],
    entity_vocab: set,
    keep_top: int = 1,
    config: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Score completions and select top performers.

    Returns list of (completion, reward_breakdown) for winners.
    """
    if not completions:
        return []

    scored = []

    for completion in completions:
        reward = compute_reward(
            completion=completion,
            answer_star=example["answer_star"],
            path_entities=example["path_star"]["entities"],
            entity_vocab=entity_vocab,
            config=config
        )
        scored.append((completion, reward))

    # Sort by total reward (descending)
    scored.sort(key=lambda x: x[1]["total"], reverse=True)

    # Tie-breaking: prefer higher correctness, then coverage, then shorter
    def tiebreak_key(item):
        comp, r = item
        return (
            r["total"],
            r["correctness"],
            r["path_coverage"],
            -len(comp)  # shorter is better
        )

    scored.sort(key=tiebreak_key, reverse=True)

    return scored[:keep_top]


def generate_rsft_dataset(
    model,
    tokenizer,
    examples_path: Path,
    kg_path: Path,
    output_path: Path,
    k_samples: int = 8,
    keep_top: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_examples: Optional[int] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate RSFT training data by rejection sampling.

    Writes JSONL with winning (prompt, completion) pairs.

    Returns:
        Statistics about the generation process
    """
    kg = load_kg(kg_path)
    entity_vocab = get_entity_vocab(kg)

    # Load examples
    examples = []
    with open(examples_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if max_examples:
        examples = examples[:max_examples]

    stats = {
        "total_examples": len(examples),
        "total_samples": 0,
        "winners": 0,
        "avg_winner_reward": 0.0,
        "correct_winners": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_out:
        for example in tqdm(examples, desc="Generating RSFT data"):
            # Sample completions
            completions = sample_completions(
                model=model,
                tokenizer=tokenizer,
                prompt=example["prompt"],
                k=k_samples,
                temperature=temperature,
                max_tokens=max_tokens
            )

            stats["total_samples"] += len(completions)

            if not completions:
                continue

            # Select winners
            winners = select_winners(
                completions=completions,
                example=example,
                entity_vocab=entity_vocab,
                keep_top=keep_top,
                config=reward_config
            )

            # Write winning pairs
            for completion, reward in winners:
                # Only include winners with positive reward or correct answer
                if reward["total"] > 0 or reward["correctness"] > 0:
                    f_out.write(json.dumps({
                        "prompt": example["prompt"],
                        "completion": completion,
                        "reward": reward["total"],
                        "ex_id": example["id"]
                    }) + "\n")

                    stats["winners"] += 1
                    stats["avg_winner_reward"] += reward["total"]

                    if reward["correctness"] > 0:
                        stats["correct_winners"] += 1

            if verbose and len(winners) > 0:
                best_completion, best_reward = winners[0]
                print(f"\nExample: {example['id']}")
                print(f"  Best reward: {best_reward['total']:.2f}")
                print(f"  Correct: {best_reward['correctness'] > 0}")
                print(f"  Coverage: {best_reward['path_coverage']:.2%}")

    # Finalize stats
    if stats["winners"] > 0:
        stats["avg_winner_reward"] /= stats["winners"]

    print(f"\nRSFT Dataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Total samples generated: {stats['total_samples']}")
    print(f"  Winners selected: {stats['winners']}")
    print(f"  Correct winners: {stats['correct_winners']}")
    print(f"  Avg winner reward: {stats['avg_winner_reward']:.2f}")

    return stats


def prepare_rsft_training_data(
    rsft_data_path: Path,
    output_path: Path,
    use_chat_format: bool = True
) -> int:
    """
    Convert RSFT winners to MLX training format.

    Similar to SFT data preparation but uses actual model completions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(rsft_data_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            item = json.loads(line)

            prompt = item["prompt"]
            completion = item["completion"]

            if use_chat_format:
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
            else:
                text = f"{prompt}\n{completion}"

            f_out.write(json.dumps({"text": text}) + "\n")
            count += 1

    print(f"Prepared {count} RSFT training examples -> {output_path}")
    return count


def run_rsft(
    base_model: str,
    sft_adapter: Path,
    examples_path: Path,
    kg_path: Path,
    output_dir: Path,
    rsft_config: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    max_examples: Optional[int] = None,
    verbose: bool = False
) -> Path:
    """
    Full RSFT pipeline:
    1. Load SFT model
    2. Generate winners dataset
    3. Continue LoRA training on winners

    Args:
        base_model: Base model ID
        sft_adapter: Path to SFT adapter weights
        examples_path: Training examples JSONL
        kg_path: Knowledge graph JSON
        output_dir: Output directory for RSFT adapter
        rsft_config: RSFT sampling configuration
        train_config: Training configuration
        max_examples: Limit number of examples to process
        verbose: Print verbose output

    Returns:
        Path to RSFT adapter directory
    """
    # Merge configs with defaults
    rsft_cfg = DEFAULT_RSFT_CONFIG.copy()
    if rsft_config:
        rsft_cfg.update(rsft_config)

    train_cfg = DEFAULT_TRAIN_CONFIG.copy()
    if train_config:
        train_cfg.update(train_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    full_config = {
        "base_model": base_model,
        "sft_adapter": str(sft_adapter),
        "rsft_config": rsft_cfg,
        "train_config": train_cfg,
    }
    with open(output_dir / "rsft_config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    # Step 1: Load SFT model
    print(f"Loading model with SFT adapter from: {sft_adapter}")
    model, tokenizer = load_model_with_lora(base_model, sft_adapter)

    # Step 2: Generate winners dataset
    print("\nGenerating RSFT training data...")
    rsft_data_path = output_dir / "rsft_winners.jsonl"

    stats = generate_rsft_dataset(
        model=model,
        tokenizer=tokenizer,
        examples_path=examples_path,
        kg_path=kg_path,
        output_path=rsft_data_path,
        k_samples=rsft_cfg["k_samples"],
        keep_top=rsft_cfg["keep_top"],
        temperature=rsft_cfg["temperature"],
        max_tokens=rsft_cfg["max_tokens"],
        max_examples=max_examples,
        verbose=verbose
    )

    # Check if we have enough winners
    if stats["winners"] < 10:
        print(f"Warning: Only {stats['winners']} winners found. RSFT may not be effective.")
        print("Consider increasing k_samples or using a better SFT model.")

    # Step 3: Prepare training data
    rsft_train_path = output_dir / "rsft_train.jsonl"
    prepare_rsft_training_data(rsft_data_path, rsft_train_path)

    # Step 4: Continue LoRA training on winners
    print("\nTraining RSFT LoRA adapter...")

    # For RSFT, we can either:
    # A) Continue training the SFT adapter (resume=True)
    # B) Train a fresh adapter on RSFT data

    # Option A is more common in practice
    adapter_dir = train_lora(
        base_model=base_model,
        train_data=rsft_train_path,
        output_dir=output_dir,
        config=train_cfg,
        resume=sft_adapter.exists()  # Resume from SFT if it exists
    )

    print(f"\nRSFT complete! Adapter saved to: {adapter_dir}")
    return adapter_dir


def main():
    """Command-line interface for RSFT."""
    parser = argparse.ArgumentParser(description="Run Rejection Sampling Fine-Tuning")
    parser.add_argument("--examples", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--kg", type=Path, required=True, help="Path to kg.json")
    parser.add_argument("--sft-adapter", type=Path, required=True, help="Path to SFT adapter")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for RSFT")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--k-samples", type=int, default=8, help="Number of samples per prompt")
    parser.add_argument("--keep-top", type=int, default=1, help="Number of winners to keep")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")
    parser.add_argument("--iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    rsft_config = {
        "k_samples": args.k_samples,
        "keep_top": args.keep_top,
        "temperature": args.temperature,
    }

    train_config = {
        "num_iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }

    run_rsft(
        base_model=args.model,
        sft_adapter=args.sft_adapter,
        examples_path=args.examples,
        kg_path=args.kg,
        output_dir=args.output,
        rsft_config=rsft_config,
        train_config=train_config,
        max_examples=args.max_examples,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
