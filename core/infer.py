"""
Model inference and episode logging.

Runs inference on examples and logs results to episodes.jsonl.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple

from .reward import compute_reward
from .kg import load_kg, get_entity_vocab


# Default model
DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


def load_examples(path: Path) -> Iterator[Dict[str, Any]]:
    """Load examples from JSONL file."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_mlx_model(
    model_path: str,
    adapter_path: Optional[Path] = None
) -> Tuple[Any, Any]:
    """
    Load MLX model and tokenizer.

    Args:
        model_path: HuggingFace model ID or local path
        adapter_path: Optional path to LoRA adapter weights

    Returns:
        (model, tokenizer) tuple
    """
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise ImportError("mlx-lm is required. Install with: pip install mlx-lm")

    if adapter_path and adapter_path.exists():
        # Load model with LoRA adapters
        model, tokenizer = load(model_path, adapter_path=str(adapter_path))
    else:
        # Load base model
        model, tokenizer = load(model_path)

    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = False
) -> str:
    """
    Run inference on a single prompt using MLX-LM.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        verbose: Print generation progress

    Returns:
        Generated completion text
    """
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        raise ImportError("mlx-lm is required. Install with: pip install mlx-lm")

    # Format as chat if the model supports it
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fall back to raw prompt if chat template fails
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt

    # Create sampler with temperature
    sampler = make_sampler(temp=temperature)

    # Generate completion
    completion = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=verbose
    )

    return completion


def run_inference_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.0
) -> List[str]:
    """
    Run inference on multiple prompts.

    Note: MLX-LM doesn't have native batching, so we process sequentially.
    """
    completions = []
    for prompt in prompts:
        completion = run_inference(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        completions.append(completion)
    return completions


def log_episode(
    example: Dict[str, Any],
    completion: str,
    reward_breakdown: Dict[str, Any],
    phase: str,
    step: int
) -> Dict[str, Any]:
    """Create an episode log entry."""
    return {
        "phase": phase,
        "step": step,
        "ex_id": example["id"],
        "split": example.get("split", "eval"),
        "prompt": example["prompt"],
        "completion": completion,
        "parsed": reward_breakdown["parsed"],
        "reward": {
            "correctness": reward_breakdown["correctness"],
            "path_coverage": reward_breakdown["path_coverage"],
            "path_reward": reward_breakdown["path_reward"],
            "spam_penalty": reward_breakdown["spam_penalty"],
            "total": reward_breakdown["total"]
        },
        "debug": reward_breakdown.get("debug", {})
    }


def run_eval(
    model,
    tokenizer,
    examples_path: Path,
    kg_path: Path,
    output_path: Path,
    phase: str = "base",
    step: int = 0,
    max_examples: Optional[int] = None,
    temperature: float = 0.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Run evaluation and log episodes.

    Returns aggregate metrics.
    """
    kg = load_kg(kg_path)
    entity_vocab = get_entity_vocab(kg)

    episodes = []
    total_correct = 0
    total_coverage = 0.0
    total_reward = 0.0
    n = 0

    examples = list(load_examples(examples_path))
    if max_examples:
        examples = examples[:max_examples]

    print(f"Running evaluation on {len(examples)} examples...")

    for i, example in enumerate(examples):
        if verbose or (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(examples)}...")

        # Run inference
        completion = run_inference(
            model, tokenizer, example["prompt"],
            temperature=temperature,
            verbose=False
        )

        # Compute reward
        reward = compute_reward(
            completion=completion,
            answer_star=example["answer_star"],
            path_entities=example["path_star"]["entities"],
            entity_vocab=entity_vocab
        )

        # Log episode
        episode = log_episode(example, completion, reward, phase, step)
        episodes.append(episode)

        # Aggregate metrics
        if reward["parsed"]["answer"] == example["answer_star"]:
            total_correct += 1
        total_coverage += reward["path_coverage"]
        total_reward += reward["total"]
        n += 1

    # Write episodes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    metrics = {
        "accuracy": total_correct / max(1, n),
        "avg_path_coverage": total_coverage / max(1, n),
        "avg_total_reward": total_reward / max(1, n)
    }

    print(f"\nResults for {phase}:")
    print(f"  Accuracy:      {metrics['accuracy']:.2%}")
    print(f"  Path Coverage: {metrics['avg_path_coverage']:.2%}")
    print(f"  Avg Reward:    {metrics['avg_total_reward']:.2f}")

    return metrics


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Run model inference on examples")
    parser.add_argument("--examples", type=Path, required=True, help="Path to examples JSONL")
    parser.add_argument("--kg", type=Path, required=True, help="Path to kg.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output episodes.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF ID")
    parser.add_argument("--adapter", type=Path, default=None, help="Path to LoRA adapter")
    parser.add_argument("--phase", type=str, default="base", choices=["base", "sft", "rsft"])
    parser.add_argument("--step", type=int, default=0, help="Training step number")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to process")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.adapter:
        print(f"With adapter: {args.adapter}")

    model, tokenizer = load_mlx_model(args.model, args.adapter)

    run_eval(
        model=model,
        tokenizer=tokenizer,
        examples_path=args.examples,
        kg_path=args.kg,
        output_path=args.output,
        phase=args.phase,
        step=args.step,
        max_examples=args.max_examples,
        temperature=args.temperature,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
