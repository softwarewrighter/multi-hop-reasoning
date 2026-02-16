"""
RSFT (Rejection Sampling Fine-Tuning) with Unsloth (Linux/CUDA).

Generates multiple completions per example, scores them with the reward function,
and fine-tunes on the best completions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random

from .reward import compute_reward
from .kg import load_kg, get_entity_vocab


@dataclass
class RSFTConfig:
    k_samples: int = 8  # Number of completions to generate per example
    keep_top: int = 1  # Number of best completions to keep
    temperature: float = 0.7  # Sampling temperature
    max_tokens: int = 256
    min_reward_threshold: float = 0.0  # Minimum reward to include


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4  # Lower than SFT
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    seed: int = 42


def load_examples(path: Path) -> List[Dict[str, Any]]:
    """Load examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    k: int,
    temperature: float,
    max_tokens: int
) -> List[str]:
    """Generate k completions for a prompt using Unsloth/Transformers."""
    from unsloth import FastLanguageModel

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    # Format prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    completions = []
    for _ in range(k):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # Decode only the new tokens
        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        completions.append(completion)

    # Re-enable training mode
    FastLanguageModel.for_training(model)

    return completions


def score_completions(
    completions: List[str],
    example: Dict[str, Any],
    entity_vocab: set
) -> List[Tuple[str, float, Dict]]:
    """Score completions with the reward function."""
    scored = []
    for completion in completions:
        reward = compute_reward(
            completion=completion,
            answer_star=example["answer_star"],
            path_entities=example["path_star"]["entities"],
            entity_vocab=entity_vocab
        )
        scored.append((completion, reward["total"], reward))
    return scored


def rejection_sample(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    entity_vocab: set,
    rsft_config: RSFTConfig,
    max_examples: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Generate and filter completions using rejection sampling.

    Returns list of (prompt, completion) pairs for training.
    """
    if max_examples:
        examples = examples[:max_examples]

    training_pairs = []
    stats = {"total": 0, "accepted": 0, "avg_reward": 0.0}

    print(f"Running rejection sampling on {len(examples)} examples...")
    print(f"  K samples: {rsft_config.k_samples}")
    print(f"  Keep top: {rsft_config.keep_top}")
    print(f"  Temperature: {rsft_config.temperature}")

    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(examples)}...")

        # Generate k completions
        completions = generate_completions(
            model, tokenizer,
            example["prompt"],
            rsft_config.k_samples,
            rsft_config.temperature,
            rsft_config.max_tokens
        )

        # Score all completions
        scored = score_completions(completions, example, entity_vocab)

        # Sort by reward (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top completions above threshold
        kept = []
        for completion, reward, breakdown in scored[:rsft_config.keep_top]:
            if reward >= rsft_config.min_reward_threshold:
                kept.append({
                    "instruction": example["prompt"],
                    "output": completion,
                    "reward": reward
                })
                stats["accepted"] += 1
                stats["avg_reward"] += reward

        training_pairs.extend(kept)
        stats["total"] += rsft_config.k_samples

    if stats["accepted"] > 0:
        stats["avg_reward"] /= stats["accepted"]

    print(f"\nRejection sampling complete:")
    print(f"  Generated: {stats['total']} completions")
    print(f"  Accepted: {stats['accepted']} ({100*stats['accepted']/max(1,stats['total']):.1f}%)")
    print(f"  Avg reward: {stats['avg_reward']:.3f}")

    return training_pairs


def train_rsft(
    base_model: str,
    sft_adapter_path: Path,
    examples_path: Path,
    kg_path: Path,
    output_dir: Path,
    rsft_config: Optional[RSFTConfig] = None,
    train_config: Optional[TrainConfig] = None,
    max_examples: Optional[int] = None
):
    """
    Run RSFT: rejection sampling + fine-tuning.

    Args:
        base_model: HuggingFace model ID
        sft_adapter_path: Path to SFT adapter (starting point)
        examples_path: Path to examples JSONL (use eval.jsonl for distribution matching!)
        kg_path: Path to knowledge graph
        output_dir: Where to save RSFT adapters
        rsft_config: Rejection sampling config
        train_config: Training config
        max_examples: Limit examples for testing
    """
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Unsloth dependencies required: {e}\n"
            "Install with: pip install unsloth trl datasets"
        )

    rsft_config = rsft_config or RSFTConfig()
    train_config = train_config or TrainConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RSFT Training with Unsloth")
    print("=" * 60)
    print(f"  Base model: {base_model}")
    print(f"  SFT adapter: {sft_adapter_path}")
    print(f"  Examples: {examples_path}")
    print(f"  Output: {output_dir}")

    # Load KG
    kg = load_kg(kg_path)
    entity_vocab = get_entity_vocab(kg)

    # Load examples
    examples = load_examples(examples_path)
    print(f"  Loaded {len(examples)} examples")

    # Load model with SFT adapter
    print("\nLoading model with SFT adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(sft_adapter_path),
        max_seq_length=train_config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Phase 1: Rejection sampling
    print("\n" + "=" * 60)
    print("Phase 1: Rejection Sampling")
    print("=" * 60)

    training_pairs = rejection_sample(
        examples, model, tokenizer, entity_vocab,
        rsft_config, max_examples
    )

    if not training_pairs:
        print("No training pairs generated! Check reward threshold.")
        return

    # Save rejection sampling results
    rs_data_path = output_dir / "rsft_train.jsonl"
    with open(rs_data_path, "w") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved {len(training_pairs)} training pairs to {rs_data_path}")

    # Phase 2: Fine-tune on accepted completions
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning on Accepted Completions")
    print("=" * 60)

    # Re-add LoRA for continued training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=train_config.seed,
    )

    # Create dataset
    def formatting_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(training_pairs)
    dataset = dataset.map(formatting_func, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        seed=train_config.seed,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=train_config.max_seq_length,
    )

    print("\nStarting RSFT training...")
    trainer.train()

    # Save
    print(f"\nSaving RSFT adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged
    merged_dir = output_dir / "merged"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\nRSFT training complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="RSFT training with Unsloth")
    parser.add_argument("--examples", type=Path, required=True,
                        help="Examples JSONL (use eval.jsonl for distribution matching)")
    parser.add_argument("--kg", type=Path, required=True, help="Knowledge graph JSON")
    parser.add_argument("--sft-adapter", type=Path, required=True, help="Path to SFT adapter")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--k-samples", type=int, default=8, help="Samples per example")
    parser.add_argument("--keep-top", type=int, default=1, help="Top completions to keep")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples")

    args = parser.parse_args()

    rsft_config = RSFTConfig(
        k_samples=args.k_samples,
        keep_top=args.keep_top,
        temperature=args.temperature,
    )

    train_config = TrainConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    train_rsft(
        base_model=args.model,
        sft_adapter_path=args.sft_adapter,
        examples_path=args.examples,
        kg_path=args.kg,
        output_dir=args.output,
        rsft_config=rsft_config,
        train_config=train_config,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
