"""
SFT training with Unsloth (Linux/CUDA).

Supervised fine-tuning using Unsloth's optimized LoRA implementation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    seed: int = 42


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj")


def load_training_data(train_path: Path) -> List[Dict[str, Any]]:
    """Load training examples from JSONL."""
    examples = []
    with open(train_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_for_sft(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format examples for SFT training.

    Converts MCQ examples to chat format with expected TRACE + ANSWER output.
    """
    formatted = []
    for ex in examples:
        # Build the expected output from reference trace
        trace = ex.get("trace_star", "")
        answer = ex.get("answer_star", "")

        expected_output = f"TRACE: {trace}\nANSWER: {answer}"

        formatted.append({
            "instruction": ex["prompt"],
            "output": expected_output
        })

    return formatted


def train_sft(
    base_model: str,
    train_path: Path,
    output_dir: Path,
    train_config: Optional[TrainConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    resume: bool = False
):
    """
    Train SFT LoRA adapters using Unsloth.

    Args:
        base_model: HuggingFace model ID (e.g., "HuggingFaceTB/SmolLM-360M-Instruct")
        train_path: Path to training JSONL
        output_dir: Where to save adapters
        train_config: Training hyperparameters
        lora_config: LoRA configuration
        resume: Whether to resume from existing adapter
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

    train_config = train_config or TrainConfig()
    lora_config = lora_config or LoraConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training SFT with Unsloth")
    print(f"  Model: {base_model}")
    print(f"  Train data: {train_path}")
    print(f"  Output: {output_dir}")

    # Load and format training data
    print("\nPreparing training data...")
    raw_examples = load_training_data(train_path)
    formatted_examples = format_for_sft(raw_examples)
    print(f"  Loaded {len(formatted_examples)} examples")

    # Save formatted data for inspection
    sft_data_path = output_dir / "sft_train.jsonl"
    with open(sft_data_path, "w") as f:
        for ex in formatted_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved formatted data to {sft_data_path}")

    # Load model with Unsloth optimizations
    print("\nLoading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=train_config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=list(lora_config.target_modules),
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=train_config.seed,
    )

    # Create dataset
    def formatting_func(examples):
        """Format examples for the trainer."""
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            # Use chat template if available
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

    dataset = Dataset.from_list(formatted_examples)
    dataset = dataset.map(formatting_func, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        seed=train_config.seed,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=train_config.max_seq_length,
    )

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume)

    # Save the LoRA adapters
    print(f"\nSaving adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save in merged format for easier inference
    merged_dir = output_dir / "merged"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("SFT training complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth")
    parser.add_argument("--train", type=Path, required=True, help="Training data JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    train_config = TrainConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    lora_config = LoraConfig(r=args.lora_rank)

    train_sft(
        base_model=args.model,
        train_path=args.train,
        output_dir=args.output,
        train_config=train_config,
        lora_config=lora_config,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
