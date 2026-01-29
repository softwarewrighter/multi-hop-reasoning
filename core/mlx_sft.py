"""
Supervised Fine-Tuning with MLX and LoRA.

Uses mlx-lm for LoRA training on reference traces.
"""

import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Default configuration
DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"

DEFAULT_LORA_CONFIG = {
    "lora_layers": 16,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

DEFAULT_TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "num_iters": 1000,
    "batch_size": 4,
    "grad_checkpoint": True,
    "seed": 42,
}


def prepare_sft_data(
    examples_path: Path,
    output_path: Path,
    use_chat_format: bool = True
) -> int:
    """
    Convert train.jsonl to MLX LoRA training format.

    MLX-LM expects JSONL with "text" field containing the full conversation.
    For chat models, we format as a conversation.

    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(examples_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            example = json.loads(line)

            prompt = example["prompt"]
            reference = example["ref"]["trace"]

            if use_chat_format:
                # Format as chat conversation for instruct models
                # SmolLM uses a simple format
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{reference}<|im_end|>"
            else:
                # Simple prompt-completion format
                text = f"{prompt}\n{reference}"

            f_out.write(json.dumps({"text": text}) + "\n")
            count += 1

    print(f"Prepared {count} examples -> {output_path}")
    return count


def create_lora_config(
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """Create LoRA configuration file for MLX-LM."""
    lora_config = DEFAULT_LORA_CONFIG.copy()
    if config:
        lora_config.update(config)

    config_path = output_dir / "lora_config.yaml"

    # Write YAML config
    with open(config_path, "w") as f:
        for key, value in lora_config.items():
            f.write(f"{key}: {value}\n")

    return config_path


def train_lora(
    base_model: str,
    train_data: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    valid_data: Optional[Path] = None,
    resume: bool = False
) -> Path:
    """
    Train LoRA adapters using MLX-LM.

    This uses the mlx_lm.lora module for training.

    Args:
        base_model: HuggingFace model ID or local path
        train_data: Path to training JSONL (with "text" field)
        output_dir: Where to save adapters
        config: Training configuration (overrides defaults)
        valid_data: Optional validation data
        resume: Resume from existing adapter

    Returns:
        Path to trained adapter directory
    """
    # Merge configs
    train_config = DEFAULT_TRAIN_CONFIG.copy()
    lora_config = DEFAULT_LORA_CONFIG.copy()

    if config:
        for key in DEFAULT_TRAIN_CONFIG:
            if key in config:
                train_config[key] = config[key]
        for key in DEFAULT_LORA_CONFIG:
            if key in config:
                lora_config[key] = config[key]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    full_config = {
        "base_model": base_model,
        "train_config": train_config,
        "lora_config": lora_config,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    # Try to use mlx_lm directly if available
    try:
        from mlx_lm import lora as mlx_lora
        from mlx_lm import load

        print(f"Training LoRA with MLX-LM...")
        print(f"  Model: {base_model}")
        print(f"  Train data: {train_data}")
        print(f"  Output: {output_dir}")
        print(f"  Config: {train_config}")

        # Load the model
        model, tokenizer = load(base_model)

        # Run training using mlx_lm CLI
        # This is the most reliable way to use mlx_lm for training
        import sys
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", base_model,
            "--train",
            "--data", str(train_data.parent),
            "--adapter-path", str(output_dir),
            "--iters", str(train_config["num_iters"]),
            "--batch-size", str(train_config["batch_size"]),
            "--learning-rate", str(train_config["learning_rate"]),
            "--num-layers", str(lora_config["lora_layers"]),
            "--seed", str(train_config["seed"]),
            "--steps-per-report", "50",
            "--save-every", str(train_config["num_iters"]),
        ]

        if train_config.get("grad_checkpoint"):
            cmd.append("--grad-checkpoint")

        if resume and (output_dir / "adapters.safetensors").exists():
            cmd.extend(["--resume-adapter-file", str(output_dir / "adapters.safetensors")])

        print(f"\nRunning: {' '.join(cmd)}")

        # Prepare data directory structure expected by mlx_lm
        data_dir = train_data.parent
        train_file = data_dir / "train.jsonl"
        if train_data != train_file:
            import shutil
            shutil.copy(train_data, train_file)

        if valid_data:
            valid_file = data_dir / "valid.jsonl"
            if valid_data != valid_file:
                import shutil
                shutil.copy(valid_data, valid_file)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Training stderr: {result.stderr}")
            # Fall back to simpler training approach
            return train_lora_simple(base_model, train_data, output_dir, train_config, lora_config)

        print(result.stdout)
        print(f"\nLoRA adapters saved to: {output_dir}")

        return output_dir

    except Exception as e:
        print(f"Error with mlx_lm.lora: {e}")
        print("Falling back to simple training...")
        return train_lora_simple(base_model, train_data, output_dir, train_config, lora_config)


def train_lora_simple(
    base_model: str,
    train_data: Path,
    output_dir: Path,
    train_config: Dict[str, Any],
    lora_config: Dict[str, Any]
) -> Path:
    """
    Simple LoRA training loop with MLX.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as e:
        raise ImportError(f"MLX-LM required for training: {e}")

    print(f"Training LoRA (simple mode)...")
    print(f"  Model: {base_model}")
    print(f"  Train data: {train_data}")

    # Load model
    model, tokenizer = load(base_model)

    # Convert to LoRA - this adds LoRA layers to the model
    # mlx-lm's linear_to_lora_layers adds trainable LoRA parameters
    linear_to_lora_layers(
        model,
        lora_config["lora_layers"],
        lora_config.get("lora_parameters", {
            "rank": lora_config["lora_rank"],
            "alpha": lora_config["lora_alpha"],
            "dropout": lora_config.get("lora_dropout", 0.0),
            "scale": lora_config["lora_alpha"] / lora_config["lora_rank"],
        })
    )

    # Put model in training mode
    model.train()

    # Load and tokenize training data
    train_tokens = []
    with open(train_data) as f:
        for line in f:
            item = json.loads(line)
            tokens = tokenizer.encode(item["text"])
            if len(tokens) > 512:
                tokens = tokens[:512]
            train_tokens.append(tokens)
    print(f"  Loaded {len(train_tokens)} training examples")

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=train_config["learning_rate"])

    num_iters = train_config["num_iters"]
    batch_size = train_config["batch_size"]

    # Loss function - takes model as first argument (required by nn.value_and_grad)
    def loss_fn(model, inputs, targets, lengths):
        logits = model(inputs)
        # Create mask for valid positions
        steps = mx.arange(targets.shape[1])
        mask = steps[None, :] < lengths[:, None]
        # Compute cross entropy loss
        ce = nn.losses.cross_entropy(logits, targets)
        ce = (ce * mask).sum() / mask.sum()
        return ce

    # Gradient function - compute gradients w.r.t. model's trainable parameters
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nStarting training for {num_iters} iterations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(num_iters):
        # Sample batch indices
        indices = np.random.choice(len(train_tokens), size=batch_size, replace=True)

        # Prepare batch with padding
        batch_tokens = [train_tokens[i] for i in indices]
        max_len = max(len(t) for t in batch_tokens)
        lengths = mx.array([len(t) - 1 for t in batch_tokens])  # -1 because targets are shifted

        # Pad sequences
        padded = np.zeros((batch_size, max_len), dtype=np.int32)
        for i, tokens in enumerate(batch_tokens):
            padded[i, :len(tokens)] = tokens

        inputs = mx.array(padded[:, :-1])
        targets = mx.array(padded[:, 1:])

        # Forward and backward pass
        loss, grads = loss_value_and_grad(model, inputs, targets, lengths)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_iters}, Loss: {loss.item():.4f}")

    # Save adapter weights
    print(f"\nSaving adapters...")
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.trainable_parameters()))

    # Save using mlx's save_safetensors
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), weights)

    # Save adapter config in mlx-lm format
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": lora_config["lora_layers"],
        "lora_parameters": {
            "rank": lora_config["lora_rank"],
            "alpha": lora_config["lora_alpha"],
            "dropout": lora_config.get("lora_dropout", 0.0),
            "scale": lora_config["lora_alpha"] / lora_config["lora_rank"],
        }
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"LoRA adapters saved to: {output_dir}")
    return output_dir


def load_model_with_lora(
    base_model: str,
    adapter_path: Path
) -> Tuple[Any, Any]:
    """
    Load base model with LoRA adapters for inference.

    Returns:
        (model, tokenizer) tuple
    """
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError("mlx-lm is required. Install with: pip install mlx-lm")

    adapter_path = Path(adapter_path)

    if adapter_path.exists():
        print(f"Loading model with adapter from: {adapter_path}")
        model, tokenizer = load(base_model, adapter_path=str(adapter_path))
    else:
        print(f"Adapter not found at {adapter_path}, loading base model")
        model, tokenizer = load(base_model)

    return model, tokenizer


def main():
    """Command-line interface for SFT training."""
    parser = argparse.ArgumentParser(description="Train LoRA adapters with MLX")
    parser.add_argument("--train", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for adapters")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from existing adapter")

    args = parser.parse_args()

    # Prepare training data
    sft_data_path = args.output / "sft_train.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)

    print("Preparing SFT training data...")
    prepare_sft_data(args.train, sft_data_path)

    # Train
    config = {
        "num_iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "lora_layers": args.lora_layers,
        "seed": args.seed,
    }

    train_lora(
        base_model=args.model,
        train_data=sft_data_path,
        output_dir=args.output,
        config=config,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
