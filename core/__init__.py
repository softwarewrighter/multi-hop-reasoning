# Core library for KG-based reward training
"""
Multi-hop reasoning demo with MLX-first KG reward training.

Modules:
- kg: Knowledge graph loading and path sampling
- dataset: MCQ dataset generation from KG paths
- reward: Reward computation for model completions
- infer: Model inference and episode logging
- mlx_sft: Supervised Fine-Tuning with MLX and LoRA
- rsft: Rejection Sampling Fine-Tuning (RL-lite)
- eval: Evaluation and metrics computation
"""

from .kg import load_kg, sample_path, get_entity_vocab
from .dataset import generate_mcq, generate_distractors, generate_dataset
from .reward import compute_reward, parse_completion
from .eval import generate_metrics, print_summary

__version__ = "0.1.0"

__all__ = [
    # KG
    "load_kg",
    "sample_path",
    "get_entity_vocab",
    # Dataset
    "generate_mcq",
    "generate_distractors",
    "generate_dataset",
    # Reward
    "compute_reward",
    "parse_completion",
    # Eval
    "generate_metrics",
    "print_summary",
]
