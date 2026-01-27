"""
Evaluation and metrics computation.

Generates metrics.json from episodes.jsonl.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def load_episodes(path: Path) -> List[Dict[str, Any]]:
    """Load all episodes from JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def compute_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from episodes.

    Groups by phase and split, computes accuracy/coverage/reward.
    """
    # Group by (phase, split)
    groups = defaultdict(list)
    for ep in episodes:
        key = (ep["phase"], ep.get("split", "eval"))
        groups[key].append(ep)

    metrics = {"splits": {}}

    for (phase, split), eps in groups.items():
        n = len(eps)
        if n == 0:
            continue

        correct = sum(
            1 for ep in eps
            if ep["parsed"]["valid_format"] and
               ep["parsed"]["answer"] == ep.get("answer_star", "")
        )

        # For episodes we need the example answer - use reward correctness
        correct = sum(1 for ep in eps if ep["reward"]["correctness"] > 0)

        avg_coverage = sum(ep["reward"]["path_coverage"] for ep in eps) / n
        avg_reward = sum(ep["reward"]["total"] for ep in eps) / n

        if split not in metrics["splits"]:
            metrics["splits"][split] = {}

        metrics["splits"][split][phase] = {
            "accuracy": correct / n,
            "avg_path_coverage": avg_coverage,
            "avg_total_reward": avg_reward,
            "n_examples": n
        }

    return metrics


def compute_curves(episodes: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Compute training curves from episodes.

    Groups by step and computes running metrics.
    """
    # Group by step
    by_step = defaultdict(list)
    for ep in episodes:
        by_step[ep["step"]].append(ep)

    steps = sorted(by_step.keys())
    accuracy_curve = []
    coverage_curve = []

    for step in steps:
        eps = by_step[step]
        n = len(eps)

        correct = sum(1 for ep in eps if ep["reward"]["correctness"] > 0)
        accuracy_curve.append(correct / max(1, n))

        avg_cov = sum(ep["reward"]["path_coverage"] for ep in eps) / max(1, n)
        coverage_curve.append(avg_cov)

    return {
        "steps": steps,
        "eval_accuracy": accuracy_curve,
        "eval_path_coverage": coverage_curve
    }


def generate_metrics(
    run_dir: Path,
    run_id: str = None
) -> Dict[str, Any]:
    """
    Generate metrics.json for a run.

    Reads episodes.jsonl, computes metrics, writes metrics.json.
    """
    episodes_path = run_dir / "episodes.jsonl"

    if not episodes_path.exists():
        return {"error": "No episodes.jsonl found"}

    episodes = load_episodes(episodes_path)

    metrics = compute_metrics(episodes)
    metrics["curves"] = compute_curves(episodes)
    metrics["run_id"] = run_id or run_dir.name

    # Write metrics.json
    output_path = run_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print a human-readable summary of metrics."""
    print(f"\nRun: {metrics.get('run_id', 'unknown')}")
    print("=" * 50)

    for split, phases in metrics.get("splits", {}).items():
        print(f"\n{split.upper()}:")
        for phase, m in phases.items():
            print(f"  {phase}:")
            print(f"    Accuracy:      {m['accuracy']:.2%}")
            print(f"    Path Coverage: {m['avg_path_coverage']:.2%}")
            print(f"    Avg Reward:    {m['avg_total_reward']:.2f}")


def compare_phases(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare metrics across phases (base -> sft -> rsft).

    Returns improvement percentages.
    """
    comparisons = {}

    for split, phases in metrics.get("splits", {}).items():
        if split not in comparisons:
            comparisons[split] = {}

        phase_order = ["base", "sft", "rsft"]
        available_phases = [p for p in phase_order if p in phases]

        for i, phase in enumerate(available_phases):
            if i == 0:
                continue

            prev_phase = available_phases[i - 1]
            curr = phases[phase]
            prev = phases[prev_phase]

            acc_delta = curr["accuracy"] - prev["accuracy"]
            cov_delta = curr["avg_path_coverage"] - prev["avg_path_coverage"]
            rew_delta = curr["avg_total_reward"] - prev["avg_total_reward"]

            comparisons[split][f"{prev_phase}_to_{phase}"] = {
                "accuracy_delta": acc_delta,
                "coverage_delta": cov_delta,
                "reward_delta": rew_delta,
            }

    return comparisons


def main():
    """Command-line interface for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate metrics from episodes")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing episodes.jsonl")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")

    args = parser.parse_args()

    print(f"Generating metrics from: {args.run_dir}")

    metrics = generate_metrics(args.run_dir, args.run_id)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print_summary(metrics)

    # Print comparisons if multiple phases available
    comparisons = compare_phases(metrics)
    if comparisons:
        print("\n" + "=" * 50)
        print("Phase Comparisons:")
        for split, comps in comparisons.items():
            for comp_name, deltas in comps.items():
                print(f"\n  {split} {comp_name}:")
                print(f"    Accuracy change:  {deltas['accuracy_delta']:+.2%}")
                print(f"    Coverage change:  {deltas['coverage_delta']:+.2%}")
                print(f"    Reward change:    {deltas['reward_delta']:+.2f}")

    print(f"\nMetrics saved to: {args.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
