#!/usr/bin/env python3
"""Generate visualizations of multi-hop reasoning training results."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

# Results data from training runs
results = {
    "SmolLM-135M (run_0001)": {
        "Base": {"accuracy": 0.0, "path_coverage": 0.0, "reward": -2.0},
        "SFT": {"accuracy": 0.30, "path_coverage": 0.31, "reward": -1.01},
        "RSFT (hard)": {"accuracy": 0.75, "path_coverage": 0.33, "reward": 0.38},
    },
    "SmolLM-360M (run_360m)": {
        "Base": {"accuracy": 0.0, "path_coverage": 0.0, "reward": -2.0},
        "SFT": {"accuracy": 0.37, "path_coverage": 0.32, "reward": -0.78},
        "RSFT (easy)": {"accuracy": 0.27, "path_coverage": 0.30, "reward": -1.12},
        "RSFT (hard)": {"accuracy": 0.67, "path_coverage": 0.33, "reward": 0.13},
    }
}

# Paper comparison data
paper_results = {
    "14B Qwen (Paper)": {
        "SFT-only": {"accuracy": 0.489},
        "SFT+RL": {"accuracy": 0.568},
    }
}


def create_accuracy_progression():
    """Figure 1: Accuracy progression through training phases."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for 360M model (most complete)
    phases = ["Base\n(Untrained)", "SFT\n(Format Learning)", "RSFT (easy)\n(Mismatched)", "RSFT (hard)\n(Matched)"]
    accuracies = [0.0, 37, 27, 67]
    colors = ['#cccccc', '#3498db', '#e74c3c', '#27ae60']

    bars = ax.bar(phases, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add annotations
    ax.annotate('', xy=(2, 30), xytext=(1, 35),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1.5, 33, 'Distribution\nMismatch!', ha='center', fontsize=10, color='red', fontweight='bold')

    ax.annotate('', xy=(3, 63), xytext=(2, 30),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.5, 50, '+40%', ha='center', fontsize=12, color='green', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('SmolLM-360M Training Progression\nKey Finding: Distribution Matching is Critical', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='SFT baseline')

    plt.tight_layout()
    return fig


def create_distribution_comparison():
    """Figure 2: RSFT Easy vs Hard comparison (the key finding)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['SFT\n(Baseline)', 'RSFT (Easy)\n1-3 hop training', 'RSFT (Hard)\n4-5 hop training']
    accuracies = [37, 27, 67]
    colors = ['#3498db', '#e74c3c', '#27ae60']

    x = np.arange(len(categories))
    bars = ax.bar(x, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Add comparison annotations
    ax.annotate('', xy=(1, 25), xytext=(0, 35),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(0.5, 32, '-10%\n(WORSE!)', ha='center', fontsize=11, color='red', fontweight='bold')

    ax.annotate('', xy=(2, 63), xytext=(0, 35),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(1.3, 55, '+30%', ha='center', fontsize=12, color='green', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Accuracy on 4-5 Hop Evaluation (%)', fontsize=12)
    ax.set_title('Key Finding: Training Distribution Matters\nRSFT on Easy Examples HURTS Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 85)

    # Add explanation box
    textstr = 'Evaluation: 4-5 hop questions\nRSFT Easy trained on 1-3 hop\nRSFT Hard trained on 4-5 hop'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def create_model_comparison():
    """Figure 3: Compare both model sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['SmolLM-135M', 'SmolLM-360M']
    phases = ['Base', 'SFT', 'RSFT (hard)']

    data_135m = [0, 30, 75]
    data_360m = [0, 37, 67]

    x = np.arange(len(phases))
    width = 0.35

    bars1 = ax.bar(x - width/2, data_135m, width, label='SmolLM-135M', color='#9b59b6', edgecolor='black')
    bars2 = ax.bar(x + width/2, data_360m, width, label='SmolLM-360M', color='#2ecc71', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Size Comparison\nBoth Models Show Same Pattern', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 90)

    # Add improvement arrows
    ax.annotate('', xy=(2 - width/2, 72), xytext=(1 - width/2, 32),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.text(1.3, 55, '+45%', ha='center', fontsize=10, color='purple', fontweight='bold')

    ax.annotate('', xy=(2 + width/2, 64), xytext=(1 + width/2, 39),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.85, 52, '+30%', ha='center', fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    return fig


def create_paper_comparison():
    """Figure 4: Compare repo results with paper claims."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Paper results (14B Qwen)
    paper_sft = 48.93
    paper_rl = 56.75
    paper_improvement = paper_rl - paper_sft

    # This repo results (SmolLM-360M)
    repo_sft = 37
    repo_rsft_hard = 67
    repo_improvement = repo_rsft_hard - repo_sft

    categories = ['Paper (14B Qwen)\n5-hop Medical QA', 'This Repo (360M)\n4-5 hop DevOps QA']

    x = np.arange(len(categories))
    width = 0.35

    sft_values = [paper_sft, repo_sft]
    rsft_values = [paper_rl, repo_rsft_hard]

    bars1 = ax.bar(x - width/2, sft_values, width, label='SFT Only', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, rsft_values, width, label='SFT + RL/RSFT', color='#27ae60', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement labels
    ax.annotate(f'+{paper_improvement:.1f}%', xy=(0, 60), fontsize=14,
                color='green', fontweight='bold', ha='center')
    ax.annotate(f'+{repo_improvement}%', xy=(1, 73), fontsize=14,
                color='green', fontweight='bold', ha='center')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Comparison with Paper Results\nThis Repo Demonstrates Same Key Finding', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 85)

    # Add note box
    textstr = ('Paper: 14B params, Medical domain\n'
               'Repo: 360M params, DevOps domain\n'
               'Both show RL/RSFT improves accuracy')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def create_reward_analysis():
    """Figure 5: Reward distribution analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Average reward progression
    phases = ['Base', 'SFT', 'RSFT (easy)', 'RSFT (hard)']
    rewards = [-2.0, -0.78, -1.12, 0.13]
    colors = ['#cccccc', '#3498db', '#e74c3c', '#27ae60']

    bars = ax1.bar(phases, rewards, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        offset = 0.1 if height > 0 else -0.15
        ax1.annotate(f'{reward:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')

    ax1.set_ylabel('Average Total Reward', fontsize=11)
    ax1.set_title('Reward Progression\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylim(-2.5, 0.8)

    # Right: Reward breakdown for RSFT (hard) model
    # Based on episodes_rsft_eval.jsonl analysis
    categories = ['Correct\n(20/30)', 'Incorrect\n(10/30)']
    counts = [20, 10]
    colors = ['#27ae60', '#e74c3c']

    wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=colors,
                                        autopct='%1.0f%%', startangle=90,
                                        explode=(0.05, 0), textprops={'fontsize': 11})
    autotexts[0].set_fontweight('bold')
    autotexts[1].set_fontweight('bold')
    ax2.set_title('RSFT (Hard) Answer Distribution\n67% Accuracy on 4-5 Hop Questions', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def create_summary_dashboard():
    """Figure 6: Summary dashboard."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('Multi-Hop Reasoning: Results Summary\nDemonstrating Knowledge Graph-Guided RSFT',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Key metric boxes (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.7, '67%', fontsize=40, ha='center', va='center', fontweight='bold', color='#27ae60')
    ax1.text(0.5, 0.25, 'Best Accuracy\n(RSFT Hard)', fontsize=11, ha='center', va='center')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_facecolor('#f0fff0')
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#27ae60')
        spine.set_linewidth(2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.7, '+30%', fontsize=40, ha='center', va='center', fontweight='bold', color='#3498db')
    ax2.text(0.5, 0.25, 'Improvement\nover SFT', fontsize=11, ha='center', va='center')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.7, '-10%', fontsize=40, ha='center', va='center', fontweight='bold', color='#e74c3c')
    ax3.text(0.5, 0.25, 'RSFT Easy\nvs SFT', fontsize=11, ha='center', va='center')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # 2. Main bar chart (middle row, full width)
    ax4 = fig.add_subplot(gs[1, :])
    phases = ['Base', 'SFT', 'RSFT\n(Easy)', 'RSFT\n(Hard)']
    accuracies = [0, 37, 27, 67]
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#27ae60']

    bars = ax4.barh(phases, accuracies, color=colors, edgecolor='black', height=0.6)
    ax4.set_xlabel('Accuracy (%)', fontsize=11)
    ax4.set_title('Training Phase Comparison (SmolLM-360M)', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 80)

    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax4.annotate(f'{acc}%',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=12, fontweight='bold')

    # 3. Key findings text (bottom left)
    ax5 = fig.add_subplot(gs[2, :2])
    findings = """
KEY FINDINGS:

1. SFT teaches format compliance (TRACE + ANSWER structure)
   - Accuracy jumps from 0% to 37%

2. Distribution matching is CRITICAL for RSFT
   - Training on easy examples (1-3 hop) HURTS performance
   - RSFT Easy: 27% (worse than SFT!)
   - RSFT Hard: 67% (best result)

3. Results align with paper methodology
   - Paper: +7.8% improvement with KG-guided RL
   - This repo: +30% with distribution-matched RSFT
"""
    ax5.text(0.02, 0.95, findings, fontsize=10, va='top', ha='left',
             family='monospace', transform=ax5.transAxes)
    ax5.axis('off')

    # 4. Comparison mini-chart (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    labels = ['Paper\n(14B)', 'Repo\n(360M)']
    improvements = [7.8, 30]
    colors = ['#9b59b6', '#27ae60']

    bars = ax6.bar(labels, improvements, color=colors, edgecolor='black')
    ax6.set_ylabel('% Improvement over SFT')
    ax6.set_title('RL/RSFT Gain', fontsize=11, fontweight='bold')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax6.annotate(f'+{imp}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations."""
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)

    print("Generating visualizations...")

    # Generate each figure
    figures = [
        ("accuracy_progression.png", create_accuracy_progression()),
        ("distribution_comparison.png", create_distribution_comparison()),
        ("model_comparison.png", create_model_comparison()),
        ("paper_comparison.png", create_paper_comparison()),
        ("reward_analysis.png", create_reward_analysis()),
        ("summary_dashboard.png", create_summary_dashboard()),
    ]

    for filename, fig in figures:
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        plt.close(fig)

    print(f"\nAll visualizations saved to {output_dir}/")
    print("\nKey findings demonstrated:")
    print("  - SFT: 0% -> 37% (format learning)")
    print("  - RSFT Easy: 37% -> 27% (distribution mismatch HURTS)")
    print("  - RSFT Hard: 37% -> 67% (distribution matching WORKS)")
    print("  - Consistent with paper's KG-guided RL methodology")


if __name__ == "__main__":
    main()
