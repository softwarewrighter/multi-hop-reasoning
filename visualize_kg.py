#!/usr/bin/env python3
"""
Visualize the DevOps Knowledge Graph used as "training wheels" during RSFT.

This script generates multiple visualizations showing:
1. Overall graph structure and entity types
2. Example reasoning paths (1-3 hop vs 4-5 hop)
3. How the KG serves as a reward signal during training
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def load_kg():
    """Load the knowledge graph."""
    with open('data/kg.json') as f:
        return json.load(f)


def create_kg_overview():
    """Figure 1: Knowledge graph structure overview."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Entity type distribution
    entity_types = {
        'Causes': 50,
        'Fixes': 49,
        'Diagnostics': 42,
        'Symptoms': 40,
        'Tools': 31,
    }

    colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6']
    wedges, texts, autotexts = ax1.pie(
        entity_types.values(),
        labels=entity_types.keys(),
        colors=colors,
        autopct='%1.0f%%',
        startangle=90,
        explode=(0.02, 0.02, 0.02, 0.02, 0.02),
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax1.set_title('Entity Types (212 Total)', fontsize=14, fontweight='bold')

    # Right: Relationship type distribution
    rel_types = {
        'diagnosed_by': 141,
        'caused_by': 127,
        'fixed_by': 120,
        'leads_to': 114,
        'uses_tool': 87,
        'related_to': 40,
    }

    y_pos = np.arange(len(rel_types))
    colors2 = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#95a5a6']

    bars = ax2.barh(y_pos, list(rel_types.values()), color=colors2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(rel_types.keys()), fontsize=11)
    ax2.set_xlabel('Number of Edges', fontsize=11)
    ax2.set_title('Relationship Types (629 Total)', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, rel_types.values()):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=10, fontweight='bold')

    ax2.set_xlim(0, 170)

    plt.tight_layout()
    return fig


def create_entity_hierarchy():
    """Figure 2: Entity type hierarchy and examples."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'DevOps Knowledge Graph: Entity Hierarchy',
            fontsize=16, fontweight='bold', ha='center')

    # Define boxes for each entity type
    boxes = [
        # (x, y, width, height, title, color, examples)
        (0.5, 6.5, 2.8, 2.5, 'SYMPTOMS\n(40 entities)', '#f39c12',
         ['ConnectionTimeout', 'HighLatency', 'OutOfMemory',
          'InternalServerError', 'SlowQueryPerformance']),
        (3.6, 6.5, 2.8, 2.5, 'CAUSES\n(50 entities)', '#e74c3c',
         ['MemoryLeak', 'ThreadDeadlock', 'FirewallBlocking',
          'ConfigurationError', 'NetworkPartition']),
        (6.7, 6.5, 2.8, 2.5, 'DIAGNOSTICS\n(42 entities)', '#3498db',
         ['CheckMetrics', 'AnalyzeLogs', 'RunTraceroute',
          'CheckFirewallRules', 'VerifyCertificate']),
        (1.5, 3.0, 2.8, 2.5, 'FIXES\n(49 entities)', '#27ae60',
         ['IncreaseMemory', 'RestartService', 'RotateCredentials',
          'OptimizeQuery', 'ClearCache']),
        (5.7, 3.0, 2.8, 2.5, 'TOOLS\n(31 entities)', '#9b59b6',
         ['GrafanaTool', 'KubectlTool', 'NetstatTool',
          'DockerTool', 'PrometheusTo...']),
    ]

    for x, y, w, h, title, color, examples in boxes:
        # Draw box
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=0.3)
        ax.add_patch(rect)

        # Title
        ax.text(x + w/2, y + h - 0.3, title, fontsize=11, fontweight='bold',
                ha='center', va='top', color=color)

        # Examples
        for i, ex in enumerate(examples):
            ax.text(x + w/2, y + h - 0.8 - i*0.35, f"• {ex}",
                    fontsize=9, ha='center', va='top')

    # Add relationship arrows between boxes
    arrow_style = dict(arrowstyle='->', color='gray', lw=2,
                       connectionstyle='arc3,rad=0.1')

    # Symptom -> Cause (caused_by)
    ax.annotate('', xy=(3.6, 7.5), xytext=(3.3, 7.5),
                arrowprops=arrow_style)
    ax.text(3.45, 7.9, 'caused_by', fontsize=8, ha='center', color='gray')

    # Symptom -> Diagnostic (diagnosed_by)
    ax.annotate('', xy=(6.7, 7.5), xytext=(3.3, 8.2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(5, 8.3, 'diagnosed_by', fontsize=8, ha='center', color='gray')

    # Cause -> Fix (fixed_by)
    ax.annotate('', xy=(2.9, 5.3), xytext=(4.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                               connectionstyle='arc3,rad=0.2'))
    ax.text(3.5, 5.8, 'fixed_by', fontsize=8, ha='center', color='gray')

    # Diagnostic -> Tool (uses_tool)
    ax.annotate('', xy=(7.1, 5.5), xytext=(8.1, 6.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(7.9, 6.1, 'uses_tool', fontsize=8, ha='center', color='gray')

    # Legend at bottom
    ax.text(5, 1.0, 'Relationships connect entities across types,\n'
            'forming paths that model DevOps troubleshooting workflows.',
            fontsize=10, ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def create_path_comparison():
    """Figure 3: Compare 1-3 hop (easy) vs 4-5 hop (hard) paths."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Common styling
    node_colors = {
        'symptom': '#f39c12',
        'cause': '#e74c3c',
        'diagnostic': '#3498db',
        'tool': '#9b59b6',
        'fix': '#27ae60',
    }

    def draw_path(ax, nodes, title, subtitle):
        ax.set_xlim(-0.5, len(nodes) - 0.5)
        ax.set_ylim(-1, 2)
        ax.axis('off')

        ax.text(len(nodes)/2 - 0.5, 1.7, title, fontsize=14, fontweight='bold',
                ha='center')
        ax.text(len(nodes)/2 - 0.5, 1.4, subtitle, fontsize=10, ha='center',
                style='italic', color='gray')

        for i, (name, ntype, rel) in enumerate(nodes):
            # Draw node
            color = node_colors.get(ntype, '#95a5a6')
            circle = plt.Circle((i, 0.5), 0.3, color=color, ec='black', lw=2)
            ax.add_patch(circle)

            # Node label
            ax.text(i, 0.5, name[:12] + ('...' if len(name) > 12 else ''),
                    fontsize=8, ha='center', va='center', fontweight='bold')

            # Type label below
            ax.text(i, -0.1, ntype.upper(), fontsize=7, ha='center',
                    color=color, fontweight='bold')

            # Draw arrow to next node
            if i < len(nodes) - 1:
                ax.annotate('', xy=(i + 0.65, 0.5), xytext=(i + 0.35, 0.5),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=2))
                # Relationship label
                ax.text(i + 0.5, 0.8, rel, fontsize=7, ha='center',
                        color='gray', style='italic')

    # Easy path (3 hops) - from train.jsonl
    easy_path = [
        ('BadGateway', 'symptom', ''),
        ('ServiceMeshError', 'cause', 'caused_by'),
        ('CheckServiceMesh', 'diagnostic', 'diagnosed_by'),
        ('JaegerTool', 'tool', 'uses_tool'),
    ]
    draw_path(ax1, easy_path,
              'EASY PATH: 3 Hops (Training Data)',
              'Question: "You\'re seeing BadGateway. What tool should you use?"')

    # Hard path (5 hops) - from eval.jsonl
    hard_path = [
        ('SlowQuery', 'symptom', ''),
        ('TableLock', 'cause', 'caused_by'),
        ('DBConnErr', 'symptom', 'leads_to'),
        ('Internal500', 'symptom', 'leads_to'),
        ('CheckMetrics', 'diagnostic', 'diagnosed_by'),
        ('GrafanaTool', 'tool', 'uses_tool'),
    ]
    draw_path(ax2, hard_path,
              'HARD PATH: 5 Hops (Evaluation Data)',
              'Question: "A user reports SlowQueryPerformance. What is the root cause?"')

    # Add comparison note
    fig.text(0.5, 0.02,
             'Key Insight: Evaluation questions require longer reasoning chains.\n'
             'Training RSFT on easy paths fails to teach the longer patterns needed for evaluation.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    return fig


def create_training_wheel_diagram():
    """Figure 4: How KG serves as 'training wheels' during RSFT."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Knowledge Graph as "Training Wheels"',
            fontsize=18, fontweight='bold', ha='center')
    ax.text(7, 9.0, 'How the KG provides reward signals during RSFT',
            fontsize=12, ha='center', color='gray')

    # Step 1: Model generates response
    box1 = FancyBboxPatch((0.5, 5.5), 3.5, 3, boxstyle="round,pad=0.1",
                          facecolor='#3498db', edgecolor='black',
                          linewidth=2, alpha=0.3)
    ax.add_patch(box1)
    ax.text(2.25, 8.2, 'STEP 1: Generate', fontsize=12, fontweight='bold',
            ha='center', color='#3498db')
    ax.text(2.25, 7.4, 'Model outputs:\n\n"TRACE: SlowQuery\nis caused by TableLock,\nTableLock leads to\nDatabaseError..."\n\nANSWER: B',
            fontsize=9, ha='center', va='top', family='monospace')

    # Step 2: Extract entities
    box2 = FancyBboxPatch((5, 5.5), 3.5, 3, boxstyle="round,pad=0.1",
                          facecolor='#f39c12', edgecolor='black',
                          linewidth=2, alpha=0.3)
    ax.add_patch(box2)
    ax.text(6.75, 8.2, 'STEP 2: Extract', fontsize=12, fontweight='bold',
            ha='center', color='#f39c12')
    ax.text(6.75, 7.4, 'Parse TRACE for\nKG entities:\n\n• SlowQueryPerf\n• TableLock\n• DatabaseConnErr',
            fontsize=9, ha='center', va='top')

    # Step 3: Check KG
    box3 = FancyBboxPatch((9.5, 5.5), 4, 3, boxstyle="round,pad=0.1",
                          facecolor='#27ae60', edgecolor='black',
                          linewidth=2, alpha=0.3)
    ax.add_patch(box3)
    ax.text(11.5, 8.2, 'STEP 3: Score', fontsize=12, fontweight='bold',
            ha='center', color='#27ae60')
    ax.text(11.5, 7.4, 'Compare to ground\ntruth KG path:\n\n✓ SlowQuery (hit)\n✓ TableLock (hit)\n✓ DBConnErr (hit)\n\nPath coverage: 50%',
            fontsize=9, ha='center', va='top')

    # Arrows between steps
    ax.annotate('', xy=(4.9, 7), xytext=(4.1, 7),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(9.4, 7), xytext=(8.6, 7),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Knowledge Graph illustration
    kg_box = FancyBboxPatch((4, 1), 6, 3.5, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='#9b59b6',
                            linewidth=3, alpha=0.9)
    ax.add_patch(kg_box)
    ax.text(7, 4.2, 'KNOWLEDGE GRAPH', fontsize=11, fontweight='bold',
            ha='center', color='#9b59b6')

    # Draw mini KG
    kg_nodes = [
        (4.8, 2.8, 'SlowQuery', '#f39c12'),
        (5.8, 1.8, 'TableLock', '#e74c3c'),
        (7.2, 2.8, 'DBConnErr', '#f39c12'),
        (8.2, 1.8, 'GrafanaTool', '#9b59b6'),
    ]
    for x, y, name, color in kg_nodes:
        circle = plt.Circle((x, y), 0.35, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)
        ax.text(x, y, name[:6], fontsize=7, ha='center', va='center', fontweight='bold')

    # KG edges
    ax.annotate('', xy=(5.55, 1.9), xytext=(5.05, 2.6),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(6.9, 2.6), xytext=(6.15, 1.9),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(7.9, 2.0), xytext=(7.5, 2.6),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Arrow from KG to Step 3
    ax.annotate('', xy=(11, 5.4), xytext=(8, 4.5),
               arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2,
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(10, 4.8, 'Ground\ntruth', fontsize=8, ha='center', color='#9b59b6')

    # Reward calculation box
    reward_box = FancyBboxPatch((10.5, 1), 3, 3.5, boxstyle="round,pad=0.1",
                                facecolor='#ecf0f1', edgecolor='black',
                                linewidth=2)
    ax.add_patch(reward_box)
    ax.text(12, 4.2, 'REWARD', fontsize=11, fontweight='bold', ha='center')
    ax.text(12, 3.5,
            'R_correct = +1.0\n'
            '(answer matches)\n\n'
            'R_path = +0.5\n'
            '(50% coverage)\n\n'
            'Total = +1.25',
            fontsize=9, ha='center', va='top', family='monospace')

    # Arrow from Step 3 to Reward
    ax.annotate('', xy=(11.5, 5.4), xytext=(11.5, 4.6),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    return fig


def create_hop_distribution():
    """Figure 5: Distribution of path lengths in train vs eval."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training data (1-3 hops)
    train_hops = [1, 2, 3]
    train_counts = [30, 40, 30]  # Approximate distribution
    colors1 = ['#3498db', '#2980b9', '#1a5276']

    bars1 = ax1.bar(train_hops, train_counts, color=colors1, edgecolor='black', width=0.6)
    ax1.set_xlabel('Number of Hops', fontsize=11)
    ax1.set_ylabel('Number of Examples', fontsize=11)
    ax1.set_title('Training Data (train.jsonl)\nEasy: 1-3 Hops', fontsize=13, fontweight='bold')
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylim(0, 50)

    for bar, count in zip(bars1, train_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', fontsize=11, fontweight='bold')

    # Evaluation data (4-5 hops)
    eval_hops = [4, 5]
    eval_counts = [15, 15]  # Approximate distribution
    colors2 = ['#e74c3c', '#c0392b']

    bars2 = ax2.bar(eval_hops, eval_counts, color=colors2, edgecolor='black', width=0.6)
    ax2.set_xlabel('Number of Hops', fontsize=11)
    ax2.set_ylabel('Number of Examples', fontsize=11)
    ax2.set_title('Evaluation Data (eval.jsonl)\nHard: 4-5 Hops', fontsize=13, fontweight='bold')
    ax2.set_xticks([4, 5])
    ax2.set_ylim(0, 50)

    for bar, count in zip(bars2, eval_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', fontsize=11, fontweight='bold')

    # Add annotation
    fig.text(0.5, 0.02,
             'Distribution Mismatch: Training on 1-3 hop examples does not prepare the model for 4-5 hop evaluation questions.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


def create_example_question():
    """Figure 6: Detailed example showing KG path through a question."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Complete Example: Multi-Hop Reasoning',
            fontsize=16, fontweight='bold', ha='center')

    # Question box
    q_box = FancyBboxPatch((0.5, 7), 13, 2, boxstyle="round,pad=0.1",
                           facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax.add_patch(q_box)
    ax.text(7, 8.7, 'QUESTION (5-hop, from eval.jsonl)', fontsize=11,
            fontweight='bold', ha='center')
    ax.text(7, 8.0,
            '"A user reports SlowQueryPerformance. What is the root cause?"\n'
            'A) RabbitMQAdmin   B) GrafanaTool   C) KafkaConsoleTool   D) CurlTool',
            fontsize=10, ha='center', family='monospace')

    # KG Path visualization
    ax.text(7, 6.5, 'Ground Truth Path in Knowledge Graph',
            fontsize=12, fontweight='bold', ha='center', color='#9b59b6')

    # Draw the path
    path_nodes = [
        (1, 5, 'SlowQuery\nPerformance', '#f39c12', 'symptom'),
        (3.5, 5, 'TableLock', '#e74c3c', 'cause'),
        (6, 5, 'Database\nConnError', '#f39c12', 'symptom'),
        (8.5, 5, 'Internal\nServerError', '#f39c12', 'symptom'),
        (11, 5, 'Check\nMetrics', '#3498db', 'diagnostic'),
        (13, 5, 'Grafana\nTool', '#9b59b6', 'tool'),
    ]

    rels = ['caused_by', 'leads_to', 'leads_to', 'diagnosed_by', 'uses_tool']

    for i, (x, y, label, color, ntype) in enumerate(path_nodes):
        # Node
        rect = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')
        ax.text(x, y-0.8, ntype, fontsize=7, ha='center', color=color, fontweight='bold')

        # Arrow to next
        if i < len(path_nodes) - 1:
            ax.annotate('', xy=(path_nodes[i+1][0]-0.7, y),
                       xytext=(x+0.7, y),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))
            ax.text((x + path_nodes[i+1][0])/2, y+0.35, rels[i],
                   fontsize=7, ha='center', color='gray', style='italic')

    # Model response boxes
    # Good response
    good_box = FancyBboxPatch((0.5, 1), 6, 2.5, boxstyle="round,pad=0.1",
                              facecolor='#27ae60', edgecolor='black',
                              linewidth=2, alpha=0.2)
    ax.add_patch(good_box)
    ax.text(3.5, 3.2, 'GOOD RESPONSE (RSFT Hard)', fontsize=10,
            fontweight='bold', ha='center', color='#27ae60')
    ax.text(3.5, 2.5,
            'TRACE: SlowQueryPerformance is caused\n'
            'by TableLock, and TableLock leads to\n'
            'DatabaseConnectionError...\n'
            'ANSWER: B (GrafanaTool) ✓',
            fontsize=9, ha='center', family='monospace')
    ax.text(3.5, 1.2, 'Reward: +1.25 (correct + path coverage)',
            fontsize=9, ha='center', color='#27ae60', fontweight='bold')

    # Bad response
    bad_box = FancyBboxPatch((7.5, 1), 6, 2.5, boxstyle="round,pad=0.1",
                             facecolor='#e74c3c', edgecolor='black',
                             linewidth=2, alpha=0.2)
    ax.add_patch(bad_box)
    ax.text(10.5, 3.2, 'BAD RESPONSE (RSFT Easy)', fontsize=10,
            fontweight='bold', ha='center', color='#e74c3c')
    ax.text(10.5, 2.5,
            'TRACE: SlowQueryPerformance is\n'
            'caused by TableLock...\n'
            '\n'
            'ANSWER: A (RabbitMQAdmin) ✗',
            fontsize=9, ha='center', family='monospace')
    ax.text(10.5, 1.2, 'Reward: -2.0 (incorrect answer)',
            fontsize=9, ha='center', color='#e74c3c', fontweight='bold')

    return fig


def main():
    """Generate all knowledge graph visualizations."""
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)

    print("Generating knowledge graph visualizations...")

    figures = [
        ("kg_overview.png", create_kg_overview()),
        ("kg_entity_hierarchy.png", create_entity_hierarchy()),
        ("kg_path_comparison.png", create_path_comparison()),
        ("kg_training_wheel.png", create_training_wheel_diagram()),
        ("kg_hop_distribution.png", create_hop_distribution()),
        ("kg_example_question.png", create_example_question()),
    ]

    for filename, fig in figures:
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        plt.close(fig)

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
