"""
Visualizations for Research Paper.

Generates figures demonstrating why the LLM pipeline is faster than grid search
for process improvement discovery.

Figures include:
1. Iterations comparison bar chart (Grid vs LLM)
2. Speedup ratio visualization
3. Combined summary panel
4. Search trajectory conceptual diagram
5. Hypothesis validation summary
6. Results table visualization
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class PaperVisualizer:
    """Generate charts for research paper."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated figures
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # style settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })

        # Colorblind-friendly palette
        self.colors = {
            'grid': '#D55E00',      # Orange-red for grid search
            'llm': '#0072B2',       # Blue for LLM
            'speedup': '#009E73',   # Green for positive results
            'calendar': '#E69F00',  # Orange for calendar adjustment
            'role': '#56B4E9',      # Light blue for role redesign
            'resource': '#CC79A7',  # Pink for resource allocation
        }

    def load_results(self, results_dir: str) -> List[Dict]:
        """
        Load results from evaluation output directories.

        Args:
            results_dir: Path to output directory containing evaluation runs

        Returns:
            List of experiment result dictionaries
        """
        results_path = Path(results_dir)
        experiments = []

        # Find all evaluation directories
        for eval_dir in sorted(results_path.glob("comprehensive_eval_*")):
            report_path = eval_dir / "evaluation_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    for result in report.get("detailed_results", []):
                        experiments.append(result)

            # Also check individual experiment result.json files
            for exp_dir in eval_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith("E"):
                    result_path = exp_dir / "result.json"
                    if result_path.exists():
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                            # Only add if not already in experiments
                            exp_id = result.get("experiment_id")
                            if not any(e.get("experiment_id") == exp_id for e in experiments):
                                experiments.append(result)

        return experiments

    def plot_iterations_comparison(
        self,
        experiments: List[Dict],
        save_name: str = "fig1_iterations_comparison"
    ) -> str:
        """
        Figure 1: Bar chart comparing iterations required.

        Shows the number of iterations Grid Search vs LLM Pipeline need
        to find the optimal improvement.
        """
        # Filter experiments with valid data
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      and e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for iterations comparison")
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(valid_exps))
        width = 0.35

        grid_iters = [e["baseline_iterations"]["mean"] for e in valid_exps]
        llm_iters = [e["llm_iterations"]["mean"] for e in valid_exps]
        labels = [f"{e['experiment_id']}\n({e.get('improvement_type', '')[:8]})" for e in valid_exps]

        bars1 = ax.bar(x - width/2, grid_iters, width,
                       label='Grid/Heuristic Search',
                       color=self.colors['grid'],
                       edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, llm_iters, width,
                       label='LLM Pipeline',
                       color=self.colors['llm'],
                       edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars1, grid_iters):
            ax.annotate(f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        for bar, val in zip(bars2, llm_iters):
            ax.annotate(f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Number of Iterations to Solution')
        ax.set_xlabel('Experiment (Improvement Type)')
        ax.set_title('Grid/Heuristic Search vs. LLM Pipeline: Iterations Required')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(grid_iters) + 1.5)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def plot_speedup_ratio(
        self,
        experiments: List[Dict],
        save_name: str = "fig2_speedup_ratio"
    ) -> str:
        """
        Figure 2: Speedup ratio achieved by LLM Pipeline.

        Visualizes how much faster the LLM pipeline is compared to grid search.
        """
        valid_exps = [e for e in experiments
                      if e.get("speedup_ratio", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for speedup visualization")
            return ""

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(valid_exps))
        speedups = [e["speedup_ratio"]["mean"] for e in valid_exps]
        labels = [e['experiment_id'] for e in valid_exps]

        # Color bars by improvement type
        bar_colors = []
        for e in valid_exps:
            imp_type = e.get("improvement_type", "")
            if "calendar" in imp_type:
                bar_colors.append(self.colors['calendar'])
            elif "role" in imp_type:
                bar_colors.append(self.colors['role'])
            else:
                bar_colors.append(self.colors['resource'])

        bars = ax.bar(x, speedups, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, speedups):
            ax.annotate(f'{val:.1f}x',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Reference lines
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Target Threshold (1.5x)')
        ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5,
                   label='Baseline (1.0x)')

        ax.set_ylabel('Speedup Ratio (Baseline Iter. / LLM Iter.)')
        ax.set_xlabel('Experiment')
        ax.set_title('LLM Pipeline Speedup Compared to Baseline Search')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(speedups) + 1.5)

        # Create legend with improvement type colors
        legend_elements = [
            mpatches.Patch(color=self.colors['calendar'], label='Calendar Adjustment'),
            mpatches.Patch(color=self.colors['role'], label='Role Redesign'),
            mpatches.Patch(color=self.colors['resource'], label='Resource Allocation'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Target (1.5x)'),
            plt.Line2D([0], [0], color='gray', linestyle='-', label='Baseline (1.0x)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def plot_combined_summary(
        self,
        experiments: List[Dict],
        save_name: str = "fig3_combined_summary"
    ) -> str:
        """
        Figure 3: Combined two-panel summary view.

        Left: Iterations comparison
        Right: Average speedup by improvement type
        """
        valid_exps = [e for e in experiments
                      if e.get("speedup_ratio", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for combined summary")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Grouped bar chart
        ax1 = axes[0]
        x = np.arange(len(valid_exps))
        width = 0.35

        grid_iters = [e.get("baseline_iterations", {}).get("mean", 0) or 0 for e in valid_exps]
        llm_iters = [e.get("llm_iterations", {}).get("mean", 0) or 0 for e in valid_exps]
        labels = [e['experiment_id'] for e in valid_exps]

        ax1.bar(x - width/2, grid_iters, width, label='Baseline', color=self.colors['grid'],
                edgecolor='black', linewidth=0.5)
        ax1.bar(x + width/2, llm_iters, width, label='LLM', color=self.colors['llm'],
                edgecolor='black', linewidth=0.5)

        ax1.set_ylabel('Iterations to Solution')
        ax1.set_xlabel('Experiment ID')
        ax1.set_title('(a) Search Iterations Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, max(grid_iters + [1]) + 1)

        # Right: Aggregate by type
        ax2 = axes[1]

        # Group by improvement type
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = []
            by_type[imp_type].append(e["speedup_ratio"]["mean"])

        types = list(by_type.keys())
        avg_speedups = [np.mean(by_type[t]) for t in types]

        # Add overall
        types.append("Overall")
        avg_speedups.append(np.mean([e["speedup_ratio"]["mean"] for e in valid_exps]))

        # Prettify type names
        type_labels = [t.replace("_", "\n").title() for t in types]

        bar_colors = []
        for t in types:
            if "calendar" in t.lower():
                bar_colors.append(self.colors['calendar'])
            elif "role" in t.lower():
                bar_colors.append(self.colors['role'])
            elif "resource" in t.lower():
                bar_colors.append(self.colors['resource'])
            else:
                bar_colors.append(self.colors['speedup'])

        bars = ax2.bar(type_labels, avg_speedups, color=bar_colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, avg_speedups):
            ax2.annotate(f'{val:.2f}x',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.axhline(y=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Target (1.5x)')
        ax2.set_ylabel('Average Speedup Ratio')
        ax2.set_xlabel('Improvement Type')
        ax2.set_title('(b) Average Speedup by Type')
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, max(avg_speedups) + 1.5)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def plot_search_trajectory(
        self,
        save_name: str = "fig4_search_trajectory"
    ) -> str:
        """
        Figure 4: Conceptual visualization of search trajectories.

        Illustrates the fundamental difference between grid search (systematic)
        and LLM pipeline (targeted hypothesis-driven).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Grid Search trajectory
        ax1 = axes[0]
        grid_steps = range(1, 7)
        objective_grid = [0.2, 0.35, 0.5, 0.65, 0.85, 1.0]

        ax1.plot(grid_steps, objective_grid, 'o-', color=self.colors['grid'],
                 linewidth=2, markersize=8, label='Grid Search Path')
        ax1.scatter([6], [1.0], s=200, c='green', marker='*', zorder=5,
                    label='Optimal Solution')
        ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5,
                    label='Acceptance Threshold')

        # Add annotations for what each step explores
        ax1.annotate('Systematic\nexploration', xy=(3.5, 0.4), fontsize=9, ha='center',
                    style='italic', color='gray')

        ax1.set_xlabel('Search Iteration')
        ax1.set_ylabel('Solution Quality (Normalized)')
        ax1.set_title('(a) Grid Search: Systematic Exploration')
        ax1.legend(loc='lower right')
        ax1.set_xlim(0.5, 6.5)
        ax1.set_ylim(0, 1.15)
        ax1.set_xticks(range(1, 7))

        # Right: LLM trajectory
        ax2 = axes[1]
        ax2.plot([0, 1], [0, 1.0], 'o-', color=self.colors['llm'],
                 linewidth=2, markersize=8, label='LLM Pipeline Path')
        ax2.scatter([1], [1.0], s=200, c='green', marker='*', zorder=5,
                    label='Optimal Solution')
        ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5,
                    label='Acceptance Threshold')

        # Annotation explaining the jump
        ax2.annotate('Bottleneck Analysis\n+ Domain Knowledge\n→ Targeted Fix',
                    xy=(0.5, 0.5), fontsize=10, ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('Search Iteration')
        ax2.set_ylabel('Solution Quality (Normalized)')
        ax2.set_title('(b) LLM Pipeline: Targeted Intervention')
        ax2.legend(loc='lower right')
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(0, 1.15)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Start', '1'])

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def plot_hypothesis_validation(
        self,
        experiments: List[Dict],
        save_name: str = "fig5_hypothesis_validation"
    ) -> str:
        """
        Figure 5: Hypothesis validation summary.

        Shows research hypotheses and whether they are supported by data.
        """
        valid_exps = [e for e in experiments
                      if e.get("speedup_ratio", {}).get("mean") is not None]

        # Calculate hypothesis metrics
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"speedups": [], "success": []}
            by_type[imp_type]["speedups"].append(e["speedup_ratio"]["mean"])
            by_type[imp_type]["success"].append(e.get("success_rates", {}).get("llm", 0))

        # Define hypotheses
        hypotheses = []

        # H1: Resource speedup
        resource_speedup = np.mean(by_type.get("resource_allocation", {}).get("speedups", [])) if "resource_allocation" in by_type else None
        hypotheses.append({
            'name': 'H1: Resource Speedup\n(>1.5x)',
            'threshold': 1.5,
            'observed': resource_speedup,
            'supported': resource_speedup and resource_speedup > 1.5 if resource_speedup else None
        })

        # H2: Calendar speedup
        calendar_speedup = np.mean(by_type.get("calendar_adjustment", {}).get("speedups", [])) if "calendar_adjustment" in by_type else None
        hypotheses.append({
            'name': 'H2: Calendar Speedup\n(>1.5x)',
            'threshold': 1.5,
            'observed': calendar_speedup,
            'supported': calendar_speedup and calendar_speedup > 1.5 if calendar_speedup else None
        })

        # H3: Bottleneck identification
        overall_success = np.mean([e.get("success_rates", {}).get("llm", 0) for e in valid_exps])
        hypotheses.append({
            'name': 'H3: Bottleneck ID\n(>80%)',
            'threshold': 0.8,
            'observed': overall_success,
            'supported': overall_success > 0.8
        })

        fig, ax = plt.subplots(figsize=(10, 5))

        y_pos = np.arange(len(hypotheses))

        for i, h in enumerate(hypotheses):
            if h['observed'] is not None:
                # Threshold bar (gray)
                ax.barh(i, h['threshold'], height=0.4, color='lightgray',
                        edgecolor='black', label='Threshold' if i == 0 else '')
                # Observed bar
                color = self.colors['speedup'] if h['supported'] else 'red'
                ax.barh(i, h['observed'], height=0.4, color=color, alpha=0.7,
                        edgecolor='black', label='Observed' if i == 0 else '')

                # Value annotation
                ax.annotate(f'{h["observed"]:.2f}', xy=(h['observed'] + 0.1, i),
                           va='center', fontsize=10, fontweight='bold')

                # Support indicator (use text instead of Unicode for font compatibility)
                symbol = 'YES' if h['supported'] else 'NO'
                symbol_color = 'green' if h['supported'] else 'red'
                ax.annotate(symbol, xy=(7, i), va='center', ha='center',
                           fontsize=11, fontweight='bold', color=symbol_color)
            else:
                ax.barh(i, h['threshold'], height=0.4, color='lightgray',
                        edgecolor='black', alpha=0.5)
                ax.annotate('N/A', xy=(h['threshold'] + 0.1, i), va='center',
                           fontsize=10, style='italic')
                ax.annotate('—', xy=(7, i), va='center', ha='center',
                           fontsize=16, color='gray')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([h['name'] for h in hypotheses])
        ax.set_xlabel('Value')
        ax.set_title('Research Hypotheses: Threshold vs. Observed Performance')
        ax.set_xlim(0, 8)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Threshold'),
            mpatches.Patch(facecolor=self.colors['speedup'], edgecolor='black',
                          alpha=0.7, label='Observed (Supported)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        # Column header
        ax.annotate('Supported?', xy=(7, len(hypotheses) - 0.3), ha='center',
                   fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def plot_results_table(
        self,
        experiments: List[Dict],
        save_name: str = "fig6_results_table"
    ) -> str:
        """
        Figure 6: Table-style visualization of all results.

        Good for paper appendix or quick reference.
        """
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      or e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for results table")
            return ""

        fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(valid_exps)))
        ax.axis('off')

        headers = ['Exp', 'Log', 'Type', 'Fault', 'Baseline', 'LLM', 'Speedup', 'Success']
        data = []

        for e in valid_exps:
            baseline = e.get("baseline_iterations", {}).get("mean")
            llm = e.get("llm_iterations", {}).get("mean")
            speedup = e.get("speedup_ratio", {}).get("mean")
            success = e.get("success_rates", {}).get("llm", 0)

            data.append([
                e.get("experiment_id", ""),
                e.get("log_name", "")[:12],
                e.get("improvement_type", "")[:10],
                e.get("fault_description", "")[:20],
                f'{baseline:.0f}' if baseline else 'N/A',
                f'{llm:.0f}' if llm else 'N/A',
                f'{speedup:.1f}x' if speedup else 'N/A',
                'Yes' if success >= 0.8 else 'No'  # Use text instead of Unicode
            ])

        table = ax.table(
            cellText=data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colColours=['#E6E6FA'] * len(headers)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_text_props(fontweight='bold')

        # Color code speedup
        for i, row in enumerate(data):
            speedup_str = row[6]
            if speedup_str != 'N/A':
                speedup = float(speedup_str.replace('x', ''))
                if speedup > 1.5:
                    table[(i + 1, 6)].set_facecolor('#90EE90')  # Light green
                elif speedup >= 1.0:
                    table[(i + 1, 6)].set_facecolor('#FFFACD')  # Light yellow

        plt.title('Comprehensive Evaluation Results Summary', fontsize=14,
                  fontweight='bold', y=0.98)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def generate_latex_table(self, experiments: List[Dict]) -> str:
        """Generate LaTeX table for paper."""
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      or e.get("llm_iterations", {}).get("mean") is not None]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparative Evaluation: Grid Search vs. LLM Pipeline}",
            r"\label{tab:results}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\textbf{ID} & \textbf{Type} & \textbf{Baseline} & \textbf{LLM} & \textbf{Speedup} & \textbf{Success} \\",
            r"\midrule",
        ]

        for e in valid_exps:
            exp_id = e.get("experiment_id", "")
            imp_type = e.get("improvement_type", "")[:10].replace("_", " ").title()
            baseline = e.get("baseline_iterations", {}).get("mean")
            llm = e.get("llm_iterations", {}).get("mean")
            speedup = e.get("speedup_ratio", {}).get("mean")
            success = e.get("success_rates", {}).get("llm", 0)

            baseline_str = f'{baseline:.0f}' if baseline else 'N/A'
            llm_str = f'{llm:.0f}' if llm else 'N/A'
            speedup_str = f'\\textbf{{{speedup:.1f}$\\times$}}' if speedup and speedup > 1.5 else (f'{speedup:.1f}$\\times$' if speedup else 'N/A')
            success_str = r'\checkmark' if success >= 0.8 else r'$\times$'

            lines.append(f"{exp_id} & {imp_type} & {baseline_str} & {llm_str} & {speedup_str} & {success_str} \\\\")

        # Add summary row
        avg_speedups = [e["speedup_ratio"]["mean"] for e in valid_exps if e.get("speedup_ratio", {}).get("mean")]
        if avg_speedups:
            avg = np.mean(avg_speedups)
            lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Average}}}} & & \\textbf{{{avg:.2f}$\\times$}} & \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        latex_path = self.figures_dir / "table_results.tex"
        with open(latex_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Generated: {latex_path}")
        return str(latex_path)

    def generate_all_figures(self, results_dir: str) -> List[str]:
        """
        Generate all figures.

        Args:
            results_dir: Path to output directory with evaluation results

        Returns:
            List of generated figure paths
        """
        print(f"\nLoading results from: {results_dir}")
        experiments = self.load_results(results_dir)
        print(f"Found {len(experiments)} experiments")

        if not experiments:
            print("No experiments found!")
            return []

        # Print summary of what we found
        for e in experiments:
            exp_id = e.get("experiment_id", "?")
            imp_type = e.get("improvement_type", "?")
            baseline = e.get("baseline_iterations", {}).get("mean")
            llm = e.get("llm_iterations", {}).get("mean")
            speedup = e.get("speedup_ratio", {}).get("mean")
            print(f"  {exp_id}: {imp_type} - Baseline={baseline}, LLM={llm}, Speedup={speedup}")

        print("\nGenerating figures...")
        figures = []

        # Generate all figures
        fig = self.plot_iterations_comparison(experiments)
        if fig:
            figures.append(fig)

        fig = self.plot_speedup_ratio(experiments)
        if fig:
            figures.append(fig)

        fig = self.plot_combined_summary(experiments)
        if fig:
            figures.append(fig)

        fig = self.plot_search_trajectory()
        if fig:
            figures.append(fig)

        fig = self.plot_hypothesis_validation(experiments)
        if fig:
            figures.append(fig)

        fig = self.plot_results_table(experiments)
        if fig:
            figures.append(fig)

        # Generate LaTeX table
        latex = self.generate_latex_table(experiments)
        if latex:
            figures.append(latex)

        print(f"\nGenerated {len(figures)} files in {self.figures_dir}")
        return figures


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate figures from evaluation results"
    )
    parser.add_argument(
        "--results-dir", "-r",
        default="output",
        help="Directory containing evaluation results (default: output)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for figures (default: results-dir)"
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    visualizer = PaperVisualizer(output_dir)
    figures = visualizer.generate_all_figures(args.results_dir)

    if figures:
        print(f"\nSuccess! Generated {len(figures)} files.")
    else:
        print("\nNo figures generated. Check that results exist in the specified directory.")


if __name__ == "__main__":
    main()
