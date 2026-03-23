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

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np


class PaperVisualizer:
    """Generate charts for research paper."""

    # Custom ordering: Calendar Adjustment first, then alphabetical
    TYPE_ORDER = ['calendar_adjustment', 'resource_allocation', 'shift_responsibility']

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
            'role': '#56B4E9',      # Light blue for shift responsibility
            'resource': '#CC79A7',  # Pink for resource allocation
        }

    def _sort_types(self, types: List[str]) -> List[str]:
        """Sort improvement types: Calendar Adjustment first, then alphabetical."""
        def sort_key(t):
            try:
                return (self.TYPE_ORDER.index(t), t)
            except ValueError:
                return (len(self.TYPE_ORDER), t)  # Unknown types go last, alphabetically
        return sorted(types, key=sort_key)

    def _format_type_label(self, imp_type: str) -> str:
        """Format improvement type for display labels."""
        if imp_type == "calendar_adjustment":
            return "Calendar\nUpdate"
        return imp_type.replace("_", "\n").title()

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
        for eval_dir in sorted(results_path.glob("impr_type_eval_*")):
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

        Shows the average number of iterations Grid Search vs LLM Pipeline need
        to find the optimal improvement, grouped by evaluation type.
        """
        # Filter experiments with valid data
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      and e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for iterations comparison")
            return ""

        # Group by improvement type and compute averages
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": []}
            by_type[imp_type]["grid"].append(e["baseline_iterations"]["mean"])
            by_type[imp_type]["llm"].append(e["llm_iterations"]["mean"])

        types = self._sort_types(list(by_type.keys()))
        grid_iters = [np.mean(by_type[t]["grid"]) for t in types]
        llm_iters = [np.mean(by_type[t]["llm"]) for t in types]
        labels = [self._format_type_label(t) for t in types]

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(types))
        width = 0.35

        bars1 = ax.bar(x - width/2, grid_iters, width,
                       label='Grid',
                       color=self.colors['grid'],
                       edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, llm_iters, width,
                       label='Framework',
                       color=self.colors['llm'],
                       edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars1, grid_iters):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        for bar, val in zip(bars2, llm_iters):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Avg # Iterations')
        ax.set_xlabel('Improvement Type')
        ax.set_title('Average Iterations to Solution by Improvement Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, max(grid_iters) + 1.5)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(fig_path)
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

        Visualizes aggregate speedup by evaluation type compared to grid search.
        Speedup = sum(grid_iterations) / sum(llm_iterations) for accurate measurement.
        """
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      and e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for speedup visualization")
            return ""

        # Group by improvement type and compute aggregate speedup
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": []}
            by_type[imp_type]["grid"].append(e["baseline_iterations"]["mean"])
            by_type[imp_type]["llm"].append(e["llm_iterations"]["mean"])

        types = self._sort_types(list(by_type.keys()))
        # Compute speedup as sum(grid) / sum(llm) for each type
        speedups = [sum(by_type[t]["grid"]) / sum(by_type[t]["llm"]) for t in types]
        labels = [self._format_type_label(t) for t in types]

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(types))

        # Color bars by improvement type
        bar_colors = []
        for t in types:
            if "calendar" in t:
                bar_colors.append(self.colors['calendar'])
            elif "shift" in t or "responsibility" in t:
                bar_colors.append(self.colors['role'])
            else:
                bar_colors.append(self.colors['resource'])

        bars = ax.bar(x, speedups, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, speedups):
            ax.annotate(f'{val:.2f}x',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Speedup (Grid / Framework # Iterations)')
        ax.set_xlabel('Improvement Type')
        ax.set_title('Framework Speedup wrt Grid Search')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(speedups) + 1.5)

        # Create legend with improvement type colors - position inside plot, upper left to avoid covering bars
        legend_elements = [
            mpatches.Patch(color=self.colors['calendar'], label='Calendar Update'),
            mpatches.Patch(color=self.colors['role'], label='Shift Responsibility'),
            mpatches.Patch(color=self.colors['resource'], label='Resource Allocation'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(fig_path)
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
        Right: Speedup by improvement type
        """
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      and e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for combined summary")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Grouped bar chart (averages by type)
        ax1 = axes[0]

        # Group by improvement type
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": []}
            by_type[imp_type]["grid"].append(e.get("baseline_iterations", {}).get("mean", 0) or 0)
            by_type[imp_type]["llm"].append(e.get("llm_iterations", {}).get("mean", 0) or 0)

        types = self._sort_types(list(by_type.keys()))
        grid_iters = [np.mean(by_type[t]["grid"]) for t in types]
        llm_iters = [np.mean(by_type[t]["llm"]) for t in types]
        labels = [self._format_type_label(t) for t in types]

        x = np.arange(len(types))
        width = 0.35

        ax1.bar(x - width/2, grid_iters, width, label='Grid', color=self.colors['grid'],
                edgecolor='black', linewidth=0.5)
        ax1.bar(x + width/2, llm_iters, width, label='Framework', color=self.colors['llm'],
                edgecolor='black', linewidth=0.5)

        ax1.set_ylabel('Avg # Iterations')
        ax1.set_xlabel('Improvement Type')
        ax1.set_title('(a) Average Iterations to Solution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, max(grid_iters + [1]) + 1)

        # Right: Aggregate speedup by type (using sum/sum for accurate calculation)
        ax2 = axes[1]

        # Group by improvement type - reuse the sorted types from left panel
        # Compute speedup as sum(grid) / sum(llm) for each type
        speedups = [sum(by_type[t]["grid"]) / sum(by_type[t]["llm"]) if sum(by_type[t]["llm"]) > 0 else 0 for t in types]

        # Add overall
        total_grid = sum(sum(by_type[t]["grid"]) for t in types)
        total_llm = sum(sum(by_type[t]["llm"]) for t in types)
        types_with_overall = types + ["Overall"]
        speedups.append(total_grid / total_llm if total_llm > 0 else 0)

        # Prettify type names
        type_labels = [self._format_type_label(t) for t in types_with_overall]

        bar_colors = []
        for t in types_with_overall:
            if "calendar" in t.lower():
                bar_colors.append(self.colors['calendar'])
            elif "shift" in t.lower() or "responsibility" in t.lower():
                bar_colors.append(self.colors['role'])
            elif "resource" in t.lower():
                bar_colors.append(self.colors['resource'])
            else:
                bar_colors.append(self.colors['speedup'])

        bars = ax2.bar(type_labels, speedups, color=bar_colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, speedups):
            ax2.annotate(f'{val:.2f}x',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Speedup (Grid / Framework # Iterations)')
        ax2.set_xlabel('Improvement Type')
        ax2.set_title('(b) Framework Speedup wrt Grid Search')
        ax2.set_ylim(0, max(speedups) + 1.5)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(fig_path)
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
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      and e.get("llm_iterations", {}).get("mean") is not None]

        # Calculate hypothesis metrics using sum/sum for accurate speedup
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": [], "success": []}
            by_type[imp_type]["grid"].append(e["baseline_iterations"]["mean"])
            by_type[imp_type]["llm"].append(e["llm_iterations"]["mean"])
            by_type[imp_type]["success"].append(e.get("success_rates", {}).get("llm", 0))

        # Define hypotheses
        hypotheses = []

        # H1: Resource speedup (using sum/sum)
        if "resource_allocation" in by_type:
            vals = by_type["resource_allocation"]
            resource_speedup = sum(vals["grid"]) / sum(vals["llm"]) if sum(vals["llm"]) > 0 else None
        else:
            resource_speedup = None
        hypotheses.append({
            'name': 'H1: Resource Speedup\n(>1.5x)',
            'threshold': 1.5,
            'observed': resource_speedup,
            'supported': resource_speedup and resource_speedup > 1.5 if resource_speedup else None
        })

        # H2: Calendar speedup (using sum/sum)
        if "calendar_adjustment" in by_type:
            vals = by_type["calendar_adjustment"]
            calendar_speedup = sum(vals["grid"]) / sum(vals["llm"]) if sum(vals["llm"]) > 0 else None
        else:
            calendar_speedup = None
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
        Figure 6: Table-style visualization of results averaged by evaluation type.

        Good for paper appendix or quick reference.
        """
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None
                      or e.get("llm_iterations", {}).get("mean") is not None]

        if not valid_exps:
            print("No valid experiments for results table")
            return ""

        # Group by improvement type and compute aggregates
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": [], "success": []}
            baseline = e.get("baseline_iterations", {}).get("mean")
            llm = e.get("llm_iterations", {}).get("mean")
            success = e.get("success_rates", {}).get("llm", 0)
            if baseline:
                by_type[imp_type]["grid"].append(baseline)
            if llm:
                by_type[imp_type]["llm"].append(llm)
            by_type[imp_type]["success"].append(success)

        fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * len(by_type)))
        ax.axis('off')

        headers = ['Evaluation Type', 'N', 'Avg Grid', 'Avg LLM', 'Speedup', 'Success Rate']
        data = []

        for imp_type in self._sort_types(list(by_type.keys())):
            vals = by_type[imp_type]
            n = len(vals["grid"]) if vals["grid"] else len(vals["success"])
            avg_grid = np.mean(vals["grid"]) if vals["grid"] else None
            avg_llm = np.mean(vals["llm"]) if vals["llm"] else None
            # Compute speedup as sum(grid) / sum(llm)
            speedup = sum(vals["grid"]) / sum(vals["llm"]) if vals["llm"] and sum(vals["llm"]) > 0 else None
            # Use actual success rate (average of per-experiment success rates)
            success_rate = np.mean(vals["success"]) if vals["success"] else 0

            data.append([
                "Calendar Update" if imp_type == "calendar_adjustment" else imp_type.replace("_", " ").title(),
                str(n),
                f'{avg_grid:.1f}' if avg_grid else 'N/A',
                f'{avg_llm:.1f}' if avg_llm else 'N/A',
                f'{speedup:.2f}x' if speedup else 'N/A',
                f'{success_rate*100:.0f}%'
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
            speedup_str = row[4]
            if speedup_str != 'N/A':
                speedup = float(speedup_str.replace('x', ''))
                if speedup > 1.5:
                    table[(i + 1, 4)].set_facecolor('#90EE90')  # Light green
                elif speedup >= 1.0:
                    table[(i + 1, 4)].set_facecolor('#FFFACD')  # Light yellow

        plt.title('Evaluation Results by Type (Averaged)', fontsize=14,
                  fontweight='bold', y=0.98)

        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.pdf"
        plt.savefig(fig_path)
        plt.savefig(str(fig_path).replace('.pdf', '.png'))
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

    def generate_latex_table(self, experiments: List[Dict]) -> str:
        """Generate LaTeX table for paper with aggregates by evaluation type."""
        # Include ALL experiments with baseline data (not just successful LLM ones)
        # This ensures success rate reflects actual performance
        valid_exps = [e for e in experiments
                      if e.get("baseline_iterations", {}).get("mean") is not None]

        # Group by improvement type
        by_type = {}
        for e in valid_exps:
            imp_type = e.get("improvement_type", "unknown")
            if imp_type not in by_type:
                by_type[imp_type] = {"grid": [], "llm": [], "success": []}
            baseline = e.get("baseline_iterations", {}).get("mean")
            llm = e.get("llm_iterations", {}).get("mean")
            success = e.get("success_rates", {}).get("llm", 0)
            if baseline:
                by_type[imp_type]["grid"].append(baseline)
            if llm:
                by_type[imp_type]["llm"].append(llm)
            by_type[imp_type]["success"].append(success)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparative Evaluation: Grid Search vs. LLM Framework.}",
            r"\label{tab:results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Evaluation Type} & \textbf{Avg Grid} & \textbf{Avg LLM} & \textbf{Speedup} & \textbf{Success Rate} \\",
            r"\midrule",
        ]

        for imp_type in self._sort_types(list(by_type.keys())):
            vals = by_type[imp_type]
            type_name = "Calendar Update" if imp_type == "calendar_adjustment" else imp_type.replace("_", " ").title()
            avg_grid = np.mean(vals["grid"]) if vals["grid"] else None
            avg_llm = np.mean(vals["llm"]) if vals["llm"] else None
            # Compute speedup as sum(grid) / sum(llm)
            speedup = sum(vals["grid"]) / sum(vals["llm"]) if vals["llm"] and sum(vals["llm"]) > 0 else None
            # Use actual success rate (average of per-experiment success rates)
            success_rate = np.mean(vals["success"]) if vals["success"] else 0

            grid_str = f'{avg_grid:.1f}' if avg_grid else 'N/A'
            llm_str = f'{avg_llm:.1f}' if avg_llm else 'N/A'
            speedup_str = f'\\textbf{{{speedup:.2f}$\\times$}}' if speedup and speedup > 1.5 else (f'{speedup:.2f}$\\times$' if speedup else 'N/A')
            success_str = f'{success_rate*100:.0f}\\%'

            lines.append(f"{type_name} & {grid_str} & {llm_str} & {speedup_str} & {success_str} \\\\")

        # Add overall summary row
        all_grid = [e["baseline_iterations"]["mean"] for e in valid_exps if e.get("baseline_iterations", {}).get("mean")]
        all_llm = [e["llm_iterations"]["mean"] for e in valid_exps if e.get("llm_iterations", {}).get("mean")]
        if all_grid and all_llm:
            total_speedup = sum(all_grid) / sum(all_llm)
            avg_grid = np.mean(all_grid)
            avg_llm = np.mean(all_llm)
            lines.append(r"\midrule")
            grid_str = f'{avg_grid:.1f}'
            llm_str = f'{avg_llm:.1f}'
            lines.append(f"\\textit{{Overall}} & {grid_str} & {llm_str} & \\textbf{{{total_speedup:.2f}$\\times$}} & \\\\")

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

    def _compute_proximity(
        self,
        trajectory: List[Dict],
        improvement_type: str,
        fault_info: Dict
    ) -> List[float]:
        """
        Compute proximity to solution for each iteration.

        If is_solution=True, proximity is 1.0 (solution found).
        Otherwise, proximity is computed based on distance from fault state
        to optimal value.

        Returns values from 0 (at fault state) to 1 (at optimal solution).
        """
        proximities = []

        for step in trajectory:
            # If this step found a solution, proximity is 1.0
            if step.get("is_solution", False):
                proximities.append(1.0)
                continue

            if improvement_type == "resource_allocation":
                # Proximity based on resource count
                # fault_count=1, optimal=original_count
                fault_count = 1
                optimal_count = fault_info.get("optimal_count", fault_info.get("target_count", 10))
                current = step.get("resource_count", step.get("proposed_count", 1))
                if optimal_count > fault_count:
                    proximity = (current - fault_count) / (optimal_count - fault_count)
                else:
                    proximity = 1.0 if current == optimal_count else 0.0

            elif improvement_type == "calendar_adjustment":
                # Proximity based on hours added
                # fault = reduced hours, optimal = original hours restored
                fault_hours = fault_info.get("fault_hours", 2)
                optimal_hours = fault_info.get("optimal_hours", 8)
                current = step.get("hours_extended", step.get("proposed_hours", fault_hours))
                if optimal_hours > fault_hours:
                    proximity = (current - fault_hours) / (optimal_hours - fault_hours)
                else:
                    proximity = 1.0 if current == optimal_hours else 0.0

            elif improvement_type == "shift_responsibility":
                # Proximity based on shift fraction
                # fault = 0% shift (overloaded), optimal = target shift fraction (e.g., 50%)
                optimal_fraction = fault_info.get("optimal_fraction", 0.5)
                current = step.get("shift_fraction", step.get("proposed_shift", 0))
                if optimal_fraction > 0:
                    proximity = min(current / optimal_fraction, 1.0)
                else:
                    proximity = 1.0
            else:
                proximity = 0.0

            proximities.append(max(0, min(1, proximity)))  # Clamp to [0, 1]

        return proximities

    def plot_proximity_trajectory(
        self,
        experiments: List[Dict],
        save_name: str = "fig4_proximity_trajectory"
    ) -> str:
        """
        Figure 4: Proximity to solution trajectory by improvement type.

        Shows how Grid and Framework approaches converge to the optimal solution
        over iterations, averaged across all experiments of each type.

        X-axis: # iterations
        Y-axis: Proximity to solution (0 = fault state, 1 = optimal)
        """
        # Group experiments by improvement type
        by_type = {}
        for e in experiments:
            imp_type = e.get("improvement_type", "unknown")
            runs = e.get("runs", [])

            if not runs:
                continue

            if imp_type not in by_type:
                by_type[imp_type] = {"grid_trajectories": [], "llm_trajectories": [], "fault_info": {}}

            # Extract fault info for proximity calculation
            if imp_type == "resource_allocation":
                by_type[imp_type]["fault_info"] = {
                    "optimal_count": 10,  # Default, will be overwritten if available
                }
            elif imp_type == "calendar_adjustment":
                by_type[imp_type]["fault_info"] = {
                    "fault_hours": 2,
                    "optimal_hours": 8,
                }
            elif imp_type == "shift_responsibility":
                by_type[imp_type]["fault_info"] = {
                    "optimal_fraction": 0.5,
                }

            for run in runs:
                baseline_traj = run.get("baseline_trajectory", [])
                llm_traj = run.get("llm_trajectory", [])

                if baseline_traj:
                    # Update fault_info based on trajectory data if available
                    if imp_type == "resource_allocation" and baseline_traj:
                        # Try to infer optimal_count from the last successful iteration
                        for step in reversed(baseline_traj):
                            if step.get("is_solution"):
                                by_type[imp_type]["fault_info"]["optimal_count"] = step.get("resource_count", 10)
                                break

                    proximities = self._compute_proximity(
                        baseline_traj, imp_type, by_type[imp_type]["fault_info"]
                    )
                    by_type[imp_type]["grid_trajectories"].append(proximities)

                if llm_traj:
                    proximities = self._compute_proximity(
                        llm_traj, imp_type, by_type[imp_type]["fault_info"]
                    )
                    by_type[imp_type]["llm_trajectories"].append(proximities)

        if not by_type:
            print("No trajectory data available for proximity plot")
            return ""

        # Create one subplot per improvement type
        num_types = len(by_type)
        if num_types == 0:
            print("No valid improvement types with trajectory data")
            return ""

        # Sort types for consistent ordering
        sorted_types = self._sort_types(list(by_type.keys()))

        fig, axes = plt.subplots(1, num_types, figsize=(5 * num_types, 5), squeeze=False)
        axes = axes.flatten()

        for idx, imp_type in enumerate(sorted_types):
            data = by_type[imp_type]
            ax = axes[idx]

            grid_trajs = data["grid_trajectories"]
            llm_trajs = data["llm_trajectories"]

            # Average grid trajectories (pad shorter ones with their final value)
            if grid_trajs:
                max_len_grid = max(len(t) for t in grid_trajs)
                padded_grid = []
                for t in grid_trajs:
                    padded = t + [t[-1]] * (max_len_grid - len(t)) if t else [0] * max_len_grid
                    padded_grid.append(padded)
                avg_grid = np.mean(padded_grid, axis=0)
                # Start from origin (0, 0)
                x_grid = [0] + list(range(1, len(avg_grid) + 1))
                y_grid = [0] + list(avg_grid)
                ax.plot(x_grid, y_grid, 'o-', color=self.colors['grid'],
                       linewidth=2, markersize=6, label='Grid')

            # Average LLM trajectories
            if llm_trajs:
                max_len_llm = max(len(t) for t in llm_trajs)
                padded_llm = []
                for t in llm_trajs:
                    padded = t + [t[-1]] * (max_len_llm - len(t)) if t else [0] * max_len_llm
                    padded_llm.append(padded)
                avg_llm = np.mean(padded_llm, axis=0)
                # Start from origin (0, 0)
                x_llm = [0] + list(range(1, len(avg_llm) + 1))
                y_llm = [0] + list(avg_llm)
                ax.plot(x_llm, y_llm, 's-', color=self.colors['llm'],
                       linewidth=2, markersize=6, label='Framework')

            # Mark solution threshold
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Solution')

            ax.set_xlabel('# Iterations')
            ax.set_ylabel('Proximity to Solution')
            # Format title: "calendar_adjustment" -> "Calendar Update", others use standard title case
            title = "Calendar Update" if imp_type == "calendar_adjustment" else imp_type.replace("_", " ").title()
            ax.set_title(title)
            ax.set_ylim(-0.05, 1.15)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer x-axis ticks
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Convergence to Solution by Improvement Type', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_path = self.figures_dir / f"{save_name}.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"Generated: {fig_path}")
        return str(fig_path)

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

        # Generate proximity trajectory plots (if trajectory data available)
        fig = self.plot_proximity_trajectory(experiments)
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
