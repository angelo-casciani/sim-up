"""
Results Visualizer for Quantitative Evaluation.

Generates charts for the paper comparing LLM-guided search vs Grid Search:
- X-Axis: Number of Simulation Calls (iterations)
- Y-Axis: Process KPIs (Cycle Time, Throughput, Waiting Time)
"""

import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class ResultsVisualizer:
    """Generate publication-quality charts from evaluation results."""

    def __init__(self, output_dir: str):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Publication-quality style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'grid': '#E74C3C',      # Red for grid search
            'llm': '#2ECC71',       # Green for LLM
            'optimal': '#3498DB',   # Blue for optimal line
            'faulty': '#95A5A6'     # Gray for faulty baseline
        }

    def plot_single_experiment(
        self,
        experiment: Dict,
        save_name: Optional[str] = None
    ) -> str:
        """
        Plot convergence comparison for a single experiment.

        Args:
            experiment: Experiment result dictionary
            save_name: Output filename (default: log_name)

        Returns:
            Path to saved figure
        """
        log_name = experiment.get("log_name", "unknown")
        grid_data = experiment.get("grid_search", {})
        llm_data = experiment.get("llm_search", {})
        optimal_metrics = experiment.get("optimal_metrics", {})
        faulty_metrics = experiment.get("faulty_metrics", {})

        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Grid Search trajectory
        grid_iters = grid_data.get("iterations", [])
        x_grid = []
        y_grid = []
        if grid_iters:
            x_grid = [it["iteration"] for it in grid_iters]
            y_grid = [it["cycle_time"] for it in grid_iters]
            ax.plot(x_grid, y_grid, 'o-', color=self.colors['grid'],
                    linewidth=2, markersize=8, label='Grid Search')

            # Mark solution point for grid
            sol_iter = grid_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                ax.plot(x_grid[sol_idx], y_grid[sol_idx], 's',
                        color=self.colors['grid'], markersize=15,
                        markeredgecolor='black', markeredgewidth=2)

        # Plot LLM Search trajectory
        llm_iters = llm_data.get("iterations", [])
        x_llm = []
        y_llm = []
        if llm_iters:
            x_llm = [it["iteration"] for it in llm_iters]
            y_llm = [it["cycle_time"] for it in llm_iters]
            ax.plot(x_llm, y_llm, 's-', color=self.colors['llm'],
                    linewidth=2, markersize=8, label='LLM Agent')

            # Mark solution point for LLM
            sol_iter = llm_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                ax.plot(x_llm[sol_idx], y_llm[sol_idx], 's',
                        color=self.colors['llm'], markersize=15,
                        markeredgecolor='black', markeredgewidth=2)

        # Reference lines
        max_iter = max(
            max(x_grid) if x_grid else 1,
            max(x_llm) if x_llm else 1
        )
        ax.axhline(y=optimal_cycle, color=self.colors['optimal'],
                   linestyle='--', linewidth=2, label=f'Optimal ({optimal_cycle:.0f}s)')
        ax.axhline(y=faulty_cycle, color=self.colors['faulty'],
                   linestyle=':', linewidth=2, label=f'Faulty ({faulty_cycle:.0f}s)')

        # Formatting
        ax.set_xlabel('Simulation Calls (Iterations)', fontsize=12)
        ax.set_ylabel('Cycle Time (seconds)', fontsize=12)
        ax.set_title(f'Convergence Comparison: {log_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(0, max_iter + 1)

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = os.path.splitext(log_name)[0]
        fig_path = os.path.join(self.output_dir, f"{save_name}_convergence.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_aggregate_comparison(
        self,
        summary: Dict,
        save_name: str = "aggregate_comparison"
    ) -> str:
        """
        Plot aggregate comparison across all experiments.
        Shows only the iterations-to-solution bar chart.

        Args:
            summary: Full evaluation summary dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        experiments = summary.get("experiments", [])
        if not experiments:
            print("No experiments to plot")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        log_names = []
        grid_iters = []
        llm_iters = []

        for exp in experiments:
            name = exp.get("log_name", "").replace(".csv", "").replace(".xes", "")
            grid_sol = exp.get("grid_iterations_to_solution")
            llm_sol = exp.get("llm_iterations_to_solution")

            if grid_sol is not None and llm_sol is not None:
                log_names.append(name[:15])  # Truncate long names
                grid_iters.append(grid_sol)
                llm_iters.append(llm_sol)

        if not log_names:
            print("No successful experiments to plot")
            plt.close()
            return ""

        x = np.arange(len(log_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, grid_iters, width,
                       label='Grid Search', color=self.colors['grid'])
        bars2 = ax.bar(x + width/2, llm_iters, width,
                       label='LLM Agent', color=self.colors['llm'])

        ax.set_xlabel('Event Log', fontsize=12)
        ax.set_ylabel('Iterations to Solution', fontsize=12)
        ax.set_title('Simulation Calls Required to Find Solution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(log_names, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save
        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_convergence_overlay(
        self,
        summary: Dict,
        metric: str = "cycle_time",
        save_name: Optional[str] = None
    ) -> str:
        """
        Plot overlaid convergence curves for all experiments.

        The chart shows normalized KPI values across iterations:
        - Y-axis: Normalized value (0% = optimal, 100% = faulty baseline)
        - X-axis: Simulation calls (iterations)

        Note: Grid search runs ALL iterations (1 to max) to collect complete data,
        even after finding a solution. The solution point is where the KPI first
        reaches the tolerance threshold, but exploration continues.

        Args:
            summary: Full evaluation summary dictionary
            metric: Which metric to plot ("cycle_time", "throughput", "waiting_time")
            save_name: Output filename (default: convergence_overlay_{metric})

        Returns:
            Path to saved figure
        """
        experiments = summary.get("experiments", [])
        if not experiments:
            return ""

        # Metric configuration
        metric_config = {
            "cycle_time": {
                "key": "cycle_time",
                "optimal_key": "avg_cycle_time",
                "faulty_key": "avg_cycle_time",
                "label": "Cycle Time",
                "lower_is_better": True
            },
            "throughput": {
                "key": "throughput",
                "optimal_key": "avg_throughput",
                "faulty_key": "avg_throughput",
                "label": "Throughput",
                "lower_is_better": False  # Higher throughput is better
            },
            "waiting_time": {
                "key": "waiting_time",
                "optimal_key": "avg_waiting_time",
                "faulty_key": "avg_waiting_time",
                "label": "Waiting Time",
                "lower_is_better": True
            }
        }

        config = metric_config.get(metric, metric_config["cycle_time"])

        fig, ax = plt.subplots(figsize=(12, 7))

        for i, exp in enumerate(experiments):
            grid_data = exp.get("grid_search", {})
            llm_data = exp.get("llm_search", {})
            faulty_val = exp.get("faulty_metrics", {}).get(config["faulty_key"], 1)
            optimal_val = exp.get("optimal_metrics", {}).get(config["optimal_key"], 1)

            # Avoid division by zero
            if abs(faulty_val - optimal_val) < 0.001:
                continue

            alpha = 0.7
            log_name = exp.get("log_name", "").replace(".csv", "").replace(".xes", "")

            # Normalize: 0% = optimal, 100% = faulty
            def normalize(val):
                if config["lower_is_better"]:
                    # Lower is better: optimal < faulty
                    return (val - optimal_val) / (faulty_val - optimal_val) * 100
                else:
                    # Higher is better: optimal > faulty
                    return (optimal_val - val) / (optimal_val - faulty_val) * 100

            # Grid search
            grid_iters = grid_data.get("iterations", [])
            if grid_iters:
                x_grid = [it["iteration"] for it in grid_iters]
                y_grid = [normalize(it[config["key"]]) for it in grid_iters]
                line_grid, = ax.plot(x_grid, y_grid, 'o-', color=self.colors['grid'],
                        alpha=alpha, linewidth=1.5, markersize=5)

                # Mark solution point
                sol_iter = grid_data.get("solution_found_at")
                if sol_iter and sol_iter <= len(x_grid):
                    ax.plot(x_grid[sol_iter - 1], y_grid[sol_iter - 1], 'o',
                            color=self.colors['grid'], markersize=12,
                            markeredgecolor='black', markeredgewidth=2, alpha=1.0)

            # LLM search
            llm_iters = llm_data.get("iterations", [])
            if llm_iters:
                x_llm = [it["iteration"] for it in llm_iters]
                y_llm = [normalize(it[config["key"]]) for it in llm_iters]
                line_llm, = ax.plot(x_llm, y_llm, 's-', color=self.colors['llm'],
                        alpha=alpha, linewidth=1.5, markersize=5)

                # Mark solution point
                sol_iter = llm_data.get("solution_found_at")
                if sol_iter and sol_iter <= len(x_llm):
                    ax.plot(x_llm[sol_iter - 1], y_llm[sol_iter - 1], 's',
                            color=self.colors['llm'], markersize=12,
                            markeredgecolor='black', markeredgewidth=2, alpha=1.0)

        # Reference lines
        ax.axhline(y=0, color=self.colors['optimal'], linestyle='--',
                   linewidth=2, label='Optimal (0%)')
        ax.axhline(y=100, color=self.colors['faulty'], linestyle=':',
                   linewidth=2, label='Faulty baseline (100%)')

        # Tolerance band (2% threshold)
        ax.axhspan(-2, 2, alpha=0.2, color=self.colors['optimal'], label='Solution zone (±2%)')

        # Legend
        grid_line = plt.Line2D([0], [0], color=self.colors['grid'],
                               linewidth=2, marker='o', label='Grid Search')
        llm_line = plt.Line2D([0], [0], color=self.colors['llm'],
                              linewidth=2, marker='s', label='LLM Agent')
        solution_marker = plt.Line2D([0], [0], color='black', marker='o',
                                      linestyle='None', markersize=10,
                                      markerfacecolor='white', markeredgewidth=2,
                                      label='Solution found')

        ax.legend(handles=[grid_line, llm_line, solution_marker,
                          plt.Line2D([0], [0], color=self.colors['optimal'],
                                     linestyle='--', label='Optimal'),
                          plt.Line2D([0], [0], color=self.colors['faulty'],
                                     linestyle=':', label='Faulty baseline')],
                  loc='upper right', fontsize=10)

        ax.set_xlabel('Simulation Calls (Iterations)', fontsize=12)
        ax.set_ylabel(f'{config["label"]} - Distance from Optimal (%)', fontsize=12)
        ax.set_title(f'Convergence Pattern: {config["label"]}\n(Grid search runs all iterations for complete data)',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(-15, 115)

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = f"convergence_overlay_{metric}"
        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def generate_all_plots(self, summary_path: str) -> List[str]:
        """
        Generate all plots from a saved evaluation summary.

        Args:
            summary_path: Path to evaluation_summary.json

        Returns:
            List of generated figure paths
        """
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        figures = []

        print("\nGenerating visualizations...")

        # Individual experiment plots
        for exp in summary.get("experiments", []):
            fig_path = self.plot_single_experiment(exp)
            if fig_path:
                figures.append(fig_path)

        # Aggregate comparison (iterations only, no speedup chart)
        fig_path = self.plot_aggregate_comparison(summary)
        if fig_path:
            figures.append(fig_path)

        # Convergence overlays for all three metrics
        for metric in ["cycle_time", "throughput", "waiting_time"]:
            fig_path = self.plot_convergence_overlay(summary, metric=metric)
            if fig_path:
                figures.append(fig_path)

        print(f"\nGenerated {len(figures)} visualizations")
        return figures
