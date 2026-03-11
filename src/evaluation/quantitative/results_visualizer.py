"""
Results Visualizer for Quantitative Evaluation.

Generates publication-quality charts for comparing LLM-guided search vs Grid Search:
- Convergence plots with log scale for time metrics
- Resource allocation trade-off plots (Pareto view)
- Aggregate comparison with error bars
"""

import os
import json
from typing import Dict, List, Optional, Tuple
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

        # Colorblind-friendly palette with high contrast
        self.colors = {
            'grid': '#D55E00',      # Vermillion (red-orange) for grid search
            'llm': '#009E73',       # Bluish green for LLM
            'llm_star': '#F0E442',  # Yellow for LLM star marker
            'baseline': '#0072B2',  # Blue for baseline line
            'faulty': '#CC79A7',    # Reddish purple for faulty state
            'original': '#56B4E9'   # Sky blue for original state
        }

        # Publication font sizes
        self.fontsize = {
            'title': 16,
            'axis_label': 14,
            'tick': 12,
            'legend': 11,
            'annotation': 10
        }

        # Line settings
        self.linewidth = 2.5

    def plot_single_experiment(
        self,
        experiment: Dict,
        save_name: Optional[str] = None
    ) -> str:
        """
        Plot convergence comparison for a single experiment with log scale
        and resource count annotations.

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
        target_resource = experiment.get("target_resource", "Resource")

        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Collect all y values for smart y-axis limits
        all_y = [optimal_cycle, faulty_cycle]

        # Plot Grid Search trajectory with annotations
        grid_iters = grid_data.get("iterations", [])
        x_grid = []
        y_grid = []
        grid_counts = []
        if grid_iters:
            x_grid = [it["iteration"] for it in grid_iters]
            y_grid = [it["cycle_time"] for it in grid_iters]
            grid_counts = [it["resource_count"] for it in grid_iters]
            all_y.extend(y_grid)

            ax.plot(x_grid, y_grid, 'o-', color=self.colors['grid'],
                    linewidth=self.linewidth, markersize=10, label='Grid Search')

            # Annotate resource counts on each point
            for i, (x, y, count) in enumerate(zip(x_grid, y_grid, grid_counts)):
                sol_iter = grid_data.get("solution_found_at")
                if sol_iter and i == sol_iter - 1:
                    # Solution point annotation
                    ax.annotate(f'{count} {target_resource}s\n(Solution)',
                                xy=(x, y), xytext=(15, 15),
                                textcoords='offset points',
                                fontsize=self.fontsize['annotation'],
                                fontweight='bold',
                                color=self.colors['grid'],
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                         edgecolor=self.colors['grid'], alpha=0.9))
                else:
                    ax.annotate(f'{count}',
                                xy=(x, y), xytext=(8, 8),
                                textcoords='offset points',
                                fontsize=self.fontsize['annotation'],
                                color=self.colors['grid'])

        # Plot LLM Search - use star marker and horizontal reference line
        llm_iters = llm_data.get("iterations", [])
        if llm_iters:
            x_llm = [it["iteration"] for it in llm_iters]
            y_llm = [it["cycle_time"] for it in llm_iters]
            llm_counts = [it["proposed_count"] for it in llm_iters]
            all_y.extend(y_llm)

            sol_iter = llm_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                sol_x = x_llm[sol_idx]
                sol_y = y_llm[sol_idx]
                sol_count = llm_counts[sol_idx]

                # Draw horizontal dashed line from LLM solution across the graph
                max_x = max(max(x_grid) if x_grid else 1, max(x_llm) if x_llm else 1) + 1
                ax.hlines(y=sol_y, xmin=sol_x, xmax=max_x,
                         colors=self.colors['llm'], linestyles='--',
                         linewidth=self.linewidth * 0.8, alpha=0.7,
                         label='LLM Target Level')

                # Plot LLM as a prominent star marker
                ax.plot(sol_x, sol_y, marker='*', color=self.colors['llm_star'],
                        markersize=25, markeredgecolor=self.colors['llm'],
                        markeredgewidth=2, zorder=10, label='LLM Solution')

                # Annotate with resource count and "LLM Jump"
                ax.annotate(f'{sol_count} {target_resource}s\n(LLM Jump)',
                            xy=(sol_x, sol_y), xytext=(-60, -40),
                            textcoords='offset points',
                            fontsize=self.fontsize['annotation'],
                            fontweight='bold',
                            color=self.colors['llm'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor=self.colors['llm'], alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color=self.colors['llm']))
            else:
                # If no solution found, plot all iterations
                ax.plot(x_llm, y_llm, 's-', color=self.colors['llm'],
                        linewidth=self.linewidth, markersize=10, label='LLM Agent')

        # Reference lines
        max_iter = max(
            max(x_grid) if x_grid else 1,
            max(x_llm) if x_llm else 1
        ) + 1

        ax.axhline(y=optimal_cycle, color=self.colors['baseline'],
                   linestyle='--', linewidth=self.linewidth,
                   label=f'Baseline ({optimal_cycle/3600:.1f}h)')
        ax.axhline(y=faulty_cycle, color=self.colors['faulty'],
                   linestyle=':', linewidth=self.linewidth,
                   label=f'Faulty ({faulty_cycle/3600:.1f}h)')

        # Use logarithmic scale for better visualization of large differences
        ax.set_yscale('log')

        # Set y-axis limits with padding
        y_min = min(all_y) * 0.5
        y_max = max(all_y) * 2
        ax.set_ylim(y_min, y_max)

        # Formatting with publication-quality fonts
        ax.set_xlabel('Simulation Calls (Iterations)', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Cycle Time (seconds, log scale)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Convergence Comparison: {log_name}\nGrid Search vs LLM-Guided Search',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black')
        ax.set_xlim(0, max_iter)
        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])

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

    def plot_resource_tradeoff(
        self,
        experiment: Dict,
        save_name: Optional[str] = None
    ) -> str:
        """
        Plot Performance vs Resource Count (Pareto trade-off view).

        This visualizes the trade-off between resource allocation and performance,
        showing where Grid stopped (minimum viable), where LLM jumped to (safer),
        and the original state (potentially over-provisioned).

        Args:
            experiment: Experiment result dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        log_name = experiment.get("log_name", "unknown")
        grid_data = experiment.get("grid_search", {})
        llm_data = experiment.get("llm_search", {})
        optimal_metrics = experiment.get("optimal_metrics", {})
        faulty_metrics = experiment.get("faulty_metrics", {})
        target_resource = experiment.get("target_resource", "Resource")
        original_count = experiment.get("original_count", 8)

        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Collect data points
        all_x = []
        all_y = []

        # Plot Grid Search curve (resource count vs cycle time)
        grid_iters = grid_data.get("iterations", [])
        if grid_iters:
            grid_counts = [it["resource_count"] for it in grid_iters]
            grid_cycles = [it["cycle_time"] for it in grid_iters]
            all_x.extend(grid_counts)
            all_y.extend(grid_cycles)

            ax.plot(grid_counts, grid_cycles, 'o-', color=self.colors['grid'],
                    linewidth=self.linewidth, markersize=12, label='Grid Search Path')

            # Mark the Grid solution point
            sol_iter = grid_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                ax.plot(grid_counts[sol_idx], grid_cycles[sol_idx], 'o',
                        color=self.colors['grid'], markersize=20,
                        markeredgecolor='black', markeredgewidth=3)
                ax.annotate(f'Grid Solution\n({grid_counts[sol_idx]} {target_resource}s)',
                            xy=(grid_counts[sol_idx], grid_cycles[sol_idx]),
                            xytext=(20, 20), textcoords='offset points',
                            fontsize=self.fontsize['annotation'], fontweight='bold',
                            color=self.colors['grid'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor=self.colors['grid'], alpha=0.9))

        # Plot LLM solution as a prominent star
        llm_iters = llm_data.get("iterations", [])
        if llm_iters:
            sol_iter = llm_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                llm_count = llm_iters[sol_idx]["proposed_count"]
                llm_cycle = llm_iters[sol_idx]["cycle_time"]
                all_x.append(llm_count)
                all_y.append(llm_cycle)

                ax.plot(llm_count, llm_cycle, marker='*', color=self.colors['llm_star'],
                        markersize=30, markeredgecolor=self.colors['llm'],
                        markeredgewidth=2, zorder=10, label='LLM Solution')
                ax.annotate(f'LLM Jump\n({llm_count} {target_resource}s)',
                            xy=(llm_count, llm_cycle),
                            xytext=(-60, 30), textcoords='offset points',
                            fontsize=self.fontsize['annotation'], fontweight='bold',
                            color=self.colors['llm'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor=self.colors['llm'], alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color=self.colors['llm']))

        # Plot Original state (baseline)
        all_x.append(original_count)
        all_y.append(optimal_cycle)
        ax.plot(original_count, optimal_cycle, 'D', color=self.colors['original'],
                markersize=15, markeredgecolor='black', markeredgewidth=2,
                label=f'Original ({original_count} {target_resource}s)')
        ax.annotate(f'Original State\n(Baseline)',
                    xy=(original_count, optimal_cycle),
                    xytext=(10, -40), textcoords='offset points',
                    fontsize=self.fontsize['annotation'],
                    color=self.colors['original'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=self.colors['original'], alpha=0.9))

        # Add horizontal baseline line
        ax.axhline(y=optimal_cycle, color=self.colors['baseline'],
                   linestyle='--', linewidth=self.linewidth * 0.7, alpha=0.7,
                   label='Baseline Performance')

        # Use log scale for y-axis
        ax.set_yscale('log')

        # Set axis limits
        ax.set_xlim(0, max(all_x) + 1)
        y_min = min(all_y) * 0.5
        y_max = max(all_y) * 2
        ax.set_ylim(y_min, y_max)

        # Formatting
        ax.set_xlabel(f'Number of {target_resource}s', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Cycle Time (seconds, log scale)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Resource-Performance Trade-off: {log_name}\n'
                     f'Grid: Minimum Viable | LLM: Intelligent Jump | Original: Baseline',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black')
        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])

        # Add grid for readability
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = os.path.splitext(log_name)[0]
        fig_path = os.path.join(self.output_dir, f"{save_name}_tradeoff.png")
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
        Plot aggregate comparison across all experiments with error bars.

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

        fig, ax = plt.subplots(figsize=(12, 7))

        log_names = []
        grid_iters = []
        llm_iters = []
        speedups = []

        for exp in experiments:
            name = exp.get("log_name", "").replace(".csv", "").replace(".xes", "")
            grid_sol = exp.get("grid_iterations_to_solution")
            llm_sol = exp.get("llm_iterations_to_solution")
            speedup = exp.get("speedup_ratio")

            if grid_sol is not None and llm_sol is not None:
                log_names.append(name[:15])  # Truncate long names
                grid_iters.append(grid_sol)
                llm_iters.append(llm_sol)
                speedups.append(speedup if speedup else 0)

        if not log_names:
            print("No successful experiments to plot")
            plt.close()
            return ""

        x = np.arange(len(log_names))
        width = 0.35

        # Calculate error bars (std dev) if multiple experiments
        grid_err = np.std(grid_iters) if len(grid_iters) > 1 else 0
        llm_err = np.std(llm_iters) if len(llm_iters) > 1 else 0

        bars1 = ax.bar(x - width/2, grid_iters, width,
                       label='Grid Search', color=self.colors['grid'],
                       yerr=grid_err if len(log_names) > 1 else None,
                       capsize=5, error_kw={'linewidth': 2})
        bars2 = ax.bar(x + width/2, llm_iters, width,
                       label='LLM Agent', color=self.colors['llm'],
                       yerr=llm_err if len(log_names) > 1 else None,
                       capsize=5, error_kw={'linewidth': 2})

        ax.set_xlabel('Event Log', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Iterations to Solution', fontsize=self.fontsize['axis_label'])
        ax.set_title('Simulation Calls Required to Find Solution\n'
                     '(Lower is better - fewer simulation calls needed)',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(log_names, rotation=45, ha='right', fontsize=self.fontsize['tick'])
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black')
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])

        # Add value labels and speedup on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            h1 = bar1.get_height()
            h2 = bar2.get_height()

            ax.annotate(f'{int(h1)}',
                        xy=(bar1.get_x() + bar1.get_width() / 2, h1),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=self.fontsize['annotation'],
                        fontweight='bold')
            ax.annotate(f'{int(h2)}',
                        xy=(bar2.get_x() + bar2.get_width() / 2, h2),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=self.fontsize['annotation'],
                        fontweight='bold')

            # Add speedup annotation between bars
            if speedups[i] > 0:
                speedup_text = f'{speedups[i]:.1f}x' if speedups[i] >= 1 else f'{1/speedups[i]:.1f}x slower'
                ax.annotate(speedup_text,
                            xy=(x[i], max(h1, h2) + 0.5),
                            xytext=(0, 15), textcoords="offset points",
                            ha='center', va='bottom', fontsize=self.fontsize['annotation'] - 1,
                            color='gray', style='italic')

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
        save_name: Optional[str] = None,
        tolerance_pct: float = 5.0,
        use_log_scale: bool = True
    ) -> str:
        """
        Plot overlaid convergence curves for all experiments with log scale
        and resource count annotations.

        Args:
            summary: Full evaluation summary dictionary
            metric: Which metric to plot ("cycle_time", "throughput", "waiting_time")
            save_name: Output filename
            tolerance_pct: Tolerance percentage for solution zone
            use_log_scale: Whether to use log scale for time-based metrics

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
                "unit": "seconds",
                "lower_is_better": True,
                "use_log": True
            },
            "throughput": {
                "key": "throughput",
                "optimal_key": "avg_throughput",
                "faulty_key": "avg_throughput",
                "label": "Throughput",
                "unit": "cases/day",
                "lower_is_better": False,
                "use_log": False
            },
            "waiting_time": {
                "key": "waiting_time",
                "optimal_key": "avg_waiting_time",
                "faulty_key": "avg_waiting_time",
                "label": "Waiting Time",
                "unit": "seconds",
                "lower_is_better": True,
                "use_log": True
            }
        }

        config = metric_config.get(metric, metric_config["cycle_time"])

        fig, ax = plt.subplots(figsize=(12, 7))

        all_y = []

        for i, exp in enumerate(experiments):
            grid_data = exp.get("grid_search", {})
            llm_data = exp.get("llm_search", {})
            faulty_val = exp.get("faulty_metrics", {}).get(config["faulty_key"], 1)
            optimal_val = exp.get("optimal_metrics", {}).get(config["optimal_key"], 1)
            target_resource = exp.get("target_resource", "Resource")

            all_y.extend([faulty_val, optimal_val])

            log_name = exp.get("log_name", "").replace(".csv", "").replace(".xes", "")

            # Grid search with annotations
            grid_iters = grid_data.get("iterations", [])
            if grid_iters:
                x_grid = [it["iteration"] for it in grid_iters]
                y_grid = [it[config["key"]] for it in grid_iters]
                counts = [it["resource_count"] for it in grid_iters]
                all_y.extend(y_grid)

                ax.plot(x_grid, y_grid, 'o-', color=self.colors['grid'],
                        linewidth=self.linewidth, markersize=10, alpha=0.8,
                        label='Grid Search' if i == 0 else '')

                # Annotate with resource counts
                for j, (x, y, count) in enumerate(zip(x_grid, y_grid, counts)):
                    sol_iter = grid_data.get("solution_found_at")
                    if sol_iter and j == sol_iter - 1:
                        ax.plot(x, y, 'o', color=self.colors['grid'], markersize=16,
                                markeredgecolor='black', markeredgewidth=2)
                        ax.annotate(f'{count}', xy=(x, y), xytext=(5, 10),
                                    textcoords='offset points',
                                    fontsize=self.fontsize['annotation'],
                                    fontweight='bold', color=self.colors['grid'])
                    else:
                        ax.annotate(f'{count}', xy=(x, y), xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=self.fontsize['annotation'] - 1,
                                    color=self.colors['grid'], alpha=0.8)

            # LLM search with star marker
            llm_iters = llm_data.get("iterations", [])
            if llm_iters:
                sol_iter = llm_data.get("solution_found_at")
                if sol_iter:
                    sol_idx = sol_iter - 1
                    sol_x = llm_iters[sol_idx]["iteration"]
                    sol_y = llm_iters[sol_idx][config["key"]]
                    sol_count = llm_iters[sol_idx]["proposed_count"]
                    all_y.append(sol_y)

                    # Horizontal dashed line from LLM solution
                    max_x = max(max(x_grid) if x_grid else 1, sol_x) + 1
                    ax.hlines(y=sol_y, xmin=sol_x, xmax=max_x,
                             colors=self.colors['llm'], linestyles='--',
                             linewidth=self.linewidth * 0.7, alpha=0.6)

                    # Star marker
                    ax.plot(sol_x, sol_y, marker='*', color=self.colors['llm_star'],
                            markersize=25, markeredgecolor=self.colors['llm'],
                            markeredgewidth=2, zorder=10,
                            label='LLM Solution' if i == 0 else '')

                    ax.annotate(f'{sol_count}', xy=(sol_x, sol_y),
                                xytext=(-15, 15), textcoords='offset points',
                                fontsize=self.fontsize['annotation'],
                                fontweight='bold', color=self.colors['llm'])

        # Reference lines
        # Use first experiment's values for reference
        exp = experiments[0]
        optimal_val = exp.get("optimal_metrics", {}).get(config["optimal_key"], 1)
        faulty_val = exp.get("faulty_metrics", {}).get(config["faulty_key"], 1)

        ax.axhline(y=optimal_val, color=self.colors['baseline'],
                   linestyle='--', linewidth=self.linewidth,
                   label=f'Baseline ({optimal_val:.0f}{config["unit"][:1]})')
        ax.axhline(y=faulty_val, color=self.colors['faulty'],
                   linestyle=':', linewidth=self.linewidth,
                   label=f'Faulty ({faulty_val:.0f}{config["unit"][:1]})')

        # Use log scale for time-based metrics
        if use_log_scale and config["use_log"]:
            ax.set_yscale('log')
            y_label = f'{config["label"]} ({config["unit"]}, log scale)'
        else:
            y_label = f'{config["label"]} ({config["unit"]})'

        # Set y limits
        if all_y:
            y_min = min(all_y) * 0.5
            y_max = max(all_y) * 2
            ax.set_ylim(y_min, y_max)

        # Formatting
        ax.set_xlabel('Simulation Calls (Iterations)', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel(y_label, fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Convergence Pattern: {config["label"]}\n'
                     f'Numbers on points indicate resource count',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black')
        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])
        ax.grid(True, alpha=0.3)

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

    def generate_all_plots(
        self,
        summary_path: str,
        tolerance_pct: float = 5.0
    ) -> List[str]:
        """
        Generate all plots from a saved evaluation summary.

        Args:
            summary_path: Path to evaluation_summary.json
            tolerance_pct: Tolerance percentage for solution zone

        Returns:
            List of generated figure paths
        """
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        figures = []

        print("\nGenerating publication-quality visualizations...")

        # Individual experiment plots (convergence + trade-off)
        for exp in summary.get("experiments", []):
            fig_path = self.plot_single_experiment(exp)
            if fig_path:
                figures.append(fig_path)

            # New: Resource trade-off plot
            fig_path = self.plot_resource_tradeoff(exp)
            if fig_path:
                figures.append(fig_path)

        # Aggregate comparison with error bars
        fig_path = self.plot_aggregate_comparison(summary)
        if fig_path:
            figures.append(fig_path)

        # Convergence overlays for all three metrics (with log scale)
        for metric in ["cycle_time", "throughput", "waiting_time"]:
            fig_path = self.plot_convergence_overlay(
                summary, metric=metric, tolerance_pct=tolerance_pct
            )
            if fig_path:
                figures.append(fig_path)

        # Resource utilization plots
        for exp in summary.get("experiments", []):
            fig_path = self.plot_resource_utilization(exp)
            if fig_path:
                figures.append(fig_path)

        print(f"\nGenerated {len(figures)} publication-quality visualizations")
        return figures

    def plot_resource_utilization(
        self,
        experiment: Dict,
        save_name: Optional[str] = None
    ) -> str:
        """
        Plot resource utilization vs resource count.

        Shows how utilization changes as resources are added, highlighting
        the trade-off between over-utilization (bottleneck) and under-utilization
        (waste).

        Args:
            experiment: Experiment result dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        log_name = experiment.get("log_name", "unknown")
        grid_data = experiment.get("grid_search", {})
        llm_data = experiment.get("llm_search", {})
        target_resource = experiment.get("target_resource", "Resource")

        # Extract utilization data from Grid Search iterations
        grid_iters = grid_data.get("iterations", [])
        grid_counts = []
        grid_utils = []

        for it in grid_iters:
            count = it.get("resource_count", 0)
            util_dict = it.get("resource_utilization", {})
            if util_dict:
                # Calculate average utilization across all resources
                avg_util = sum(util_dict.values()) / len(util_dict) if util_dict else 0
                grid_counts.append(count)
                grid_utils.append(avg_util * 100)  # Convert to percentage

        # Check if we have enough data to plot
        if not grid_counts and not llm_data.get("iterations", []):
            print(f"  [!] Skipping utilization chart for {log_name}: no utilization data available")
            return ""

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot Grid Search utilization curve
        if grid_counts and grid_utils:
            ax.plot(grid_counts, grid_utils, 'o-', color=self.colors['grid'],
                    linewidth=self.linewidth, markersize=10, label='Grid Search')

            # Mark solution point
            sol_iter = grid_data.get("solution_found_at")
            if sol_iter and sol_iter <= len(grid_counts):
                sol_idx = sol_iter - 1
                ax.plot(grid_counts[sol_idx], grid_utils[sol_idx], 'o',
                        color=self.colors['grid'], markersize=20,
                        markeredgecolor='black', markeredgewidth=3)
                ax.annotate(f'Grid Solution\n({grid_counts[sol_idx]} {target_resource}s)\n{grid_utils[sol_idx]:.1f}% util',
                            xy=(grid_counts[sol_idx], grid_utils[sol_idx]),
                            xytext=(20, 20), textcoords='offset points',
                            fontsize=self.fontsize['annotation'], fontweight='bold',
                            color=self.colors['grid'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor=self.colors['grid'], alpha=0.9))

        # Extract utilization from LLM Search iterations
        llm_iters = llm_data.get("iterations", [])
        if llm_iters:
            sol_iter = llm_data.get("solution_found_at")
            if sol_iter:
                sol_idx = sol_iter - 1
                llm_count = llm_iters[sol_idx].get("proposed_count", 0)
                util_dict = llm_iters[sol_idx].get("resource_utilization", {})
                if util_dict:
                    avg_util = sum(util_dict.values()) / len(util_dict) * 100

                    ax.plot(llm_count, avg_util, marker='*', color=self.colors['llm_star'],
                            markersize=30, markeredgecolor=self.colors['llm'],
                            markeredgewidth=2, zorder=10, label='LLM Solution')
                    ax.annotate(f'LLM Jump\n({llm_count} {target_resource}s)\n{avg_util:.1f}% util',
                                xy=(llm_count, avg_util),
                                xytext=(-60, 30), textcoords='offset points',
                                fontsize=self.fontsize['annotation'], fontweight='bold',
                                color=self.colors['llm'],
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                         edgecolor=self.colors['llm'], alpha=0.9),
                                arrowprops=dict(arrowstyle='->', color=self.colors['llm']))

        # Reference zones for utilization
        ax.axhspan(70, 85, alpha=0.15, color='green', label='Optimal Zone (70-85%)')
        ax.axhspan(85, 100, alpha=0.15, color='red', label='Over-utilized (>85%)')
        ax.axhspan(0, 70, alpha=0.1, color='yellow', label='Under-utilized (<70%)')

        # Formatting
        if grid_counts:
            ax.set_xlim(0, max(grid_counts) + 1)
        ax.set_ylim(0, 105)

        ax.set_xlabel(f'Number of {target_resource}s', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Average Resource Utilization (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Resource Utilization Analysis: {log_name}\n'
                     f'Trade-off between bottleneck (high util) and waste (low util)',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black')
        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        if save_name is None:
            save_name = os.path.splitext(log_name)[0]
        fig_path = os.path.join(self.output_dir, f"{save_name}_utilization.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path
