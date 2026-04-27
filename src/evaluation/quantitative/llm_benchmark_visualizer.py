"""
LLM Benchmark Visualizer

Generates publication-quality charts for LLM benchmark evaluation:
- Convergence trajectory: Distance to 1% improvement threshold vs proposals
- Model comparison: Bar charts for proposals, time, and impact
- Speed vs Impact scatter plots
"""

import os
import json
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class LLMBenchmarkVisualizer:
    """Generate charts from LLM benchmark results."""

    # Validation threshold (1% improvement required)
    IMPROVEMENT_THRESHOLD = 0.01

    def __init__(self, output_dir: str, log_name: str = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated charts
            log_name: Name of the event log (for figure titles)
        """
        self.output_dir = output_dir
        self.log_name = log_name
        os.makedirs(output_dir, exist_ok=True)

        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')

        # Colorblind-friendly palette
        self.model_colors = {
            # OpenAI - Blue shades
            'gpt-4.1': '#1E90FF',
            'gpt-5': '#0072B2',
            'gpt-5-mini': '#56B4E9',
            'gpt-5-nano': '#ADD8E6',
            'o3': '#003366',
            # Anthropic - Orange/red shades
            'anthropic--claude-4.5-opus': '#D55E00',
            'anthropic--claude-4.5-sonnet': '#E69F00',
            'anthropic--claude-4.5-haiku': '#F0E442',
            # Mistral - Green shades
            'mistralai--mistral-large-instruct': '#009E73',
            'mistralai--mistral-small-instruct': '#B2DF8A',
        }

        # Fallback colors
        self.fallback_colors = plt.cm.tab10.colors

        # Publication font sizes
        self.fontsize = {
            'title': 16,
            'axis_label': 14,
            'tick': 12,
            'legend': 10,
            'annotation': 10
        }

        self.linewidth = 2.0
        self.markersize = 8

    def _get_model_color(self, model_name: str, index: int = 0) -> str:
        """Get color for a model, with fallback."""
        if model_name in self.model_colors:
            return self.model_colors[model_name]
        return self.fallback_colors[index % len(self.fallback_colors)]

    def _get_title_suffix(self) -> str:
        """Get log name suffix for titles."""
        if self.log_name:
            return f" ({self.log_name})"
        return ""

    def plot_score_progression(
        self,
        report: Dict,
        save_name: str = "score_progression"
    ) -> str:
        """
        Plot score progression: averaged_score vs proposals for each model.

        Y-axis: Averaged validation score (higher is better)
        X-axis: Proposal number
        Includes 1% threshold line.
        Rejected proposals (before simulation) are shown at 0.0.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            print("No results to plot")
            return ""

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            runs = model_result.get("runs", [])

            if not runs:
                continue

            # Aggregate all runs for this model
            max_proposals = max(run.get("total_proposals", len(run.get("iterations", []))) for run in runs)
            proposal_scores = {prop_num: [] for prop_num in range(1, max_proposals + 1)}

            for run in runs:
                # Get all proposals including rejected ones
                all_proposals = run.get("all_proposals", run.get("iterations", []))
                for prop in all_proposals:
                    prop_num = prop.get("proposal_num", prop.get("iteration_num", 0))
                    # Rejected proposals have score 0.0
                    if prop.get("rejected_before_simulation", False):
                        score = 0.0
                    else:
                        score = prop.get("kpi_impact", {}).get("averaged_score")
                    if score is not None and prop_num in proposal_scores:
                        proposal_scores[prop_num].append(score)

            # Calculate mean and std for each proposal
            x_vals = []
            y_vals = []
            y_std = []

            for prop_num in sorted(proposal_scores.keys()):
                scores = proposal_scores[prop_num]
                if scores:
                    x_vals.append(prop_num)
                    y_vals.append(np.mean(scores))
                    y_std.append(np.std(scores) if len(scores) > 1 else 0)

            if x_vals and y_vals:
                color = self._get_model_color(model_name, i)

                ax.plot(x_vals, y_vals, 'o-',
                        color=color,
                        linewidth=self.linewidth,
                        markersize=self.markersize,
                        label=model_name)

                # Add error bands
                if any(s > 0 for s in y_std):
                    y_lower = np.array(y_vals) - np.array(y_std)
                    y_upper = np.array(y_vals) + np.array(y_std)
                    ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.15)

                # Mark threshold crossings
                for j, (x, y) in enumerate(zip(x_vals, y_vals)):
                    if y >= self.IMPROVEMENT_THRESHOLD:
                        ax.plot(x, y, '*', color=color, markersize=15,
                                markeredgecolor='black', markeredgewidth=1)
                        break

        # Add threshold line
        ax.axhline(y=self.IMPROVEMENT_THRESHOLD, color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label='1% Improvement Threshold')

        # Add 0 reference line
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Formatting
        ax.set_xlabel('# Proposal', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Averaged Validation Score', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'LLM Score Progression{self._get_title_suffix()}\n'
                     f'Averaged Validation Score per Proposal',
                     fontsize=self.fontsize['title'], fontweight='bold')

        max_proposals = report.get("max_proposals", report.get("max_iterations", 5))
        ax.set_xlim(0.5, max_proposals + 0.5)
        # Show ticks at 1, 5, 10, 15, ... intervals
        ticks = [1] + [i for i in range(5, max_proposals + 1, 5)]
        ax.set_xticks(ticks)

        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])
        ax.legend(loc='lower right', fontsize=self.fontsize['legend'],
                  frameon=True, facecolor='white', edgecolor='black',
                  ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save (PNG only)
        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_proposals_comparison(
        self,
        report: Dict,
        save_name: str = "proposals_comparison"
    ) -> str:
        """
        Plot bar chart comparing # proposals to first improvement across models.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))

        model_names = []
        proposals_means = []
        proposals_stds = []
        colors = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            # Use proposals_to_improvement if available, otherwise fall back to iterations_to_improvement
            proposals_mean = model_result.get("proposals_to_improvement_mean") or model_result.get("iterations_to_improvement_mean")
            proposals_std = model_result.get("proposals_to_improvement_std") or model_result.get("iterations_to_improvement_std", 0)

            if proposals_mean is not None:
                model_names.append(model_name)  # Use full model name
                proposals_means.append(proposals_mean)
                proposals_stds.append(proposals_std or 0)
                colors.append(self._get_model_color(model_name, i))

        if not model_names:
            print("No successful runs to plot")
            plt.close()
            return ""

        x = np.arange(len(model_names))
        bars = ax.bar(x, proposals_means, color=colors)

        # Add value labels
        for bar, mean in zip(bars, proposals_means):
            height = bar.get_height()
            label = f'{mean:.1f}'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'],
                        fontweight='bold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Avg # Proposals to First Improvement', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Avg # Proposals to First Validated Improvement{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.set_ylim(0, max(proposals_means) * 1.3)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_time_comparison(
        self,
        report: Dict,
        save_name: str = "time_comparison"
    ) -> str:
        """
        Plot bar chart comparing time to first improvement across models.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))

        model_names = []
        time_means = []
        time_stds = []
        colors = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            time_mean = model_result.get("time_to_improvement_mean")
            time_std = model_result.get("time_to_improvement_std", 0)

            if time_mean is not None:
                model_names.append(model_name)  # Use full model name
                time_means.append(time_mean / 60)  # Convert to minutes
                time_stds.append((time_std or 0) / 60)
                colors.append(self._get_model_color(model_name, i))

        if not model_names:
            print("No successful runs to plot")
            plt.close()
            return ""

        x = np.arange(len(model_names))
        bars = ax.bar(x, time_means, color=colors)

        # Add value labels
        for bar, mean in zip(bars, time_means):
            height = bar.get_height()
            label = f'{mean:.1f}m'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'],
                        fontweight='bold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Avg Time to First Improvement (minutes)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Avg Time to First Validated Improvement{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.set_ylim(0, max(time_means) * 1.3)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_speed_vs_impact(
        self,
        report: Dict,
        save_name: str = "speed_vs_impact"
    ) -> str:
        """
        Plot scatter: Speed (# proposals) vs Impact (score) with full model names.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 9))

        # Collect data points first
        points = []
        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            proposals = model_result.get("proposals_to_improvement_mean") or model_result.get("iterations_to_improvement_mean")
            score = model_result.get("averaged_score_mean")

            if proposals is not None and score is not None:
                color = self._get_model_color(model_name, i)
                points.append((proposals, score * 100, model_name, color))

        if not points:
            plt.close()
            return ""

        # Calculate axis limits with padding for labels
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_padding = (x_max - x_min) * 0.25 if x_max > x_min else 5
        y_padding = (y_max - y_min) * 0.25 if y_max > y_min else 1

        # Plot points
        for proposals, score, model_name, color in points:
            ax.scatter(proposals, score, s=200, c=[color],
                      edgecolors='black', linewidths=1.5, zorder=5)

        # Add labels to the right of each point
        for proposals, score, model_name, color in points:
            ax.annotate(model_name,
                       xy=(proposals, score),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=self.fontsize['annotation'],
                       fontweight='bold',
                       ha='left', va='center')

        # Set axis limits with padding
        ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

        # Formatting
        ax.set_xlabel('Avg # Proposals to First Improvement',
                     fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Avg Impact Score (%)',
                     fontsize=self.fontsize['axis_label'])
        ax.set_title(f'# Proposals vs Impact Trade-off{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.tick_params(axis='both', labelsize=self.fontsize['tick'])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_success_rate(
        self,
        report: Dict,
        save_name: str = "success_rate"
    ) -> str:
        """
        Plot bar chart showing success rate per model.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))

        model_names = []
        success_rates = []
        colors = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            success_rate = model_result.get("success_rate", 0)

            model_names.append(model_name)  # Use full model name
            success_rates.append(success_rate * 100)
            colors.append(self._get_model_color(model_name, i))

        x = np.arange(len(model_names))
        bars = ax.bar(x, success_rates, color=colors)

        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.annotate(f'{rate:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'],
                        fontweight='bold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Success Rate (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Success Rate: Finding Validated Improvement{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.set_ylim(0, 110)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_improvement_by_type(
        self,
        report: Dict,
        save_name: str = "improvement_by_type"
    ) -> str:
        """
        Plot grouped bar chart showing improvement magnitude by hypothesis type per model.

        Shows which improvement types (resource/calendar/shift_responsibility) each model
        excels at proposing.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        # Improvement type colors
        type_colors = {
            'resource_allocation': '#0072B2',  # Blue
            'calendar_adjustment': '#E69F00',  # Orange
            'shift_responsibility': '#009E73',        # Green
        }

        # Collect scores by model and type (excluding unknown)
        model_type_scores = {}
        all_types = set()

        for model_result in results:
            model_name = model_result.get("model_name", "Unknown")
            model_type_scores[model_name] = {}

            runs = model_result.get("runs", [])
            for run in runs:
                iterations = run.get("iterations", [])
                for it in iterations:
                    if it.get("is_improvement", False):
                        hyp_type = it.get("hypothesis_type", "unknown")
                        # Skip unknown types
                        if hyp_type == "unknown":
                            continue
                        score = it.get("kpi_impact", {}).get("averaged_score")
                        if score is not None:
                            all_types.add(hyp_type)
                            if hyp_type not in model_type_scores[model_name]:
                                model_type_scores[model_name][hyp_type] = []
                            model_type_scores[model_name][hyp_type].append(score * 100)

        if not all_types:
            print("No improvement types to plot")
            plt.close()
            return ""

        # Sort types for consistent ordering (exclude unknown)
        type_order = ['resource_allocation', 'calendar_adjustment', 'shift_responsibility']
        types = [t for t in type_order if t in all_types]
        types.extend([t for t in all_types if t not in type_order and t != 'unknown'])

        fig, ax = plt.subplots(figsize=(14, 8))

        model_names = list(model_type_scores.keys())
        x = np.arange(len(model_names))
        width = 0.8 / len(types) if types else 0.8

        # Plot bars for each type
        for i, hyp_type in enumerate(types):
            means = []
            stds = []
            for model in model_names:
                scores = model_type_scores[model].get(hyp_type, [])
                if scores:
                    means.append(np.mean(scores))
                    stds.append(np.std(scores) if len(scores) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - len(types) / 2 + 0.5) * width
            color = type_colors.get(hyp_type, '#999999')
            label = hyp_type.replace('_', ' ').title()

            bars = ax.bar(x + offset, means, width, label=label,
                         color=color, alpha=0.8, yerr=stds if any(stds) else None,
                         capsize=3)

        # Add 1% threshold line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='1% Threshold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Avg Improvement Magnitude (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Improvement Magnitude by Hypothesis Type{self._get_title_suffix()}\n'
                     '(Which improvement types does each model propose effectively?)',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_best_improvement_comparison(
        self,
        report: Dict,
        save_name: str = "best_improvement_comparison"
    ) -> str:
        """
        Plot bar chart comparing the BEST improvement found by each model.

        Shows the maximum improvement magnitude achieved by each model,
        useful for understanding the upper potential of each LLM.
        Only shows hypothesis type if it's known (not 'unknown').

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))

        model_names = []
        best_scores = []
        colors = []
        best_hypotheses = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            runs = model_result.get("runs", [])

            # Find the best improvement across all runs
            best_score = None
            best_hyp = None
            for run in runs:
                iterations = run.get("iterations", [])
                for it in iterations:
                    if it.get("is_improvement", False):
                        score = it.get("kpi_impact", {}).get("averaged_score")
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score
                            best_hyp = it.get("hypothesis_type", "unknown")

            if best_score is not None:
                model_names.append(model_name)  # Use full model name
                best_scores.append(best_score * 100)  # Convert to percentage
                colors.append(self._get_model_color(model_name, i))
                best_hypotheses.append(best_hyp)

        if not model_names:
            print("No improvements to plot")
            plt.close()
            return ""

        x = np.arange(len(model_names))
        bars = ax.bar(x, best_scores, color=colors)

        # Add value labels with hypothesis type (only if not 'unknown')
        for bar, score, hyp in zip(bars, best_scores, best_hypotheses):
            height = bar.get_height()
            if hyp and hyp != "unknown":
                hyp_short = hyp.replace('_', ' ').replace('allocation', '').replace('adjustment', '').strip()
                label = f'{score:.2f}%\n({hyp_short})'
            else:
                label = f'{score:.2f}%'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'] - 1,
                        fontweight='bold')

        # Add 1% threshold line
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label='1% Threshold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Best Improvement Magnitude (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Best Validated Improvement per Model{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'])
        ax.set_ylim(0, max(best_scores) * 1.3 if best_scores else 10)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_avg_improvement_comparison(
        self,
        report: Dict,
        save_name: str = "avg_improvement_comparison"
    ) -> str:
        """
        Plot bar chart comparing the AVERAGE improvement found by each model.

        Shows the mean improvement magnitude achieved by each model across all
        validated improvements.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 7))

        model_names = []
        avg_scores = []
        colors = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            # Use averaged_score_mean which is already computed
            avg_score = model_result.get("averaged_score_mean")

            if avg_score is not None:
                model_names.append(model_name)
                avg_scores.append(avg_score * 100)  # Convert to percentage
                colors.append(self._get_model_color(model_name, i))

        if not model_names:
            print("No improvements to plot")
            plt.close()
            return ""

        x = np.arange(len(model_names))
        bars = ax.bar(x, avg_scores, color=colors)

        # Add value labels
        for bar, score in zip(bars, avg_scores):
            height = bar.get_height()
            label = f'{score:.2f}%'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'],
                        fontweight='bold')

        # Add 1% threshold line
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label='1% Threshold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Avg Improvement Magnitude (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(f'Average Validated Improvement per Model{self._get_title_suffix()}',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])
        ax.legend(loc='upper right', fontsize=self.fontsize['legend'])
        ax.set_ylim(0, max(avg_scores) * 1.3 if avg_scores else 10)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def plot_first_vs_best_score(
        self,
        report: Dict,
        save_name: str = "first_vs_best_score"
    ) -> str:
        """
        Plot grouped bar chart comparing first simulated proposal score vs best/final score.

        Shows how iteration improves the initial proposal - demonstrating that even if
        the first proposal wasn't good enough (below 1% threshold), subsequent iterations
        can improve it.

        Args:
            report: Benchmark report dictionary
            save_name: Output filename

        Returns:
            Path to saved figure
        """
        results = report.get("results", [])
        if not results:
            return ""

        fig, ax = plt.subplots(figsize=(14, 8))

        model_names = []
        first_scores = []
        best_scores = []
        colors = []

        for i, model_result in enumerate(results):
            model_name = model_result.get("model_name", f"Model {i}")
            runs = model_result.get("runs", [])

            # Collect first and best scores across all runs
            run_first_scores = []
            run_best_scores = []

            for run in runs:
                all_proposals = run.get("all_proposals", run.get("iterations", []))

                # Find first simulated proposal (not rejected before simulation)
                first_score = None
                for prop in all_proposals:
                    if not prop.get("rejected_before_simulation", False):
                        score = prop.get("kpi_impact", {}).get("averaged_score")
                        if score is not None:
                            first_score = score
                            break

                # Find best score across all simulated proposals
                best_score = None
                for prop in all_proposals:
                    if not prop.get("rejected_before_simulation", False):
                        score = prop.get("kpi_impact", {}).get("averaged_score")
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score

                if first_score is not None:
                    run_first_scores.append(first_score)
                if best_score is not None:
                    run_best_scores.append(best_score)

            # Median across runs
            if run_first_scores and run_best_scores:
                model_names.append(model_name)
                first_scores.append(np.median(run_first_scores) * 100)  # Convert to %
                best_scores.append(np.median(run_best_scores) * 100)
                colors.append(self._get_model_color(model_name, i))

        if not model_names:
            print("No data to plot for first vs best score")
            plt.close()
            return ""

        x = np.arange(len(model_names))
        width = 0.35

        # Plot grouped bars
        bars_first = ax.bar(x - width/2, first_scores, width, label='First Proposal',
                           color=[c for c in colors], alpha=0.5, edgecolor='black', linewidth=1)
        bars_best = ax.bar(x + width/2, best_scores, width, label='Best Proposal',
                          color=[c for c in colors], alpha=1.0, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, score in zip(bars_first, first_scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'] - 1)

        for bar, score in zip(bars_best, best_scores):
            height = bar.get_height()
            ax.annotate(f'{score:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=self.fontsize['annotation'] - 1,
                        fontweight='bold')

        # Formatting
        ax.set_xlabel('Model', fontsize=self.fontsize['axis_label'])
        ax.set_ylabel('Median Performance Improvement Score (%)', fontsize=self.fontsize['axis_label'])
        ax.set_title(self._get_title_suffix().strip(' ()') if self._get_title_suffix() else '',
                     fontsize=self.fontsize['title'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right',
                          fontsize=self.fontsize['tick'])
        ax.tick_params(axis='y', labelsize=self.fontsize['tick'])

        # Set y limits to show negative scores if any
        y_min = min(min(first_scores), 0) - 0.5
        y_max = max(best_scores) * 1.3
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {fig_path}")
        return fig_path

    def generate_all_plots(
        self,
        report_path: str
    ) -> List[str]:
        """
        Generate all benchmark visualizations from a saved report.

        Args:
            report_path: Path to benchmark_results.json

        Returns:
            List of generated figure paths
        """
        with open(report_path, 'r') as f:
            report = json.load(f)

        # Extract log name from report if not already set
        if not self.log_name:
            event_log = report.get("event_log", "")
            if event_log:
                self.log_name = os.path.basename(event_log)

        figures = []

        print("\nGenerating LLM benchmark visualizations...")

        # Score progression
        fig_path = self.plot_score_progression(report)
        if fig_path:
            figures.append(fig_path)

        # Bar charts
        fig_path = self.plot_proposals_comparison(report)
        if fig_path:
            figures.append(fig_path)

        fig_path = self.plot_time_comparison(report)
        if fig_path:
            figures.append(fig_path)

        # Scatter plot
        fig_path = self.plot_speed_vs_impact(report)
        if fig_path:
            figures.append(fig_path)

        # Success rate
        fig_path = self.plot_success_rate(report)
        if fig_path:
            figures.append(fig_path)

        # Improvement by type
        fig_path = self.plot_improvement_by_type(report)
        if fig_path:
            figures.append(fig_path)

        # Best improvement comparison
        fig_path = self.plot_best_improvement_comparison(report)
        if fig_path:
            figures.append(fig_path)

        # Average improvement comparison
        fig_path = self.plot_avg_improvement_comparison(report)
        if fig_path:
            figures.append(fig_path)

        # First vs Best score comparison (iteration improvement)
        fig_path = self.plot_first_vs_best_score(report)
        if fig_path:
            figures.append(fig_path)

        print(f"\nGenerated {len(figures)} visualizations")
        return figures


# ============================================================================
# Aggregated Cross-Log Visualizations
# ============================================================================

# Standard model order for consistent visualization across logs
MODEL_ORDER = [
    'gpt-4.1',
    'gpt-5-nano',
    'gpt-5-mini',
    'gpt-5',
    'o3',
    'anthropic--claude-4.5-haiku',
    'anthropic--claude-4.5-sonnet',
    'anthropic--claude-4.5-opus',
    'mistralai--mistral-small-instruct',
    'mistralai--mistral-large-instruct',
]

# Log directory to display name mapping
LOG_DISPLAY_NAMES = {
    'llm_benchmark_20260322_095513_loanApp': 'Loan Application',
    'llm_benchmark_20260328_124047_bpic12': 'BPIC12W',
    'llm_benchmark_20260407_132324': 'BPIC17W',
    'llm_benchmark_20260403_122407_o2c': 'SAP O2C',
}

# Log positions in 2x2 grid
LOG_POSITIONS = {
    'llm_benchmark_20260322_095513_loanApp': 0,  # top left
    'llm_benchmark_20260328_124047_bpic12': 1,   # top right
    'llm_benchmark_20260407_132324': 2,          # bottom left
    'llm_benchmark_20260403_122407_o2c': 3,      # bottom right
}

# Model colors (colorblind-friendly)
MODEL_COLORS = {
    'gpt-4.1': '#1E90FF',
    'gpt-5': '#0072B2',
    'gpt-5-mini': '#56B4E9',
    'gpt-5-nano': '#ADD8E6',
    'o3': '#003366',
    'anthropic--claude-4.5-opus': '#D55E00',
    'anthropic--claude-4.5-sonnet': '#E69F00',
    'anthropic--claude-4.5-haiku': '#F0E442',
    'mistralai--mistral-large-instruct': '#009E73',
    'mistralai--mistral-small-instruct': '#B2DF8A',
}

# DES/ABS colors
DES_COLOR = '#0072B2'  # Blue
ABS_COLOR = '#D55E00'  # Orange/Vermillion


def generate_aggregated_first_vs_best(output_dir: str, save_name: str = "first_vs_best_score_all_logs") -> str:
    """
    Generate aggregated first vs best proposal score chart across all event logs.

    Creates a 2x2 grid with one subplot per event log, showing median first and best
    proposal scores for each model.

    Args:
        output_dir: Directory containing benchmark result folders and where to save output
        save_name: Output filename (without extension)

    Returns:
        Path to saved figure
    """
    import json
    import os

    # Collect data per log
    logs_data = [None] * 4

    for bench_dir in os.listdir(output_dir):
        if not bench_dir.startswith('llm_benchmark_'):
            continue
        if bench_dir not in LOG_POSITIONS:
            continue

        results_path = os.path.join(output_dir, bench_dir, 'benchmark_results.json')
        if not os.path.exists(results_path):
            continue

        with open(results_path, 'r') as f:
            report = json.load(f)

        display_name = LOG_DISPLAY_NAMES.get(bench_dir, bench_dir)
        pos = LOG_POSITIONS[bench_dir]
        log_data = {'log_name': display_name, 'models': {}}

        for model_result in report.get('results', []):
            model_name = model_result.get('model_name', 'Unknown')
            runs = model_result.get('runs', [])

            first_scores = []
            best_scores = []

            for run in runs:
                all_proposals = run.get('all_proposals', run.get('iterations', []))

                # Find first simulated proposal score
                first_score = None
                for prop in all_proposals:
                    if not prop.get('rejected_before_simulation', False):
                        score = prop.get('kpi_impact', {}).get('averaged_score')
                        if score is not None:
                            first_score = score
                            break

                # Find best score
                best_score = None
                for prop in all_proposals:
                    if not prop.get('rejected_before_simulation', False):
                        score = prop.get('kpi_impact', {}).get('averaged_score')
                        if score is not None and (best_score is None or score > best_score):
                            best_score = score

                if first_score is not None:
                    first_scores.append(first_score)
                if best_score is not None:
                    best_scores.append(best_score)

            if first_scores and best_scores:
                log_data['models'][model_name] = {
                    'first': np.median(first_scores) * 100,
                    'best': np.median(best_scores) * 100
                }

        logs_data[pos] = log_data
        print(f'  {display_name}: {len(log_data["models"])} models')

    # Create 2x2 subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    fallback_colors = plt.cm.tab10.colors

    for idx, log_data in enumerate(logs_data):
        if log_data is None:
            continue
        ax = axes[idx]

        # Use MODEL_ORDER, only include models that exist in this log
        models = [m for m in MODEL_ORDER if m in log_data['models']]
        first_vals = [log_data['models'][m]['first'] for m in models]
        best_vals = [log_data['models'][m]['best'] for m in models]
        colors = [MODEL_COLORS.get(m, fallback_colors[i % len(fallback_colors)]) for i, m in enumerate(models)]

        x = np.arange(len(models))
        width = 0.35

        # First proposal (lighter)
        ax.bar(x - width/2, first_vals, width,
               color=colors, alpha=0.5, edgecolor='black', linewidth=0.5)
        # Best proposal (solid)
        ax.bar(x + width/2, best_vals, width,
               color=colors, alpha=1.0, edgecolor='black', linewidth=0.5)

        # Value labels
        for i, (first, best) in enumerate(zip(first_vals, best_vals)):
            ax.annotate(f'{first:.1f}',
                        xy=(x[i] - width/2, first),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)
            ax.annotate(f'{best:.1f}',
                        xy=(x[i] + width/2, best),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Median Performance Improvement Score (%)', fontsize=10)
        ax.set_title(log_data['log_name'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    # Save as PDF
    fig_path = os.path.join(output_dir, f'{save_name}.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {fig_path}')
    return fig_path


def generate_aggregated_des_vs_abs(output_dir: str, save_name: str = "des_vs_abs_scores") -> str:
    """
    Generate aggregated DES vs ABS score chart across all event logs.

    Creates a 2x2 grid with one subplot per event log, showing median DES and ABS
    scores for each model.

    Args:
        output_dir: Directory containing benchmark result folders and where to save output
        save_name: Output filename (without extension)

    Returns:
        Path to saved figure
    """
    import json
    import os

    # Collect data per log
    logs_data = [None] * 4

    for bench_dir in os.listdir(output_dir):
        if not bench_dir.startswith('llm_benchmark_'):
            continue
        if bench_dir not in LOG_POSITIONS:
            continue

        results_path = os.path.join(output_dir, bench_dir, 'benchmark_results.json')
        if not os.path.exists(results_path):
            continue

        with open(results_path, 'r') as f:
            report = json.load(f)

        display_name = LOG_DISPLAY_NAMES.get(bench_dir, bench_dir)
        pos = LOG_POSITIONS[bench_dir]
        log_data = {'log_name': display_name, 'models': {}}

        for model_result in report.get('results', []):
            model_name = model_result.get('model_name', 'Unknown')
            runs = model_result.get('runs', [])

            des_scores = []
            abs_scores = []

            for run in runs:
                all_proposals = run.get('all_proposals', run.get('iterations', []))

                # Find best DES and ABS scores
                best_des = None
                best_abs = None
                for prop in all_proposals:
                    if not prop.get('rejected_before_simulation', False):
                        kpi = prop.get('kpi_impact', {})
                        des = kpi.get('des', {}).get('score')
                        abs_s = kpi.get('abs', {}).get('score')
                        if des is not None and (best_des is None or des > best_des):
                            best_des = des
                        if abs_s is not None and (best_abs is None or abs_s > best_abs):
                            best_abs = abs_s

                if best_des is not None:
                    des_scores.append(best_des)
                if best_abs is not None:
                    abs_scores.append(best_abs)

            if des_scores and abs_scores:
                log_data['models'][model_name] = {
                    'des': np.median(des_scores) * 100,
                    'abs': np.median(abs_scores) * 100
                }

        logs_data[pos] = log_data
        print(f'  {display_name}: {len(log_data["models"])} models')

    # Create 2x2 subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, log_data in enumerate(logs_data):
        if log_data is None:
            continue
        ax = axes[idx]

        # Use MODEL_ORDER, only include models that exist in this log
        models = [m for m in MODEL_ORDER if m in log_data['models']]
        des_vals = [log_data['models'][m]['des'] for m in models]
        abs_vals = [log_data['models'][m]['abs'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        # DES bars (Control-Flow-first)
        ax.bar(x - width/2, des_vals, width, label='Control-Flow-first',
               color=DES_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
        # ABS bars (Resource-first)
        ax.bar(x + width/2, abs_vals, width, label='Resource-first',
               color=ABS_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Value labels
        for i, (des, abs_v) in enumerate(zip(des_vals, abs_vals)):
            ax.annotate(f'{des:.1f}',
                        xy=(x[i] - width/2, des),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)
            ax.annotate(f'{abs_v:.1f}',
                        xy=(x[i] + width/2, abs_v),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Median Performance Improvement Score (%)', fontsize=10)
        ax.set_title(log_data['log_name'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)

        # Legend: upper left for BPIC12W (idx=1) and SAP O2C (idx=3), upper right for others
        if idx in [1, 3]:
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save as PDF
    fig_path = os.path.join(output_dir, f'{save_name}.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {fig_path}')
    return fig_path


def generate_all_aggregated_charts(output_dir: str) -> List[str]:
    """
    Generate all aggregated cross-log charts.

    Args:
        output_dir: Directory containing benchmark result folders

    Returns:
        List of generated figure paths
    """
    print("\nGenerating aggregated cross-log visualizations...")
    figures = []

    fig_path = generate_aggregated_first_vs_best(output_dir)
    if fig_path:
        figures.append(fig_path)

    fig_path = generate_aggregated_des_vs_abs(output_dir)
    if fig_path:
        figures.append(fig_path)

    print(f"\nGenerated {len(figures)} aggregated visualizations")
    return figures
