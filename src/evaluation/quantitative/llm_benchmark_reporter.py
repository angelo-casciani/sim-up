"""
LLM Benchmark Reporter

Generates reports from LLM benchmark results in multiple formats:
- JSON: Full structured data
- CSV: Spreadsheet-friendly export
- Markdown: Human-readable summary with rankings
- Figures: Publication-quality visualizations
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

from evaluation.quantitative.llm_benchmark_runner import (
    LLMBenchmarkReport,
    ModelBenchmarkResult
)
from evaluation.quantitative.llm_benchmark_visualizer import LLMBenchmarkVisualizer


class LLMBenchmarkReporter:
    """Generate reports from LLM benchmark results."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_full_report(self, report: LLMBenchmarkReport) -> str:
        """
        Generate all report formats.

        Returns:
            Path to the evaluation directory containing all reports.
        """
        eval_dir = os.path.join(self.output_dir, f"llm_benchmark_{report.timestamp}")
        os.makedirs(eval_dir, exist_ok=True)

        # Generate all formats
        self._generate_json(report, eval_dir)
        self._generate_csv(report, eval_dir)
        self._generate_markdown(report, eval_dir)

        # Generate visualizations
        self._generate_figures(report, eval_dir)

        print(f"\nReports saved to: {eval_dir}")
        return eval_dir

    def _generate_figures(self, report: LLMBenchmarkReport, output_dir: str):
        """Generate publication-quality visualizations."""
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        try:
            visualizer = LLMBenchmarkVisualizer(figures_dir)
            report_dict = report.to_dict()
            figures = []

            # Generate all plots
            fig = visualizer.plot_score_progression(report_dict)
            if fig:
                figures.append(fig)

            fig = visualizer.plot_proposals_comparison(report_dict)
            if fig:
                figures.append(fig)

            fig = visualizer.plot_time_comparison(report_dict)
            if fig:
                figures.append(fig)

            fig = visualizer.plot_speed_vs_impact(report_dict)
            if fig:
                figures.append(fig)

            fig = visualizer.plot_success_rate(report_dict)
            if fig:
                figures.append(fig)

            # Improvement analysis plots
            fig = visualizer.plot_improvement_by_type(report_dict)
            if fig:
                figures.append(fig)

            fig = visualizer.plot_best_improvement_comparison(report_dict)
            if fig:
                figures.append(fig)

            print(f"  - Figures: {len(figures)} charts in figures/")
        except Exception as e:
            print(f"  - Figures: Failed to generate ({e})")

    def _generate_json(self, report: LLMBenchmarkReport, output_dir: str):
        """Generate full results JSON."""
        json_path = os.path.join(output_dir, "benchmark_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  - JSON: benchmark_results.json")

    def _generate_csv(self, report: LLMBenchmarkReport, output_dir: str):
        """Generate CSV export for spreadsheet analysis."""
        csv_path = os.path.join(output_dir, "benchmark_results.csv")

        # Header
        headers = [
            "model_name",
            "run_id",
            "status",
            "time_to_first_improvement",
            "proposals_to_first_improvement",
            "iterations_to_first_improvement",
            "averaged_score",
            "des_score",
            "abs_score",
            "total_time",
            "total_proposals",
            "total_iterations",
            "first_improvement_hypothesis",
            "error_message"
        ]

        lines = [",".join(headers)]

        # Data rows
        for model_result in report.results:
            for run in model_result.runs:
                row = [
                    self._csv_escape(run.model_name),
                    str(run.run_id),
                    run.status,
                    self._format_float(run.time_to_first_improvement),
                    str(run.proposals_to_first_improvement) if run.proposals_to_first_improvement is not None else "",
                    str(run.iterations_to_first_improvement) if run.iterations_to_first_improvement is not None else "",
                    self._format_float(run.averaged_score),
                    self._format_float(run.des_score),
                    self._format_float(run.abs_score),
                    self._format_float(run.total_time),
                    str(run.total_proposals) if run.total_proposals else "",
                    str(run.total_iterations) if run.total_iterations else "",
                    self._csv_escape(run.first_improvement_hypothesis or ""),
                    self._csv_escape(run.error_message or "")
                ]
                lines.append(",".join(row))

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"  - CSV: benchmark_results.csv")

    def _generate_markdown(self, report: LLMBenchmarkReport, output_dir: str):
        """Generate markdown summary with rankings."""
        md_path = os.path.join(output_dir, "benchmark_report.md")

        lines = [
            "# LLM Benchmark Results",
            "",
            f"**Timestamp:** {report.timestamp}",
            f"**Event Log:** {os.path.basename(report.event_log)}",
            f"**Repetitions per Model:** {report.num_repetitions}",
            f"**Max Iterations:** {report.max_iterations}",
            "",
            "---",
            "",
            "## Summary Table",
            "",
            "| Model | Success Rate | Avg Time (s) | Avg Proposals | Avg Score |",
            "|-------|--------------|--------------|---------------|-----------|",
        ]

        for result in report.results:
            success_rate = f"{result.success_rate*100:.0f}%"
            avg_time = self._format_stat(result.time_to_improvement_mean, result.time_to_improvement_std, 1)
            # Use proposals if available, otherwise fall back to iterations
            avg_proposals = self._format_stat(
                result.proposals_to_improvement_mean or result.iterations_to_improvement_mean,
                result.proposals_to_improvement_std or result.iterations_to_improvement_std,
                2
            )
            avg_score = self._format_stat(result.averaged_score_mean, result.averaged_score_std, 4)

            lines.append(f"| {result.model_name} | {success_rate} | {avg_time} | {avg_proposals} | {avg_score} |")

        # Rankings section
        lines.extend([
            "",
            "---",
            "",
            "## Rankings",
            "",
        ])

        # Speed ranking
        lines.extend([
            "### Speed Ranking (Time to First Improvement)",
            "",
            "| Rank | Model | Avg Time (s) |",
            "|------|-------|--------------|",
        ])

        for i, model_name in enumerate(report.speed_ranking, 1):
            result = next((r for r in report.results if r.model_name == model_name), None)
            if result:
                time_str = self._format_stat(result.time_to_improvement_mean, result.time_to_improvement_std, 1)
                lines.append(f"| {i} | {model_name} | {time_str} |")

        if not report.speed_ranking:
            lines.append("| - | No successful runs | - |")

        # Proposals ranking
        lines.extend([
            "",
            "### Proposals Ranking (# Proposals to First Improvement)",
            "",
            "| Rank | Model | Avg # Proposals |",
            "|------|-------|-----------------|",
        ])

        proposals_ranking = getattr(report, 'proposals_ranking', []) or report.iteration_ranking
        for i, model_name in enumerate(proposals_ranking, 1):
            result = next((r for r in report.results if r.model_name == model_name), None)
            if result:
                proposals_str = self._format_stat(
                    result.proposals_to_improvement_mean or result.iterations_to_improvement_mean,
                    result.proposals_to_improvement_std or result.iterations_to_improvement_std,
                    2
                )
                lines.append(f"| {i} | {model_name} | {proposals_str} |")

        if not proposals_ranking:
            lines.append("| - | No successful runs | - |")

        # Impact ranking
        lines.extend([
            "",
            "### Impact Ranking (Validation Score)",
            "",
            "| Rank | Model | Avg Score |",
            "|------|-------|-----------|",
        ])

        for i, model_name in enumerate(report.impact_ranking, 1):
            result = next((r for r in report.results if r.model_name == model_name), None)
            if result:
                score_str = self._format_stat(result.averaged_score_mean, result.averaged_score_std, 4)
                lines.append(f"| {i} | {model_name} | {score_str} |")

        if not report.impact_ranking:
            lines.append("| - | No successful runs | - |")

        # Detailed results section
        lines.extend([
            "",
            "---",
            "",
            "## Detailed Results by Model",
            "",
        ])

        for result in report.results:
            lines.extend([
                f"### {result.model_name}",
                "",
                f"- **Runs:** {result.num_runs}",
                f"- **Successful:** {result.num_successful}",
                f"- **Errors:** {result.num_errors}",
                f"- **No Improvement:** {result.num_no_improvement}",
                "",
            ])

            if result.runs:
                lines.extend([
                    "| Run | Status | Time (s) | # Proposals | Score |",
                    "|-----|--------|----------|-------------|-------|",
                ])

                for run in result.runs:
                    time_str = f"{run.time_to_first_improvement:.1f}" if run.time_to_first_improvement else "-"
                    proposals_str = str(run.proposals_to_first_improvement or run.iterations_to_first_improvement) if (run.proposals_to_first_improvement or run.iterations_to_first_improvement) else "-"
                    score_str = f"{run.averaged_score:.4f}" if run.averaged_score else "-"
                    lines.append(f"| {run.run_id} | {run.status} | {time_str} | {proposals_str} | {score_str} |")

                lines.append("")

                # Add proposal details for each run
                for run in result.runs:
                    all_proposals = getattr(run, 'all_proposals', []) or run.iterations
                    if all_proposals:
                        lines.extend([
                            f"#### Run {run.run_id} - Proposal Details",
                            "",
                        ])

                        # Show baseline KPIs if available
                        if run.baseline_kpis:
                            lines.extend([
                                "**Baseline KPIs:**",
                                f"- Cycle Time: {run.baseline_kpis.get('avg_cycle_time', 'N/A')}",
                                f"- Throughput: {run.baseline_kpis.get('throughput', 'N/A')}",
                                f"- Waiting Time: {run.baseline_kpis.get('avg_waiting_time', 'N/A')}",
                                "",
                            ])

                        lines.extend([
                            "| # | Type | Hypothesis | Rejected? | Valid? | Score |",
                            "|---|------|------------|-----------|--------|-------|",
                        ])

                        for prop in all_proposals:
                            prop_num = getattr(prop, 'proposal_num', None) or getattr(prop, 'iteration_num', 0)
                            rejected = "Yes" if getattr(prop, 'rejected_before_simulation', False) else "No"
                            valid_str = "Yes" if prop.is_improvement else "No"
                            hyp_short = (prop.hypothesis_description[:40] + "...") if len(prop.hypothesis_description) > 40 else prop.hypothesis_description
                            score = f"{prop.averaged_score:.4f}" if prop.averaged_score is not None else "-"
                            lines.append(f"| {prop_num} | {prop.hypothesis_type} | {hyp_short} | {rejected} | {valid_str} | {score} |")

                        lines.append("")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"  - Markdown: benchmark_report.md")

    def _csv_escape(self, value: str) -> str:
        """Escape a value for CSV output."""
        if not value:
            return ""
        # If value contains comma, newline, or quote, wrap in quotes
        if ',' in value or '\n' in value or '"' in value:
            return '"' + value.replace('"', '""') + '"'
        return value

    def _format_float(self, value: float) -> str:
        """Format a float for CSV output."""
        if value is None:
            return ""
        return f"{value:.4f}"

    def _format_stat(self, mean: float, std: float, decimals: int) -> str:
        """Format a statistic with optional standard deviation."""
        if mean is None:
            return "-"

        fmt = f"{{:.{decimals}f}}"
        mean_str = fmt.format(mean)

        if std is not None and std > 0:
            std_str = fmt.format(std)
            return f"{mean_str} +/- {std_str}"

        return mean_str
