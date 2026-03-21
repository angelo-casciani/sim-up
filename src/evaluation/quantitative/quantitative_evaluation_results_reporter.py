"""
Comprehensive Results Reporter for Evaluation Framework.

Generates reports with:
- Mean ± std statistics across repetitions
- Per-improvement-type breakdowns
- Intervention cost analysis
- LaTeX table generation
- Markdown summary reports
"""

import os
import json
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime


class ComprehensiveResultsReporter:
    """Generate comprehensive evaluation reports for journal paper submission."""

    def __init__(self, output_dir: str):
        """
        Initialize reporter.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_full_report(self, report_data: Dict) -> Dict[str, str]:
        """
        Generate all report formats from evaluation data.

        Args:
            report_data: Full evaluation report dictionary

        Returns:
            Dictionary mapping report type to file path
        """
        generated = {}

        # Markdown summary report
        md_path = self._generate_markdown_report(report_data)
        if md_path:
            generated["markdown"] = md_path

        # LaTeX tables for paper
        latex_path = self._generate_latex_tables(report_data)
        if latex_path:
            generated["latex"] = latex_path

        # JSON summary with computed statistics
        json_path = self._generate_json_summary(report_data)
        if json_path:
            generated["json"] = json_path

        # CSV for spreadsheet analysis
        csv_path = self._generate_csv_export(report_data)
        if csv_path:
            generated["csv"] = csv_path

        return generated

    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate comprehensive markdown report."""
        timestamp = report_data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_name = report_data.get("model_name", "Unknown")
        num_reps = report_data.get("num_repetitions", 1)

        lines = [
            "# Comprehensive Evaluation Report",
            "",
            f"**Generated:** {timestamp}",
            f"**LLM Model:** {model_name}",
            f"**Repetitions per experiment:** {num_reps}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]

        # Aggregate statistics
        agg = report_data.get("aggregate_statistics", {})
        if agg:
            speedup = agg.get("overall_avg_speedup")
            llm_rate = agg.get("overall_llm_success_rate", 0)
            baseline_rate = agg.get("overall_baseline_success_rate", 0)

            lines.extend([
                f"- **Overall Average Speedup:** {speedup:.2f}× (LLM vs Baseline)" if speedup else "- **Overall Average Speedup:** N/A",
                f"- **LLM Success Rate:** {llm_rate:.1%}",
                f"- **Baseline Success Rate:** {baseline_rate:.1%}",
                "",
            ])

        # Results by improvement type
        lines.extend([
            "## Results by Improvement Type",
            "",
        ])

        by_type = agg.get("by_improvement_type", {})
        for type_name, stats in by_type.items():
            lines.extend([
                f"### {type_name.replace('_', ' ').title()}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Experiments | {stats.get('num_experiments', 0)} |",
                f"| Avg Speedup | {stats.get('avg_speedup', 0):.2f}× |" if stats.get('avg_speedup') else f"| Avg Speedup | N/A |",
                f"| LLM Success Rate | {stats.get('avg_llm_success_rate', 0):.1%} |",
                "",
            ])

        # Detailed results table
        lines.extend([
            "## Detailed Experiment Results",
            "",
            "| ID | Log | Type | Baseline (mean±std) | LLM (mean±std) | Speedup (mean±std) | LLM Success |",
            "|----|-----|------|---------------------|----------------|--------------------| ------------|",
        ])

        for result in report_data.get("detailed_results", []):
            exp_id = result.get("experiment_id", "")
            log_name = result.get("log_name", "")
            imp_type = result.get("improvement_type", "")

            baseline = result.get("baseline_iterations", {})
            llm = result.get("llm_iterations", {})
            speedup = result.get("speedup_ratio", {})
            success = result.get("success_rates", {})

            baseline_str = self._format_mean_std(baseline.get("mean"), baseline.get("std"))
            llm_str = self._format_mean_std(llm.get("mean"), llm.get("std"))
            speedup_str = self._format_mean_std(speedup.get("mean"), speedup.get("std"), suffix="×")
            success_str = f"{success.get('llm', 0):.0%}"

            lines.append(
                f"| {exp_id} | {log_name[:15]} | {imp_type[:10]} | {baseline_str} | {llm_str} | {speedup_str} | {success_str} |"
            )

        lines.extend([
            "",
            "## Intervention Cost Analysis",
            "",
            "| Cost Level | Description | Experiments |",
            "|------------|-------------|-------------|",
            "| 1 (Low) | Shift Responsibility - redistribute existing workforce | Shift responsibility experiments |",
            "| 2 (Medium) | Calendar Extension - increase working hours | Calendar adjustment experiments |",
            "| 3 (High) | Resource Addition - hire new staff | Resource allocation experiments |",
            "",
        ])

        # Research questions
        lines.extend([
            "## Research Question Analysis",
            "",
            "### RQ1: Simulation Efficiency",
            "",
            "*How many simulation calls does the LLM need to find an optimal improvement compared to baseline search?*",
            "",
        ])

        if agg.get("overall_avg_speedup"):
            lines.append(f"**Finding:** The LLM achieves an average speedup of **{agg['overall_avg_speedup']:.2f}×** "
                        f"compared to baseline search methods.")
        else:
            lines.append("**Finding:** Insufficient data to compute speedup.")

        lines.extend([
            "",
            "### RQ2: Bottleneck Identification",
            "",
            "*Does the LLM correctly identify bottleneck resources/roles?*",
            "",
            "See per-experiment results for role pair accuracy metrics.",
            "",
            "### RQ3: Cross-type Performance",
            "",
            "*How does LLM performance vary across different improvement types?*",
            "",
        ])

        for type_name, stats in by_type.items():
            speedup = stats.get("avg_speedup")
            success = stats.get("avg_llm_success_rate", 0)
            lines.append(f"- **{type_name.replace('_', ' ').title()}:** "
                        f"Speedup = {speedup:.2f}× | Success = {success:.1%}" if speedup else
                        f"- **{type_name.replace('_', ' ').title()}:** Success = {success:.1%}")

        lines.extend([
            "",
            "---",
            "",
            "*Report generated by Comprehensive Evaluation Framework*",
        ])

        # Write report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Saved markdown report: {report_path}")
        return report_path

    def _generate_latex_tables(self, report_data: Dict) -> str:
        """Generate LaTeX tables for paper inclusion."""
        lines = [
            "% Evaluation Results Tables",
            "% Auto-generated - do not edit manually",
            "",
            "% Main results table",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Comparison of LLM-Guided Search vs Baseline Methods}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "ID & Type & Baseline & LLM & Speedup & Success \\\\",
            "\\midrule",
        ]

        for result in report_data.get("detailed_results", []):
            exp_id = result.get("experiment_id", "")
            imp_type = result.get("improvement_type", "")[:8]

            baseline = result.get("baseline_iterations", {})
            llm = result.get("llm_iterations", {})
            speedup = result.get("speedup_ratio", {})
            success = result.get("success_rates", {})

            baseline_str = self._format_latex_mean_std(baseline.get("mean"), baseline.get("std"))
            llm_str = self._format_latex_mean_std(llm.get("mean"), llm.get("std"))
            speedup_str = self._format_latex_mean_std(speedup.get("mean"), speedup.get("std"), suffix="$\\times$")
            success_str = f"{success.get('llm', 0)*100:.0f}\\%"

            lines.append(f"{exp_id} & {imp_type} & {baseline_str} & {llm_str} & {speedup_str} & {success_str} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% Aggregate statistics by improvement type",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Results by Improvement Type}",
            "\\label{tab:by_type}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Improvement Type & Experiments & Avg Speedup & Success Rate \\\\",
            "\\midrule",
        ])

        agg = report_data.get("aggregate_statistics", {})
        by_type = agg.get("by_improvement_type", {})
        for type_name, stats in by_type.items():
            type_label = type_name.replace("_", " ").title()
            n_exp = stats.get("num_experiments", 0)
            speedup = stats.get("avg_speedup")
            success = stats.get("avg_llm_success_rate", 0)

            speedup_str = f"{speedup:.2f}$\\times$" if speedup else "N/A"
            lines.append(f"{type_label} & {n_exp} & {speedup_str} & {success*100:.0f}\\% \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        # Write LaTeX
        latex_path = os.path.join(self.output_dir, "evaluation_tables.tex")
        with open(latex_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Saved LaTeX tables: {latex_path}")
        return latex_path

    def _generate_json_summary(self, report_data: Dict) -> str:
        """Generate JSON summary with computed statistics."""
        summary = {
            "metadata": {
                "timestamp": report_data.get("timestamp"),
                "model_name": report_data.get("model_name"),
                "num_repetitions": report_data.get("num_repetitions"),
                "total_experiments": report_data.get("total_experiments")
            },
            "hypotheses": self._evaluate_hypotheses(report_data),
            "aggregate_statistics": report_data.get("aggregate_statistics", {}),
            "success_criteria": self._evaluate_success_criteria(report_data)
        }

        json_path = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved JSON summary: {json_path}")
        return json_path

    def _generate_csv_export(self, report_data: Dict) -> str:
        """Generate CSV export for spreadsheet analysis."""
        lines = [
            "experiment_id,log_name,improvement_type,intervention_cost,"
            "baseline_mean,baseline_std,llm_mean,llm_std,"
            "speedup_mean,speedup_std,baseline_success,llm_success,llm_accuracy"
        ]

        for result in report_data.get("detailed_results", []):
            exp_id = result.get("experiment_id", "")
            log_name = result.get("log_name", "")
            imp_type = result.get("improvement_type", "")
            cost = result.get("intervention_cost", "")

            baseline = result.get("baseline_iterations", {})
            llm = result.get("llm_iterations", {})
            speedup = result.get("speedup_ratio", {})
            success = result.get("success_rates", {})

            lines.append(
                f"{exp_id},{log_name},{imp_type},{cost},"
                f"{baseline.get('mean', '')},{baseline.get('std', '')},"
                f"{llm.get('mean', '')},{llm.get('std', '')},"
                f"{speedup.get('mean', '')},{speedup.get('std', '')},"
                f"{success.get('baseline', '')},{success.get('llm', '')},{success.get('llm_accuracy', '')}"
            )

        csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        with open(csv_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Saved CSV export: {csv_path}")
        return csv_path

    def _evaluate_hypotheses(self, report_data: Dict) -> Dict[str, Any]:
        """Evaluate the research hypotheses from the proposal."""
        agg = report_data.get("aggregate_statistics", {})
        by_type = agg.get("by_improvement_type", {})

        results = {}

        # H1: LLM speedup > 1.5× for resource allocation
        resource_stats = by_type.get("resource_allocation", {})
        h1_speedup = resource_stats.get("avg_speedup")
        results["H1_resource_speedup"] = {
            "hypothesis": "LLM achieves >1.5× speedup for resource allocation",
            "threshold": 1.5,
            "observed": h1_speedup,
            "supported": h1_speedup and h1_speedup > 1.5
        }

        # H2: LLM speedup > 1.5× for calendar adjustment
        calendar_stats = by_type.get("calendar_adjustment", {})
        h2_speedup = calendar_stats.get("avg_speedup")
        results["H2_calendar_speedup"] = {
            "hypothesis": "LLM achieves >1.5× speedup for calendar adjustment",
            "threshold": 1.5,
            "observed": h2_speedup,
            "supported": h2_speedup and h2_speedup > 1.5
        }

        # H3: LLM correctly identifies bottleneck in >80% of cases
        # This would need per-experiment accuracy data
        overall_success = agg.get("overall_llm_success_rate", 0)
        results["H3_bottleneck_identification"] = {
            "hypothesis": "LLM identifies bottleneck in >80% of cases",
            "threshold": 0.8,
            "observed": overall_success,
            "supported": overall_success > 0.8
        }

        # H4: LLM proposes lower-cost interventions (would need more data)
        results["H4_cost_optimization"] = {
            "hypothesis": "LLM proposes lower-cost interventions when viable",
            "observed": "Requires mixed-type evaluation",
            "supported": None
        }

        return results

    def _evaluate_success_criteria(self, report_data: Dict) -> Dict[str, Any]:
        """Evaluate success criteria from the proposal."""
        agg = report_data.get("aggregate_statistics", {})

        criteria = {
            "avg_speedup_gt_1.5": {
                "criterion": "Average speedup ratio > 1.5×",
                "threshold": 1.5,
                "observed": agg.get("overall_avg_speedup"),
                "met": (agg.get("overall_avg_speedup") or 0) > 1.5
            },
            "llm_success_gt_80": {
                "criterion": "LLM success rate > 80%",
                "threshold": 0.8,
                "observed": agg.get("overall_llm_success_rate"),
                "met": (agg.get("overall_llm_success_rate") or 0) > 0.8
            },
            "convergence_within_95": {
                "criterion": "Convergence quality within 95% of grid",
                "threshold": 0.95,
                "observed": None,  # Would need per-experiment data
                "met": None
            }
        }

        return criteria

    def _format_mean_std(
        self,
        mean: Optional[float],
        std: Optional[float],
        suffix: str = ""
    ) -> str:
        """Format mean ± std string."""
        if mean is None:
            return "N/A"
        if std is None or std == 0:
            return f"{mean:.1f}{suffix}"
        return f"{mean:.1f} ± {std:.1f}{suffix}"

    def _format_latex_mean_std(
        self,
        mean: Optional[float],
        std: Optional[float],
        suffix: str = ""
    ) -> str:
        """Format mean ± std string for LaTeX."""
        if mean is None:
            return "N/A"
        if std is None or std == 0:
            return f"${mean:.1f}${suffix}"
        return f"${mean:.1f} \\pm {std:.1f}${suffix}"


def generate_report_from_file(summary_path: str, output_dir: Optional[str] = None):
    """
    Generate comprehensive report from a saved evaluation summary.

    Args:
        summary_path: Path to evaluation_report.json
        output_dir: Output directory (default: same as input)
    """
    with open(summary_path, 'r') as f:
        report_data = json.load(f)

    if output_dir is None:
        output_dir = os.path.dirname(summary_path)

    reporter = ComprehensiveResultsReporter(output_dir)
    generated = reporter.generate_full_report(report_data)

    print(f"\nGenerated {len(generated)} report files:")
    for report_type, path in generated.items():
        print(f"  - {report_type}: {path}")

    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation reports")
    parser.add_argument("summary_path", help="Path to evaluation_report.json")
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()
    generate_report_from_file(args.summary_path, args.output_dir)
