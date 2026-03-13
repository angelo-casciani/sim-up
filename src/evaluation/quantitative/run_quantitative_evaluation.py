#!/usr/bin/env python3
"""
Run Comprehensive Evaluation

Entry point for running the full evaluation framework across all improvement types:
1. Resource Allocation
2. Calendar Adjustment
3. Role Redesign

Usage:
    python run_comprehensive_evaluation.py --logs-dir data/logs --output-dir output
    python run_comprehensive_evaluation.py --help
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from evaluation.quantitative.quantitative_evaluation_runner import (
    ComprehensiveEvaluator,
    ImprovementType,
    ExperimentConfig,
    InterventionCost
)
from evaluation.quantitative.quantitative_evaluation_results_reporter import ComprehensiveResultsReporter
from evaluation.quantitative.results_visualizer import ResultsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation framework for LLM-guided process improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation with defaults
  python run_comprehensive_evaluation.py --logs-dir data/logs --output-dir output

  # Run with specific model and more repetitions
  python run_comprehensive_evaluation.py --logs-dir data/logs --model claude-3-opus --repetitions 10

  # Run only resource allocation experiments
  python run_comprehensive_evaluation.py --logs-dir data/logs --type resource_allocation
        """
    )

    parser.add_argument(
        "--logs-dir", "-l",
        default="logs",
        help="Directory containing event logs (default: logs)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for results (default: output)"
    )

    parser.add_argument(
        "--model", "-m",
        default="anthropic--claude-4.5-opus",
        help="LLM model name (default: anthropic--claude-4.5-opus)"
    )

    parser.add_argument(
        "--repetitions", "-r",
        type=int,
        default=5,
        help="Number of repetitions per experiment (default: 5)"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="KPI tolerance percentage for convergence (default: 5.0)"
    )

    parser.add_argument(
        "--des-runs",
        type=int,
        default=5,
        help="Number of DES (Prosimos) simulation runs to average (default: 5)"
    )

    parser.add_argument(
        "--abs-runs",
        type=int,
        default=5,
        help="Number of ABS (AgentSimulator) simulation runs (default: 5)"
    )

    parser.add_argument(
        "--no-dual-validation",
        action="store_true",
        help="Disable dual-simulator validation (DES + ABS)"
    )

    parser.add_argument(
        "--type", "-t",
        choices=["all", "resource_allocation", "calendar_adjustment", "role_redesign"],
        default="all",
        help="Improvement type to evaluate (default: all)"
    )

    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Don't use sudo for Docker commands"
    )

    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip generating visualization plots"
    )

    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Skip running grid search, use theoretical iterations instead (faster evaluation)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running evaluation"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.logs_dir):
        print(f"Error: Logs directory not found: {args.logs_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter experiments by type if specified
    experiments = None
    if args.type != "all":
        type_map = {
            "resource_allocation": ImprovementType.RESOURCE_ALLOCATION,
            "calendar_adjustment": ImprovementType.CALENDAR_ADJUSTMENT,
            "role_redesign": ImprovementType.ROLE_REDESIGN
        }
        target_type = type_map[args.type]
        experiments = [
            exp for exp in ComprehensiveEvaluator.DEFAULT_EXPERIMENTS
            if exp.improvement_type == target_type
        ]
        print(f"\nFiltered to {len(experiments)} {args.type} experiments")

    # Print configuration
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"  Logs directory:     {args.logs_dir}")
    print(f"  Output directory:   {args.output_dir}")
    print(f"  LLM model:          {args.model}")
    print(f"  Repetitions:        {args.repetitions}")
    print(f"  DES runs:           {args.des_runs}")
    print(f"  ABS runs:           {args.abs_runs}")
    print(f"  Dual validation:    {not args.no_dual_validation}")
    print(f"  Tolerance:          {args.tolerance}%")
    print(f"  Improvement type:   {args.type}")
    print(f"  Use sudo:           {not args.no_sudo}")
    print(f"  Skip grid search:   {args.skip_grid_search}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. Exiting without running evaluation.")
        return

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        model_name=args.model,
        temperature=0.0,
        use_sudo=not args.no_sudo,
        num_repetitions=args.repetitions,
        tolerance_pct=args.tolerance,
        des_runs=args.des_runs,
        abs_runs=args.abs_runs,
        use_dual_validation=not args.no_dual_validation,
        skip_grid_search=args.skip_grid_search
    )

    # Run evaluation
    print("\nStarting evaluation...")
    report = evaluator.run_full_evaluation(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        experiments=experiments
    )

    # Generate reports
    print("\nGenerating reports...")
    report_data = report.to_dict()

    eval_dir = os.path.join(args.output_dir, f"comprehensive_eval_{report.timestamp}")
    reporter = ComprehensiveResultsReporter(eval_dir)
    reporter.generate_full_report(report_data)

    # Generate visualizations
    if not args.skip_visualization:
        print("\nGenerating visualizations...")
        viz_dir = os.path.join(eval_dir, "figures")
        visualizer = ResultsVisualizer(viz_dir)

        # Load the full report for visualization
        report_file = os.path.join(eval_dir, "evaluation_report.json")
        if os.path.exists(report_file):
            visualizer.generate_all_plots(report_file, tolerance_pct=args.tolerance)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
