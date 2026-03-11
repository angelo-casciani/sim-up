"""
Main entry point for Quantitative Evaluation.

Run the LLM vs Grid Search comparison experiment:
    python -m evaluation.quantitative.run_evaluation --model gpt-4.1

This script:
1. Runs SIMOD discovery on each log to get M_opt
2. Injects fault (reduce resource count to 1)
3. Races Grid Search vs LLM Search to find the fix
4. Generates charts comparing simulation calls needed
"""

import os
import sys
import platform
from argparse import ArgumentParser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.quantitative.evaluation_runner import QuantitativeEvaluator
from evaluation.quantitative.results_visualizer import ResultsVisualizer


def parse_args():
    parser = ArgumentParser(description="Quantitative Evaluation: LLM vs Grid Search")
    is_windows = platform.system() == "Windows"

    parser.add_argument('--model', type=str, default='gpt-4.1',
                        help='LLM model for guided search (default: gpt-4.1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (default: 0.0)')
    parser.add_argument('--grid-min', type=int, default=1,
                        help='Min resource count for grid search (default: 1)')
    parser.add_argument('--grid-max', type=int, default=10,
                        help='Max resource count for grid search (default: 10)')
    parser.add_argument('--fault-count', type=int, default=1,
                        help='Resource count to inject as fault (default: 1)')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Tolerance %% for solution matching (default: 2.0)')
    parser.add_argument('--max-llm-iters', type=int, default=10,
                        help='Max LLM search iterations (default: 10)')
    parser.add_argument('--logs', type=str, nargs='*', default=None,
                        help='Specific log files to evaluate (default: all)')
    parser.add_argument('--no-sudo', action='store_true', default=is_windows,
                        help='Do not use sudo for Docker commands (auto-detected: disabled on Windows)')
    parser.add_argument('--visualize-only', type=str, default=None,
                        help='Skip evaluation, only visualize from existing summary.json')

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    logs_dir = os.path.join(project_root, "logs")
    output_dir = os.path.join(project_root, "reports", "evaluation")

    # Visualization only mode
    if args.visualize_only:
        print("\n" + "=" * 60)
        print("VISUALIZATION MODE")
        print("=" * 60)

        viz = ResultsVisualizer(output_dir=os.path.dirname(args.visualize_only))
        viz.generate_all_plots(args.visualize_only)
        return

    # Full evaluation mode
    use_sudo = not args.no_sudo
    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION: LLM vs Grid Search")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Grid search range: {args.grid_min}-{args.grid_max}")
    print(f"  Fault injection: reduce to {args.fault_count} resource(s)")
    print(f"  Solution tolerance: {args.tolerance}%")
    print(f"  Max LLM iterations: {args.max_llm_iters}")
    print(f"  Use sudo: {use_sudo}")
    print(f"  Logs directory: {logs_dir}")
    print(f"  Output directory: {output_dir}")

    # Create evaluator
    evaluator = QuantitativeEvaluator(
        model_name=args.model,
        temperature=args.temperature,
        use_sudo=use_sudo,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        fault_count=args.fault_count,
        tolerance_pct=args.tolerance,
        max_llm_iterations=args.max_llm_iters
    )

    # Run evaluation
    summary = evaluator.run_evaluation(
        logs_dir=logs_dir,
        output_dir=output_dir,
        logs=args.logs
    )

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 60)

    viz_output = os.path.join(output_dir, f"quantitative_eval_{summary.timestamp}", "figures")
    viz = ResultsVisualizer(output_dir=viz_output)

    # Individual plots (convergence + trade-off)
    for exp in summary.experiments:
        viz.plot_single_experiment(exp.to_dict())
        viz.plot_resource_tradeoff(exp.to_dict())

    # Aggregate comparison (iterations to solution with error bars)
    viz.plot_aggregate_comparison(summary.to_dict())

    # Convergence overlays for all three KPIs (with log scale)
    for metric in ["cycle_time", "throughput", "waiting_time"]:
        viz.plot_convergence_overlay(summary.to_dict(), metric=metric, tolerance_pct=args.tolerance)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
