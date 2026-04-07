#!/usr/bin/env python3
"""
LLM Benchmark Evaluation

Benchmarks multiple LLM models on the improvement validation pipeline.
Measures speed (time and iterations to first validated improvement) and
impact quality (averaged validation formula score).

Supports checkpoint/resume functionality for long-running benchmarks.

Usage:
    python run_llm_benchmark.py --log LoanApp.csv
    python run_llm_benchmark.py --log p2p.csv --repetitions 3 --timeout 1800
    python run_llm_benchmark.py --models gpt-4.1 anthropic--claude-4.5-opus --log LoanApp.csv
    python run_llm_benchmark.py --dry-run
    python run_llm_benchmark.py --resume --log LoanApp.csv  # Resume interrupted benchmark
    python run_llm_benchmark.py --status --log LoanApp.csv  # Check checkpoint status
"""

import argparse
import os
import platform
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from evaluation.quantitative.llm_benchmark_runner import (
    LLMBenchmarkRunner,
    LLMBenchmarkConfig
)
from evaluation.quantitative.llm_benchmark_reporter import LLMBenchmarkReporter
from evaluation.quantitative.benchmark_checkpoint import BenchmarkCheckpoint


def main():
    # Auto-detect Windows for sudo default
    is_windows = platform.system() == "Windows"

    parser = argparse.ArgumentParser(
        description="Benchmark LLM models on process improvement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default settings
  python run_llm_benchmark.py --log LoanApp.csv

  # Run with specific models
  python run_llm_benchmark.py --models gpt-4.1 anthropic--claude-4.5-opus --log LoanApp.csv

  # Run with multiple repetitions for statistical significance
  python run_llm_benchmark.py --log LoanApp.csv --repetitions 5

  # Run with custom timeout
  python run_llm_benchmark.py --log LoanApp.csv --timeout 1800

  # Dry run to see configuration
  python run_llm_benchmark.py --log LoanApp.csv --dry-run

  # Resume an interrupted benchmark
  python run_llm_benchmark.py --resume --log LoanApp.csv

  # Check status of a checkpoint
  python run_llm_benchmark.py --status --log LoanApp.csv

  # Clear an existing checkpoint (start fresh)
  python run_llm_benchmark.py --clear-checkpoint --log LoanApp.csv
        """
    )

    parser.add_argument(
        "--log", "-l",
        default="LoanApp.csv",
        help="Event log file in logs/ directory (default: LoanApp.csv)"
    )

    parser.add_argument(
        "--models", "-m",
        nargs="*",
        default=None,
        help="Models to benchmark (default: all configured models)"
    )

    parser.add_argument(
        "--repetitions", "-r",
        type=int,
        default=1,
        help="Number of runs per model (default: 1)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations per run (default: 5)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for results (default: output)"
    )

    parser.add_argument(
        "--no-sudo",
        action="store_true",
        default=is_windows,
        help="Don't use sudo for Docker (auto-detected on Windows)"
    )

    parser.add_argument(
        "--des-runs",
        type=int,
        default=5,
        help="DES simulation runs (default: 5)"
    )

    parser.add_argument(
        "--abs-runs",
        type=int,
        default=5,
        help="ABS simulation runs (default: 5)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running"
    )

    # Checkpoint/resume options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show checkpoint status and exit"
    )

    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and start fresh"
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing (not recommended for long runs)"
    )

    args = parser.parse_args()

    # Resolve log path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    log_path = os.path.join(project_root, "logs", args.log)

    # Also try relative to src directory
    if not os.path.exists(log_path):
        src_dir = os.path.dirname(os.path.dirname(script_dir))
        log_path = os.path.join(src_dir, "..", "logs", args.log)

    # Normalize path
    log_path = os.path.normpath(log_path)

    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        print(f"\nTried looking in:")
        print(f"  - {os.path.join(project_root, 'logs', args.log)}")
        print(f"\nMake sure the log file exists in the logs/ directory.")
        sys.exit(1)

    # Initialize checkpoint manager
    checkpoint_manager = BenchmarkCheckpoint()

    # Handle --status: show checkpoint info and exit
    if args.status:
        info = checkpoint_manager.get_checkpoint_info(log_path)
        if info:
            print("\n" + "=" * 70)
            print("CHECKPOINT STATUS")
            print("=" * 70)
            print(f"  Event log:        {args.log}")
            print(f"  Saved at:         {info['saved_at']}")
            print(f"  Progress:         {info['completed_runs']}/{info['total_runs']} runs ({info['progress_pct']:.1f}%)")
            print(f"  Completed models: {info['completed_models']}/{info['total_models']}")
            print(f"  Current model:    {info['current_model']}")
            print(f"  Current run:      {info['current_run_idx'] + 1}")
            print("=" * 70)
            print("\nTo resume: python run_llm_benchmark.py --resume --log", args.log)
            print("To clear:  python run_llm_benchmark.py --clear-checkpoint --log", args.log)
        else:
            print(f"\nNo checkpoint found for {args.log}")
        return

    # Handle --clear-checkpoint: delete checkpoint and exit
    if args.clear_checkpoint:
        if checkpoint_manager.delete_checkpoint(log_path):
            print(f"Checkpoint cleared for {args.log}")
        else:
            print(f"No checkpoint found for {args.log}")
        return

    # Get models list
    models = args.models if args.models else LLMBenchmarkRunner.DEFAULT_MODELS

    # Create config
    config = LLMBenchmarkConfig(
        models=models,
        event_log=log_path,
        repetitions=args.repetitions,
        max_iterations=args.max_iterations,
        use_sudo=not args.no_sudo,
        des_runs=args.des_runs,
        abs_runs=args.abs_runs
    )

    # Check for existing checkpoint
    if not args.resume and checkpoint_manager.has_checkpoint(log_path):
        info = checkpoint_manager.get_checkpoint_info(log_path)
        if info:
            print("\n" + "=" * 70)
            print("WARNING: EXISTING CHECKPOINT FOUND")
            print("=" * 70)
            print(f"  Progress: {info['completed_runs']}/{info['total_runs']} runs ({info['progress_pct']:.1f}%)")
            print(f"  Saved at: {info['saved_at']}")
            print("=" * 70)
            print("\nOptions:")
            print("  --resume           Resume from checkpoint")
            print("  --clear-checkpoint Clear checkpoint and start fresh")
            print("\nStarting fresh will DISCARD existing progress!")
            sys.exit(1)

    # Print configuration
    print("\n" + "=" * 70)
    print("LLM BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"  Event log:      {args.log}")
    print(f"  Log path:       {log_path}")
    print(f"  Models:         {len(config.models)}")
    print(f"  Repetitions:    {config.repetitions}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Use sudo:       {config.use_sudo}")
    print(f"  DES runs:       {config.des_runs}")
    print(f"  ABS runs:       {config.abs_runs}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Checkpointing:  {'DISABLED' if args.no_checkpoint else 'ENABLED'}")
    print(f"  Resume mode:    {'YES' if args.resume else 'NO'}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. Would benchmark these models:")
        for i, model in enumerate(config.models, 1):
            print(f"  {i}. {model}")
        return

    # Run benchmark
    print("\nStarting benchmark...")
    runner = LLMBenchmarkRunner(
        config,
        checkpoint_manager=None if args.no_checkpoint else checkpoint_manager,
        enable_checkpointing=not args.no_checkpoint
    )
    report = runner.run_benchmark(resume=args.resume)

    # Generate reports
    os.makedirs(args.output_dir, exist_ok=True)
    reporter = LLMBenchmarkReporter(args.output_dir)
    eval_dir = reporter.generate_full_report(report)

    # Print final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print("\nSpeed Ranking (fastest to slowest):")
    for i, model in enumerate(report.speed_ranking[:5], 1):
        result = next((r for r in report.results if r.model_name == model), None)
        if result:
            print(f"  {i}. {model}: {result.time_to_improvement_mean:.1f}s")

    print("\nImpact Ranking (highest to lowest score):")
    for i, model in enumerate(report.impact_ranking[:5], 1):
        result = next((r for r in report.results if r.model_name == model), None)
        if result:
            print(f"  {i}. {model}: {result.averaged_score_mean:.4f}")

    print(f"\nFull results: {eval_dir}")


if __name__ == "__main__":
    main()
