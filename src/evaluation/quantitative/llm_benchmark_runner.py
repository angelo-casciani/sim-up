"""
LLM Benchmark Runner

Benchmarks multiple LLM models on the improvement validation pipeline.
Measures speed (time and iterations to first validated improvement) and
impact (averaged validation formula score).

Supports checkpoint/resume functionality for long-running benchmarks.
"""

import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from improvement_validation_pipeline import ImprovementValidationAgent

if TYPE_CHECKING:
    from evaluation.quantitative.benchmark_checkpoint import BenchmarkCheckpoint


@dataclass
class LLMBenchmarkConfig:
    """Configuration for a benchmark run."""
    models: List[str]
    event_log: str
    repetitions: int = 1
    max_iterations: int = 5
    use_sudo: bool = True
    des_runs: int = 5
    abs_runs: int = 5


@dataclass
class ProposalResult:
    """Result of a single proposal within a run (includes rejected proposals)."""
    proposal_num: int
    hypothesis_description: str
    hypothesis_type: str  # e.g., "resource_allocation", "calendar_adjustment", "shift_responsibility"
    is_improvement: bool
    rejected_before_simulation: bool = False  # True if rejected by strategic screening

    # KPI impacts vs baseline (percentage change) - None if rejected before simulation
    des_cycle_time_change: Optional[float] = None
    des_throughput_change: Optional[float] = None
    des_waiting_time_change: Optional[float] = None
    abs_cycle_time_change: Optional[float] = None
    abs_throughput_change: Optional[float] = None
    abs_waiting_time_change: Optional[float] = None

    # Validation scores - 0.0 if rejected before simulation
    des_score: Optional[float] = None
    abs_score: Optional[float] = None
    averaged_score: Optional[float] = None

    # Distance to 1% improvement threshold (0 when achieved, positive when below)
    # Used for convergence visualization
    distance_to_threshold: Optional[float] = None

    # Timing
    proposal_time: float = 0.0
    cumulative_time: float = 0.0  # Time from start including baseline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_num": self.proposal_num,
            "iteration_num": self.proposal_num,  # Backward compatibility
            "hypothesis_description": self.hypothesis_description,
            "hypothesis_type": self.hypothesis_type,
            "is_improvement": self.is_improvement,
            "rejected_before_simulation": self.rejected_before_simulation,
            "kpi_impact": {
                "des": {
                    "cycle_time_change_pct": self.des_cycle_time_change,
                    "throughput_change_pct": self.des_throughput_change,
                    "waiting_time_change_pct": self.des_waiting_time_change,
                    "score": self.des_score
                },
                "abs": {
                    "cycle_time_change_pct": self.abs_cycle_time_change,
                    "throughput_change_pct": self.abs_throughput_change,
                    "waiting_time_change_pct": self.abs_waiting_time_change,
                    "score": self.abs_score
                },
                "averaged_score": self.averaged_score,
                "distance_to_threshold": self.distance_to_threshold
            },
            "proposal_time": self.proposal_time,
            "iteration_time": self.proposal_time,  # Backward compatibility
            "cumulative_time": self.cumulative_time
        }


# Backward compatibility alias
IterationResult = ProposalResult


@dataclass
class SingleModelRun:
    """Result of a single model run."""
    run_id: int
    model_name: str
    status: str  # "success", "error", "no_improvement"

    # Speed metrics
    time_to_first_improvement: Optional[float] = None  # seconds
    iterations_to_first_improvement: Optional[int] = None  # Legacy: simulated iterations only
    proposals_to_first_improvement: Optional[int] = None  # New: includes rejected proposals
    total_time: float = 0.0
    total_iterations: int = 0  # Legacy: simulated iterations only
    total_proposals: int = 0   # New: includes rejected proposals

    # Impact metrics (from first successful improvement)
    averaged_score: Optional[float] = None  # (des_score + abs_score) / 2
    des_score: Optional[float] = None
    abs_score: Optional[float] = None

    # Details
    first_improvement_hypothesis: Optional[str] = None
    error_message: Optional[str] = None

    # Detailed iteration history (simulated only, for backward compatibility)
    iterations: List[ProposalResult] = field(default_factory=list)

    # All proposals including rejected ones (for visualizations)
    all_proposals: List[ProposalResult] = field(default_factory=list)

    # Baseline KPIs for reference
    baseline_kpis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "status": self.status,
            "time_to_first_improvement": self.time_to_first_improvement,
            "iterations_to_first_improvement": self.iterations_to_first_improvement,
            "proposals_to_first_improvement": self.proposals_to_first_improvement,
            "total_time": self.total_time,
            "total_iterations": self.total_iterations,
            "total_proposals": self.total_proposals,
            "averaged_score": self.averaged_score,
            "des_score": self.des_score,
            "abs_score": self.abs_score,
            "first_improvement_hypothesis": self.first_improvement_hypothesis,
            "error_message": self.error_message,
            "baseline_kpis": self.baseline_kpis,
            "iterations": [it.to_dict() for it in self.iterations],
            "all_proposals": [p.to_dict() for p in self.all_proposals]
        }


@dataclass
class ModelBenchmarkResult:
    """Aggregated results for a single model across repetitions."""
    model_name: str
    num_runs: int = 0
    num_successful: int = 0
    num_errors: int = 0
    num_no_improvement: int = 0

    # Speed statistics (mean/std) - only from successful runs
    time_to_improvement_mean: Optional[float] = None
    time_to_improvement_std: Optional[float] = None
    iterations_to_improvement_mean: Optional[float] = None  # Legacy: simulated iterations only
    iterations_to_improvement_std: Optional[float] = None
    proposals_to_improvement_mean: Optional[float] = None   # New: includes rejected proposals
    proposals_to_improvement_std: Optional[float] = None

    # Impact statistics
    averaged_score_mean: Optional[float] = None
    averaged_score_std: Optional[float] = None

    # Success rate
    success_rate: float = 0.0

    # Individual runs
    runs: List[SingleModelRun] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_runs": self.num_runs,
            "num_successful": self.num_successful,
            "num_errors": self.num_errors,
            "num_no_improvement": self.num_no_improvement,
            "success_rate": self.success_rate,
            "time_to_improvement_mean": self.time_to_improvement_mean,
            "time_to_improvement_std": self.time_to_improvement_std,
            "iterations_to_improvement_mean": self.iterations_to_improvement_mean,
            "iterations_to_improvement_std": self.iterations_to_improvement_std,
            "proposals_to_improvement_mean": self.proposals_to_improvement_mean,
            "proposals_to_improvement_std": self.proposals_to_improvement_std,
            "averaged_score_mean": self.averaged_score_mean,
            "averaged_score_std": self.averaged_score_std,
            "runs": [r.to_dict() for r in self.runs]
        }


@dataclass
class LLMBenchmarkReport:
    """Complete benchmark report across all models."""
    timestamp: str
    event_log: str
    num_repetitions: int
    max_iterations: int
    max_proposals: int = 0  # Max proposals across all runs (for visualization x-axis)

    # Per-model results
    results: List[ModelBenchmarkResult] = field(default_factory=list)

    # Rankings (model names sorted by metric)
    speed_ranking: List[str] = field(default_factory=list)
    iteration_ranking: List[str] = field(default_factory=list)
    proposals_ranking: List[str] = field(default_factory=list)  # New: by proposals
    impact_ranking: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_log": self.event_log,
            "num_repetitions": self.num_repetitions,
            "max_iterations": self.max_iterations,
            "max_proposals": self.max_proposals,
            "results": [r.to_dict() for r in self.results],
            "rankings": {
                "speed": self.speed_ranking,
                "iterations": self.iteration_ranking,
                "proposals": self.proposals_ranking,
                "impact": self.impact_ranking
            }
        }


class LLMBenchmarkRunner:
    """
    Benchmarks multiple LLM models on the improvement validation pipeline.

    Measures:
    1. Time to first validated improvement (seconds)
    2. Iterations to first validated improvement (count)
    3. Averaged validation formula score (impact quality)

    Supports checkpoint/resume for long-running benchmarks.
    """

    DEFAULT_MODELS = [
        # OpenAI
        "gpt-4.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o3",
        # Anthropic
        "anthropic--claude-4.5-opus",
        "anthropic--claude-4.5-sonnet",
        "anthropic--claude-4.5-haiku",
        # Mistral
        "mistralai--mistral-small-instruct",
        "mistralai--mistral-large-instruct"
    ]

    def __init__(
        self,
        config: LLMBenchmarkConfig,
        checkpoint_manager: Optional["BenchmarkCheckpoint"] = None,
        enable_checkpointing: bool = True
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            checkpoint_manager: Optional checkpoint manager for save/resume
            enable_checkpointing: Whether to save checkpoints (default True)
        """
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.enable_checkpointing = enable_checkpointing
        self._timestamp: Optional[str] = None

    def run_benchmark(self, resume: bool = False) -> LLMBenchmarkReport:
        """
        Run full benchmark across all models.

        Args:
            resume: If True, attempt to resume from checkpoint

        Returns:
            LLMBenchmarkReport with all results
        """
        # Try to resume from checkpoint
        start_model_idx = 0
        start_run_idx = 0
        results: List[ModelBenchmarkResult] = []
        current_model_runs: List[SingleModelRun] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if resume and self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.config.event_log)
            if checkpoint_data:
                # Validate config matches
                saved_config = checkpoint_data.get("config", {})
                if self._config_matches(saved_config):
                    print("\n" + "=" * 70)
                    print("RESUMING FROM CHECKPOINT")
                    print("=" * 70)

                    # Restore state
                    timestamp = checkpoint_data.get("timestamp", timestamp)
                    progress = checkpoint_data.get("progress", {})
                    start_model_idx = progress.get("current_model_idx", 0)
                    start_run_idx = progress.get("current_run_idx", 0)

                    # Reconstruct completed results
                    from evaluation.quantitative.benchmark_checkpoint import (
                        reconstruct_model_result,
                        reconstruct_single_run
                    )
                    results = [
                        reconstruct_model_result(r)
                        for r in checkpoint_data.get("completed_results", [])
                    ]
                    current_model_runs = [
                        reconstruct_single_run(r)
                        for r in checkpoint_data.get("current_model_runs", [])
                    ]

                    info = self.checkpoint_manager.get_checkpoint_info(self.config.event_log)
                    if info:
                        print(f"  Saved at: {info['saved_at']}")
                        print(f"  Progress: {info['completed_runs']}/{info['total_runs']} runs ({info['progress_pct']:.1f}%)")
                        print(f"  Resuming from: {info['current_model']} (run {start_run_idx + 1})")
                    print("=" * 70)
                else:
                    print("\nWarning: Checkpoint config doesn't match current config. Starting fresh.")
                    if self.checkpoint_manager:
                        self.checkpoint_manager.delete_checkpoint(self.config.event_log)

        self._timestamp = timestamp

        print(f"\nLLM Benchmark - {timestamp}")
        print(f"Models to benchmark: {len(self.config.models)}")
        print(f"Repetitions per model: {self.config.repetitions}")
        print(f"Max iterations per run: {self.config.max_iterations}")

        if self.enable_checkpointing and self.checkpoint_manager:
            print(f"Checkpointing: ENABLED (can resume with --resume)")
        else:
            print(f"Checkpointing: DISABLED")

        for model_idx, model_name in enumerate(self.config.models):
            # Skip already completed models
            if model_idx < start_model_idx:
                continue

            print(f"\n{'='*70}")
            print(f"[{model_idx + 1}/{len(self.config.models)}] Benchmarking: {model_name}")
            print(f"{'='*70}")

            # Determine starting run for this model
            run_start = start_run_idx if model_idx == start_model_idx else 0
            runs_for_model = current_model_runs if model_idx == start_model_idx else []

            # Clear for next model
            if model_idx > start_model_idx:
                current_model_runs = []

            model_result = self._benchmark_model_with_checkpoint(
                model_name,
                model_idx,
                run_start,
                runs_for_model,
                results
            )
            results.append(model_result)

            # Reset for next model
            start_run_idx = 0
            current_model_runs = []

            # Print summary for this model
            self._print_model_summary(model_result)

        # Create report with rankings
        report = self._create_report(timestamp, results)

        # Delete checkpoint on successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.delete_checkpoint(self.config.event_log)
            print("\nCheckpoint cleared after successful completion.")

        return report

    def _config_matches(self, saved_config: Dict[str, Any]) -> bool:
        """Check if saved config matches current config."""
        return (
            saved_config.get("models") == self.config.models and
            saved_config.get("repetitions") == self.config.repetitions and
            saved_config.get("max_iterations") == self.config.max_iterations and
            saved_config.get("des_runs") == self.config.des_runs and
            saved_config.get("abs_runs") == self.config.abs_runs
        )

    def _save_checkpoint(
        self,
        completed_results: List[ModelBenchmarkResult],
        current_model_idx: int,
        current_run_idx: int,
        current_model_runs: List[SingleModelRun]
    ):
        """Save checkpoint if checkpointing is enabled."""
        if self.enable_checkpointing and self.checkpoint_manager and self._timestamp:
            self.checkpoint_manager.save_checkpoint(
                config=self.config,
                completed_results=completed_results,
                current_model_idx=current_model_idx,
                current_run_idx=current_run_idx,
                current_model_runs=current_model_runs,
                timestamp=self._timestamp
            )

    def _benchmark_model_with_checkpoint(
        self,
        model_name: str,
        model_idx: int,
        start_run_idx: int,
        existing_runs: List[SingleModelRun],
        completed_results: List[ModelBenchmarkResult]
    ) -> ModelBenchmarkResult:
        """Run benchmark for a single model with checkpointing."""
        runs = list(existing_runs)  # Copy existing runs

        for run_id in range(start_run_idx + 1, self.config.repetitions + 1):
            print(f"\n  Run {run_id}/{self.config.repetitions}...")
            run_result = self._run_single_benchmark(model_name, run_id)
            runs.append(run_result)

            # Print run result
            status_emoji = {
                "success": "[OK]",
                "timeout": "[TIMEOUT]",
                "error": "[ERROR]",
                "no_improvement": "[NO IMPROVEMENT]"
            }.get(run_result.status, "[?]")

            if run_result.status == "success":
                print(f"    {status_emoji} Time: {run_result.time_to_first_improvement:.1f}s, "
                      f"Iterations: {run_result.iterations_to_first_improvement}, "
                      f"Score: {run_result.averaged_score:.4f}")
            else:
                msg = run_result.error_message or run_result.status
                print(f"    {status_emoji} {msg}")

            # Save checkpoint after each run
            self._save_checkpoint(
                completed_results=completed_results,
                current_model_idx=model_idx,
                current_run_idx=run_id,  # run_id is 1-indexed, represents completed
                current_model_runs=runs
            )

        return self._aggregate_model_results(model_name, runs)

    def _run_single_benchmark(self, model_name: str, run_id: int) -> SingleModelRun:
        """Execute a single benchmark run. Stops based on max_iterations only."""
        result_holder = {
            "agent": None,
            "status": "error",
            "error": None,
            "total_time": 0.0
        }

        try:
            agent = ImprovementValidationAgent(
                model_name=model_name,
                temperature=0.0,
                max_tokens=4096,
                max_iterations=self.config.max_iterations,
                use_sudo=self.config.use_sudo,
                des_runs=self.config.des_runs,
                abs_runs=self.config.abs_runs
            )

            start_time = time.time()
            agent.run_improvement_pipeline(self.config.event_log)
            result_holder["total_time"] = time.time() - start_time
            result_holder["agent"] = agent
            result_holder["status"] = "completed"

        except Exception as e:
            result_holder["status"] = "error"
            result_holder["error"] = str(e)

        # Extract metrics from completed run
        return self._extract_metrics_from_result(
            result_holder, run_id, model_name
        )

    def _extract_metrics_from_result(
        self,
        result_holder: Dict,
        run_id: int,
        model_name: str
    ) -> SingleModelRun:
        """Extract benchmark metrics from completed agent run."""

        if result_holder["status"] == "error":
            return SingleModelRun(
                run_id=run_id,
                model_name=model_name,
                status="error",
                total_time=result_holder.get("total_time", 0),
                error_message=result_holder["error"]
            )

        agent = result_holder["agent"]
        if agent is None:
            return SingleModelRun(
                run_id=run_id,
                model_name=model_name,
                status="error",
                total_time=0,
                error_message="Agent was not initialized"
            )

        # Extract baseline KPIs
        baseline_kpis = None
        for entry in agent.history:
            if entry.get("step") == "baseline":
                baseline_kpis = entry.get("metrics", {})
                break

        # Process all iterations and proposals
        baseline_time = agent.execution_times.get('step_1_baseline_discovery', 0)
        cumulative_time = baseline_time

        # Improvement threshold (1%)
        IMPROVEMENT_THRESHOLD = 0.01

        iterations_list = []  # Only simulated proposals (backward compatible)
        all_proposals_list = []  # All proposals including rejected ones
        proposal_counter = 0  # Track total proposals
        first_improvement = None
        first_improvement_iteration = None
        first_improvement_proposal = None
        first_improvement_time = None
        hypothesis_description = None
        first_improvement_entry = None

        for entry in agent.history:
            step = entry.get("step", "")
            if step.startswith("iteration_"):
                iteration_num = int(step.split("_")[1])
                iteration_time = entry.get("execution_time", 0)
                cumulative_time += iteration_time

                # First, add any rejected proposals from strategic screening
                rejected_proposals = entry.get("rejected_proposals", [])
                for rejected in rejected_proposals:
                    proposal_counter += 1
                    rejected_hyp = rejected.get("hypothesis", {})
                    rejected_result = ProposalResult(
                        proposal_num=proposal_counter,
                        hypothesis_description=rejected_hyp.get("description", ""),
                        hypothesis_type=rejected_hyp.get("type", "unknown"),
                        is_improvement=False,
                        rejected_before_simulation=True,
                        averaged_score=0.0,  # Rejected = 0.0
                        distance_to_threshold=IMPROVEMENT_THRESHOLD,  # Max distance
                        proposal_time=0.0,  # Minimal time for rejection
                        cumulative_time=cumulative_time
                    )
                    all_proposals_list.append(rejected_result)

                # Then add the simulated proposal
                proposal_counter += 1
                hypothesis = entry.get("hypothesis", {})
                improvement = entry.get("improvement", {})
                des_impact = entry.get("des_impact_vs_baseline", {})
                abs_impact = entry.get("abs_impact_vs_baseline", {})
                des_validation = entry.get("des_validation", {})
                abs_validation = entry.get("abs_validation", {})

                # Get averaged score and compute distance to threshold
                avg_score = improvement.get("averaged_score")
                distance_to_threshold = None
                if avg_score is not None:
                    # Distance is 0 when threshold achieved, positive when below
                    distance_to_threshold = max(0, IMPROVEMENT_THRESHOLD - avg_score)

                # Create proposal result
                proposal_result = ProposalResult(
                    proposal_num=proposal_counter,
                    hypothesis_description=hypothesis.get("description", ""),
                    hypothesis_type=hypothesis.get("type", "unknown"),
                    is_improvement=improvement.get("is_improvement", False),
                    rejected_before_simulation=False,
                    # DES KPI changes
                    des_cycle_time_change=des_impact.get("cycle_time"),
                    des_throughput_change=des_impact.get("throughput"),
                    des_waiting_time_change=des_impact.get("waiting_time"),
                    # ABS KPI changes
                    abs_cycle_time_change=abs_impact.get("cycle_time"),
                    abs_throughput_change=abs_impact.get("throughput"),
                    abs_waiting_time_change=abs_impact.get("waiting_time"),
                    # Scores
                    des_score=des_validation.get("score") if des_validation else None,
                    abs_score=abs_validation.get("score") if abs_validation else None,
                    averaged_score=avg_score,
                    distance_to_threshold=distance_to_threshold,
                    proposal_time=iteration_time,
                    cumulative_time=cumulative_time
                )
                iterations_list.append(proposal_result)  # Backward compatible
                all_proposals_list.append(proposal_result)

                # Track first improvement
                if improvement.get("is_improvement") and first_improvement is None:
                    first_improvement = improvement
                    first_improvement_iteration = iteration_num
                    first_improvement_proposal = proposal_counter
                    first_improvement_time = cumulative_time
                    hypothesis_description = hypothesis.get("description")
                    first_improvement_entry = entry

        total_iterations = len(iterations_list)
        total_proposals = len(all_proposals_list)

        if first_improvement is None:
            return SingleModelRun(
                run_id=run_id,
                model_name=model_name,
                status="no_improvement",
                total_time=agent.execution_times.get('total_pipeline', result_holder.get("total_time", 0)),
                total_iterations=total_iterations,
                total_proposals=total_proposals,
                iterations=iterations_list,
                all_proposals=all_proposals_list,
                baseline_kpis=baseline_kpis
            )

        # Extract scores from first improvement
        averaged_score = first_improvement.get('averaged_score')
        des_validation = first_improvement_entry.get('des_validation', {}) if first_improvement_entry else {}
        abs_validation = first_improvement_entry.get('abs_validation', {}) if first_improvement_entry else {}
        des_score = des_validation.get('score') if des_validation else None
        abs_score = abs_validation.get('score') if abs_validation else None

        return SingleModelRun(
            run_id=run_id,
            model_name=model_name,
            status="success",
            time_to_first_improvement=first_improvement_time,
            iterations_to_first_improvement=first_improvement_iteration,
            proposals_to_first_improvement=first_improvement_proposal,
            total_time=agent.execution_times.get('total_pipeline', result_holder.get("total_time", 0)),
            total_iterations=total_iterations,
            total_proposals=total_proposals,
            averaged_score=averaged_score,
            des_score=des_score,
            abs_score=abs_score,
            first_improvement_hypothesis=hypothesis_description,
            iterations=iterations_list,
            all_proposals=all_proposals_list,
            baseline_kpis=baseline_kpis
        )

    def _aggregate_model_results(
        self,
        model_name: str,
        runs: List[SingleModelRun]
    ) -> ModelBenchmarkResult:
        """Aggregate results from multiple runs into statistics."""
        result = ModelBenchmarkResult(model_name=model_name)
        result.runs = runs
        result.num_runs = len(runs)

        # Count statuses
        for run in runs:
            if run.status == "success":
                result.num_successful += 1
            elif run.status == "error":
                result.num_errors += 1
            elif run.status == "no_improvement":
                result.num_no_improvement += 1

        # Success rate
        result.success_rate = result.num_successful / result.num_runs if result.num_runs > 0 else 0.0

        # Compute statistics from successful runs only
        successful_runs = [r for r in runs if r.status == "success"]

        if successful_runs:
            # Time statistics
            times = [r.time_to_first_improvement for r in successful_runs if r.time_to_first_improvement is not None]
            if times:
                result.time_to_improvement_mean = statistics.mean(times)
                result.time_to_improvement_std = statistics.stdev(times) if len(times) > 1 else 0.0

            # Iteration statistics (legacy: simulated only)
            iterations = [r.iterations_to_first_improvement for r in successful_runs if r.iterations_to_first_improvement is not None]
            if iterations:
                result.iterations_to_improvement_mean = statistics.mean(iterations)
                result.iterations_to_improvement_std = statistics.stdev(iterations) if len(iterations) > 1 else 0.0

            # Proposals statistics (new: includes rejected)
            proposals = [r.proposals_to_first_improvement for r in successful_runs if r.proposals_to_first_improvement is not None]
            if proposals:
                result.proposals_to_improvement_mean = statistics.mean(proposals)
                result.proposals_to_improvement_std = statistics.stdev(proposals) if len(proposals) > 1 else 0.0

            # Score statistics
            scores = [r.averaged_score for r in successful_runs if r.averaged_score is not None]
            if scores:
                result.averaged_score_mean = statistics.mean(scores)
                result.averaged_score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return result

    def _create_report(
        self,
        timestamp: str,
        results: List[ModelBenchmarkResult]
    ) -> LLMBenchmarkReport:
        """Create final report with rankings."""
        # Calculate max proposals across all runs
        max_proposals = 0
        for model_result in results:
            for run in model_result.runs:
                if run.total_proposals > max_proposals:
                    max_proposals = run.total_proposals

        report = LLMBenchmarkReport(
            timestamp=timestamp,
            event_log=self.config.event_log,
            num_repetitions=self.config.repetitions,
            max_iterations=self.config.max_iterations,
            max_proposals=max_proposals,
            results=results
        )

        # Compute rankings (only include models with successful runs)
        successful_results = [r for r in results if r.num_successful > 0]

        # Speed ranking (ascending - lower time is better)
        speed_sorted = sorted(
            [r for r in successful_results if r.time_to_improvement_mean is not None],
            key=lambda r: r.time_to_improvement_mean
        )
        report.speed_ranking = [r.model_name for r in speed_sorted]

        # Iteration ranking (ascending - fewer iterations is better) - legacy
        iter_sorted = sorted(
            [r for r in successful_results if r.iterations_to_improvement_mean is not None],
            key=lambda r: r.iterations_to_improvement_mean
        )
        report.iteration_ranking = [r.model_name for r in iter_sorted]

        # Proposals ranking (ascending - fewer proposals is better)
        proposals_sorted = sorted(
            [r for r in successful_results if r.proposals_to_improvement_mean is not None],
            key=lambda r: r.proposals_to_improvement_mean
        )
        report.proposals_ranking = [r.model_name for r in proposals_sorted]

        # Impact ranking (descending - higher score is better)
        impact_sorted = sorted(
            [r for r in successful_results if r.averaged_score_mean is not None],
            key=lambda r: r.averaged_score_mean,
            reverse=True
        )
        report.impact_ranking = [r.model_name for r in impact_sorted]

        return report

    def _print_model_summary(self, result: ModelBenchmarkResult):
        """Print summary for a completed model benchmark."""
        print(f"\n  Summary for {result.model_name}:")
        print(f"    Success rate: {result.success_rate*100:.0f}% ({result.num_successful}/{result.num_runs})")

        if result.num_successful > 0:
            if result.time_to_improvement_mean is not None:
                std_str = f" +/- {result.time_to_improvement_std:.1f}" if result.time_to_improvement_std else ""
                print(f"    Avg time to improvement: {result.time_to_improvement_mean:.1f}s{std_str}")

            if result.iterations_to_improvement_mean is not None:
                std_str = f" +/- {result.iterations_to_improvement_std:.2f}" if result.iterations_to_improvement_std else ""
                print(f"    Avg iterations: {result.iterations_to_improvement_mean:.2f}{std_str}")

            if result.averaged_score_mean is not None:
                std_str = f" +/- {result.averaged_score_std:.4f}" if result.averaged_score_std else ""
                print(f"    Avg impact score: {result.averaged_score_mean:.4f}{std_str}")

        if result.num_errors > 0:
            print(f"    Errors: {result.num_errors}")
        if result.num_no_improvement > 0:
            print(f"    No improvement found: {result.num_no_improvement}")
