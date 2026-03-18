"""
LLM Benchmark Runner

Benchmarks multiple LLM models on the improvement validation pipeline.
Measures speed (time and iterations to first validated improvement) and
impact (averaged validation formula score).
"""

import os
import sys
import time
import threading
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from improvement_validation_pipeline import ImprovementValidationAgent


@dataclass
class LLMBenchmarkConfig:
    """Configuration for a benchmark run."""
    models: List[str]
    event_log: str
    repetitions: int = 1
    timeout_seconds: int = 900  # 15 minutes default
    max_iterations: int = 5
    use_sudo: bool = True
    des_runs: int = 5
    abs_runs: int = 5


@dataclass
class SingleModelRun:
    """Result of a single model run."""
    run_id: int
    model_name: str
    status: str  # "success", "timeout", "error", "no_improvement"

    # Speed metrics
    time_to_first_improvement: Optional[float] = None  # seconds
    iterations_to_first_improvement: Optional[int] = None
    total_time: float = 0.0

    # Impact metrics
    averaged_score: Optional[float] = None  # (des_score + abs_score) / 2
    des_score: Optional[float] = None
    abs_score: Optional[float] = None

    # Details
    first_improvement_hypothesis: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "status": self.status,
            "time_to_first_improvement": self.time_to_first_improvement,
            "iterations_to_first_improvement": self.iterations_to_first_improvement,
            "total_time": self.total_time,
            "averaged_score": self.averaged_score,
            "des_score": self.des_score,
            "abs_score": self.abs_score,
            "first_improvement_hypothesis": self.first_improvement_hypothesis,
            "error_message": self.error_message
        }


@dataclass
class ModelBenchmarkResult:
    """Aggregated results for a single model across repetitions."""
    model_name: str
    num_runs: int = 0
    num_successful: int = 0
    num_timeouts: int = 0
    num_errors: int = 0
    num_no_improvement: int = 0

    # Speed statistics (mean/std) - only from successful runs
    time_to_improvement_mean: Optional[float] = None
    time_to_improvement_std: Optional[float] = None
    iterations_to_improvement_mean: Optional[float] = None
    iterations_to_improvement_std: Optional[float] = None

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
            "num_timeouts": self.num_timeouts,
            "num_errors": self.num_errors,
            "num_no_improvement": self.num_no_improvement,
            "success_rate": self.success_rate,
            "time_to_improvement_mean": self.time_to_improvement_mean,
            "time_to_improvement_std": self.time_to_improvement_std,
            "iterations_to_improvement_mean": self.iterations_to_improvement_mean,
            "iterations_to_improvement_std": self.iterations_to_improvement_std,
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
    timeout_seconds: int
    max_iterations: int

    # Per-model results
    results: List[ModelBenchmarkResult] = field(default_factory=list)

    # Rankings (model names sorted by metric)
    speed_ranking: List[str] = field(default_factory=list)
    iteration_ranking: List[str] = field(default_factory=list)
    impact_ranking: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_log": self.event_log,
            "num_repetitions": self.num_repetitions,
            "timeout_seconds": self.timeout_seconds,
            "max_iterations": self.max_iterations,
            "results": [r.to_dict() for r in self.results],
            "rankings": {
                "speed": self.speed_ranking,
                "iterations": self.iteration_ranking,
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
        "mistralai--mistral-medium-instruct",
        "mistralai--mistral-large-instruct"
    ]

    def __init__(self, config: LLMBenchmarkConfig):
        self.config = config

    def run_benchmark(self) -> LLMBenchmarkReport:
        """Run full benchmark across all models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []

        print(f"\nStarting LLM Benchmark at {timestamp}")
        print(f"Models to benchmark: {len(self.config.models)}")
        print(f"Repetitions per model: {self.config.repetitions}")
        print(f"Timeout per run: {self.config.timeout_seconds}s ({self.config.timeout_seconds/60:.0f} min)")

        for i, model_name in enumerate(self.config.models, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(self.config.models)}] Benchmarking: {model_name}")
            print(f"{'='*70}")

            model_result = self._benchmark_model(model_name)
            results.append(model_result)

            # Print summary for this model
            self._print_model_summary(model_result)

        # Create report with rankings
        report = self._create_report(timestamp, results)
        return report

    def _benchmark_model(self, model_name: str) -> ModelBenchmarkResult:
        """Run benchmark for a single model with repetitions."""
        runs = []

        for run_id in range(1, self.config.repetitions + 1):
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

        return self._aggregate_model_results(model_name, runs)

    def _run_single_benchmark(self, model_name: str, run_id: int) -> SingleModelRun:
        """Execute a single benchmark run with timeout handling."""
        result_holder = {
            "agent": None,
            "status": "timeout",
            "error": None,
            "total_time": 0.0
        }

        def run_agent():
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

        # Run in thread with timeout
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()
        thread.join(timeout=self.config.timeout_seconds)

        if thread.is_alive():
            # Timeout occurred - thread is still running
            return SingleModelRun(
                run_id=run_id,
                model_name=model_name,
                status="timeout",
                total_time=self.config.timeout_seconds,
                error_message=f"Exceeded {self.config.timeout_seconds}s timeout"
            )

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

        # Find first validated improvement in history
        baseline_time = agent.execution_times.get('step_1_baseline_discovery', 0)
        cumulative_time = baseline_time

        first_improvement = None
        first_improvement_iteration = None
        first_improvement_time = None
        hypothesis_description = None
        first_improvement_entry = None

        for entry in agent.history:
            step = entry.get("step", "")
            if step.startswith("iteration_"):
                iteration_num = int(step.split("_")[1])
                iteration_time = entry.get("execution_time", 0)
                cumulative_time += iteration_time

                improvement = entry.get("improvement", {})
                if improvement.get("is_improvement"):
                    first_improvement = improvement
                    first_improvement_iteration = iteration_num
                    first_improvement_time = cumulative_time
                    hypothesis_description = entry.get("hypothesis", {}).get("description")
                    first_improvement_entry = entry
                    break

        if first_improvement is None:
            return SingleModelRun(
                run_id=run_id,
                model_name=model_name,
                status="no_improvement",
                total_time=agent.execution_times.get('total_pipeline', result_holder.get("total_time", 0))
            )

        # Extract scores
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
            total_time=agent.execution_times.get('total_pipeline', result_holder.get("total_time", 0)),
            averaged_score=averaged_score,
            des_score=des_score,
            abs_score=abs_score,
            first_improvement_hypothesis=hypothesis_description
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
            elif run.status == "timeout":
                result.num_timeouts += 1
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

            # Iteration statistics
            iterations = [r.iterations_to_first_improvement for r in successful_runs if r.iterations_to_first_improvement is not None]
            if iterations:
                result.iterations_to_improvement_mean = statistics.mean(iterations)
                result.iterations_to_improvement_std = statistics.stdev(iterations) if len(iterations) > 1 else 0.0

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
        report = LLMBenchmarkReport(
            timestamp=timestamp,
            event_log=self.config.event_log,
            num_repetitions=self.config.repetitions,
            timeout_seconds=self.config.timeout_seconds,
            max_iterations=self.config.max_iterations,
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

        # Iteration ranking (ascending - fewer iterations is better)
        iter_sorted = sorted(
            [r for r in successful_results if r.iterations_to_improvement_mean is not None],
            key=lambda r: r.iterations_to_improvement_mean
        )
        report.iteration_ranking = [r.model_name for r in iter_sorted]

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

        if result.num_timeouts > 0:
            print(f"    Timeouts: {result.num_timeouts}")
        if result.num_errors > 0:
            print(f"    Errors: {result.num_errors}")
        if result.num_no_improvement > 0:
            print(f"    No improvement found: {result.num_no_improvement}")
