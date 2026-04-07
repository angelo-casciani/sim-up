"""
Checkpoint manager for LLM Benchmark.

Provides save/resume functionality to handle interrupted benchmark runs.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

from evaluation.quantitative.llm_benchmark_runner import (
    LLMBenchmarkConfig,
    ModelBenchmarkResult,
    SingleModelRun,
    ProposalResult,
)


class BenchmarkCheckpoint:
    """
    Manages checkpoint state for benchmark runs.

    Saves progress after each model/run completion so benchmarks can be
    resumed after interruption.
    """

    DEFAULT_CHECKPOINT_DIR = ".benchmark_checkpoints"

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints. Defaults to .benchmark_checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_checkpoint_path(self, event_log: str) -> str:
        """Get checkpoint file path for a given event log."""
        # Use log name as identifier
        log_name = os.path.splitext(os.path.basename(event_log))[0]
        return os.path.join(self.checkpoint_dir, f"checkpoint_{log_name}.json")

    def save_checkpoint(
        self,
        config: LLMBenchmarkConfig,
        completed_results: List[ModelBenchmarkResult],
        current_model_idx: int,
        current_run_idx: int,
        current_model_runs: List[SingleModelRun],
        timestamp: str
    ) -> str:
        """
        Save current benchmark state to checkpoint file.

        Args:
            config: Benchmark configuration
            completed_results: Fully completed model results
            current_model_idx: Index of model currently being processed
            current_run_idx: Index of run within current model (next run to execute)
            current_model_runs: Completed runs for current model
            timestamp: Benchmark timestamp

        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            "version": "1.0",
            "timestamp": timestamp,
            "saved_at": datetime.now().isoformat(),
            "config": {
                "models": config.models,
                "event_log": config.event_log,
                "repetitions": config.repetitions,
                "max_iterations": config.max_iterations,
                "use_sudo": config.use_sudo,
                "des_runs": config.des_runs,
                "abs_runs": config.abs_runs,
            },
            "progress": {
                "current_model_idx": current_model_idx,
                "current_run_idx": current_run_idx,
            },
            "completed_results": [r.to_dict() for r in completed_results],
            "current_model_runs": [r.to_dict() for r in current_model_runs],
        }

        checkpoint_path = self.get_checkpoint_path(config.event_log)

        # Write atomically (write to temp, then rename)
        temp_path = checkpoint_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Atomic rename
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        os.rename(temp_path, checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, event_log: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data if it exists.

        Args:
            event_log: Event log path to identify the checkpoint

        Returns:
            Checkpoint data dict or None if no checkpoint exists
        """
        checkpoint_path = self.get_checkpoint_path(event_log)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None

    def delete_checkpoint(self, event_log: str) -> bool:
        """
        Delete checkpoint file after successful completion.

        Args:
            event_log: Event log path to identify the checkpoint

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.get_checkpoint_path(event_log)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            return True
        return False

    def has_checkpoint(self, event_log: str) -> bool:
        """Check if a checkpoint exists for the given event log."""
        return os.path.exists(self.get_checkpoint_path(event_log))

    def get_checkpoint_info(self, event_log: str) -> Optional[Dict[str, Any]]:
        """
        Get summary info about an existing checkpoint.

        Returns dict with progress info or None if no checkpoint.
        """
        data = self.load_checkpoint(event_log)
        if not data:
            return None

        config = data.get("config", {})
        progress = data.get("progress", {})
        completed = data.get("completed_results", [])
        current_runs = data.get("current_model_runs", [])

        total_models = len(config.get("models", []))
        total_runs_per_model = config.get("repetitions", 1)

        completed_models = len(completed)
        current_model_idx = progress.get("current_model_idx", 0)
        current_run_idx = progress.get("current_run_idx", 0)

        # Calculate total progress
        total_runs = total_models * total_runs_per_model
        completed_runs = (completed_models * total_runs_per_model) + len(current_runs)

        current_model = config.get("models", [])[current_model_idx] if current_model_idx < total_models else "N/A"

        return {
            "saved_at": data.get("saved_at"),
            "timestamp": data.get("timestamp"),
            "total_models": total_models,
            "completed_models": completed_models,
            "current_model": current_model,
            "current_model_idx": current_model_idx,
            "current_run_idx": current_run_idx,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "progress_pct": (completed_runs / total_runs * 100) if total_runs > 0 else 0,
        }


def reconstruct_model_result(data: Dict[str, Any]) -> ModelBenchmarkResult:
    """Reconstruct ModelBenchmarkResult from checkpoint data."""
    runs = [reconstruct_single_run(r) for r in data.get("runs", [])]

    return ModelBenchmarkResult(
        model_name=data["model_name"],
        num_runs=data.get("num_runs", 0),
        num_successful=data.get("num_successful", 0),
        num_errors=data.get("num_errors", 0),
        num_no_improvement=data.get("num_no_improvement", 0),
        time_to_improvement_mean=data.get("time_to_improvement_mean"),
        time_to_improvement_std=data.get("time_to_improvement_std"),
        iterations_to_improvement_mean=data.get("iterations_to_improvement_mean"),
        iterations_to_improvement_std=data.get("iterations_to_improvement_std"),
        proposals_to_improvement_mean=data.get("proposals_to_improvement_mean"),
        proposals_to_improvement_std=data.get("proposals_to_improvement_std"),
        averaged_score_mean=data.get("averaged_score_mean"),
        averaged_score_std=data.get("averaged_score_std"),
        success_rate=data.get("success_rate", 0.0),
        runs=runs,
    )


def reconstruct_single_run(data: Dict[str, Any]) -> SingleModelRun:
    """Reconstruct SingleModelRun from checkpoint data."""
    iterations = [reconstruct_proposal(p) for p in data.get("iterations", [])]
    all_proposals = [reconstruct_proposal(p) for p in data.get("all_proposals", [])]

    return SingleModelRun(
        run_id=data["run_id"],
        model_name=data["model_name"],
        status=data["status"],
        time_to_first_improvement=data.get("time_to_first_improvement"),
        iterations_to_first_improvement=data.get("iterations_to_first_improvement"),
        proposals_to_first_improvement=data.get("proposals_to_first_improvement"),
        total_time=data.get("total_time", 0.0),
        total_iterations=data.get("total_iterations", 0),
        total_proposals=data.get("total_proposals", 0),
        averaged_score=data.get("averaged_score"),
        des_score=data.get("des_score"),
        abs_score=data.get("abs_score"),
        first_improvement_hypothesis=data.get("first_improvement_hypothesis"),
        error_message=data.get("error_message"),
        iterations=iterations,
        all_proposals=all_proposals,
        baseline_kpis=data.get("baseline_kpis"),
    )


def reconstruct_proposal(data: Dict[str, Any]) -> ProposalResult:
    """Reconstruct ProposalResult from checkpoint data."""
    kpi = data.get("kpi_impact", {})
    des = kpi.get("des", {})
    abs_kpi = kpi.get("abs", {})

    return ProposalResult(
        proposal_num=data.get("proposal_num", data.get("iteration_num", 0)),
        hypothesis_description=data.get("hypothesis_description", ""),
        hypothesis_type=data.get("hypothesis_type", "unknown"),
        is_improvement=data.get("is_improvement", False),
        rejected_before_simulation=data.get("rejected_before_simulation", False),
        des_cycle_time_change=des.get("cycle_time_change_pct"),
        des_throughput_change=des.get("throughput_change_pct"),
        des_waiting_time_change=des.get("waiting_time_change_pct"),
        abs_cycle_time_change=abs_kpi.get("cycle_time_change_pct"),
        abs_throughput_change=abs_kpi.get("throughput_change_pct"),
        abs_waiting_time_change=abs_kpi.get("waiting_time_change_pct"),
        des_score=des.get("score"),
        abs_score=abs_kpi.get("score"),
        averaged_score=kpi.get("averaged_score"),
        distance_to_threshold=kpi.get("distance_to_threshold"),
        proposal_time=data.get("proposal_time", data.get("iteration_time", 0.0)),
        cumulative_time=data.get("cumulative_time", 0.0),
    )
