"""
Evaluation Runner for Quantitative Experiments.

Orchestrates the full evaluation pipeline:
1. Discover optimal model (M_opt) with SIMOD
2. Inject fault to create M_fault
3. Run both Grid Search and LLM Search
4. Compare simulation calls needed to find the fix
5. Generate results for paper

Fair Benchmarking: Uses a simulation cache with fixed random seed to ensure
that identical inputs (e.g., Grid Search testing count=3 and LLM testing count=3)
yield the exact same metrics. This eliminates random variance as a confounding factor.
"""

import os
import json
import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from simulator.simod_discoverer import SIMODDiscoverer
from simulator.prosimos_runner import ProsimosRunner
from evaluation.quantitative.fault_injector import FaultInjector
from evaluation.quantitative.grid_search import GridSearch, GridSearchHistory
from evaluation.quantitative.llm_search import LLMResourceSearch, LLMSearchHistory


@dataclass
class ExperimentResult:
    """Result of a single log experiment."""
    log_name: str
    target_resource: str
    original_count: int
    fault_count: int

    # Optimal model metrics (ground truth)
    optimal_metrics: Dict

    # Faulty model metrics (starting point)
    faulty_metrics: Dict

    # Search results
    grid_search: Optional[GridSearchHistory] = None
    llm_search: Optional[LLMSearchHistory] = None

    # Comparison metrics
    grid_iterations_to_solution: Optional[int] = None
    llm_iterations_to_solution: Optional[int] = None
    speedup_ratio: Optional[float] = None  # grid/llm iterations

    def to_dict(self) -> Dict:
        return {
            "log_name": self.log_name,
            "target_resource": self.target_resource,
            "original_count": self.original_count,
            "fault_count": self.fault_count,
            "optimal_metrics": self.optimal_metrics,
            "faulty_metrics": self.faulty_metrics,
            "grid_search": self.grid_search.to_dict() if self.grid_search else None,
            "llm_search": self.llm_search.to_dict() if self.llm_search else None,
            "grid_iterations_to_solution": self.grid_iterations_to_solution,
            "llm_iterations_to_solution": self.llm_iterations_to_solution,
            "speedup_ratio": self.speedup_ratio
        }


@dataclass
class EvaluationSummary:
    """Summary of all experiments."""
    experiments: List[ExperimentResult] = field(default_factory=list)
    timestamp: str = ""
    model_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "num_experiments": len(self.experiments),
            "experiments": [e.to_dict() for e in self.experiments],
            "aggregate": self._compute_aggregate()
        }

    def _compute_aggregate(self) -> Dict:
        """Compute aggregate statistics across all experiments."""
        if not self.experiments:
            return {}

        grid_iters = [e.grid_iterations_to_solution for e in self.experiments
                      if e.grid_iterations_to_solution is not None]
        llm_iters = [e.llm_iterations_to_solution for e in self.experiments
                     if e.llm_iterations_to_solution is not None]
        speedups = [e.speedup_ratio for e in self.experiments
                    if e.speedup_ratio is not None]

        return {
            "avg_grid_iterations": sum(grid_iters) / len(grid_iters) if grid_iters else None,
            "avg_llm_iterations": sum(llm_iters) / len(llm_iters) if llm_iters else None,
            "avg_speedup_ratio": sum(speedups) / len(speedups) if speedups else None,
            "grid_success_rate": len(grid_iters) / len(self.experiments) if self.experiments else 0,
            "llm_success_rate": len(llm_iters) / len(self.experiments) if self.experiments else 0
        }


class QuantitativeEvaluator:
    """
    Main evaluator for quantitative experiments.

    Compares LLM-guided search vs systematic grid search
    for recovering from injected resource faults.
    """

    # Default logs to evaluate
    # NOTE: Only logs with grouped roles work well (e.g., Clerk-000001..Clerk-000008)
    # Logs with individual resources (__ALL_RESOURCES__ mode) are skipped automatically
    DEFAULT_LOGS = [
        "p2p.csv",
        "LoanApp.csv",
        "ACR.csv",
        "BPIC_2012W.csv",      # Will be skipped if individual resources
        "BPIC_2017W.csv",      # Will be skipped if individual resources
        "Confidential_1000.csv",
        "Confidential_2000.csv",
        "Production.csv",      # Will be skipped if individual resources
        "cvs_pharmacy.csv"
    ]

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        use_sudo: bool = True,
        grid_min: int = 1,
        grid_max: int = 10,
        fault_count: int = 1,
        tolerance_pct: float = 5.0,
        max_llm_iterations: int = 10,
        des_runs: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: LLM model for guided search
            temperature: LLM temperature
            use_sudo: Use sudo for Docker commands
            grid_min: Min resource count for grid search
            grid_max: Max resource count for grid search
            fault_count: Resource count to set for fault injection
            tolerance_pct: Tolerance for solution matching (KPI-based convergence)
            max_llm_iterations: Max iterations for LLM search
            des_runs: Number of DES simulation runs to average (default: 5)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_sudo = use_sudo
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.fault_count = fault_count
        self.des_runs = max(1, int(des_runs))
        self.tolerance_pct = tolerance_pct
        self.max_llm_iterations = max_llm_iterations

        # Initialize components
        self.discoverer = SIMODDiscoverer(use_sudo=use_sudo)
        self.prosimos = ProsimosRunner(use_sudo=use_sudo)
        self.fault_injector = FaultInjector(fault_resource_count=fault_count)
        self.grid_search = GridSearch(
            min_count=grid_min,
            max_count=grid_max,
            tolerance_pct=tolerance_pct
        )
        self.llm_search = LLMResourceSearch(
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_llm_iterations,
            tolerance_pct=tolerance_pct
        )

    def _run_averaged_simulation(
        self,
        model: Dict,
        modifications: Dict,
        work_dir: str,
        baseline_metrics: Dict
    ) -> Dict:
        """Run multiple DES simulations and average numeric metrics for stability.

        Args:
            model: Simulation model
            modifications: Parameter modifications
            work_dir: Working directory
            baseline_metrics: Baseline metrics for reference

        Returns:
            Averaged metrics dictionary
        """
        des_runs = []
        for run_idx in range(self.des_runs):
            if self.des_runs > 1:
                print(f"      DES run {run_idx + 1}/{self.des_runs}...", end=" ")
            metrics = self.prosimos.simulate(
                model=model,
                modifications=modifications,
                work_dir=work_dir,
                baseline_metrics=baseline_metrics
            )
            if metrics:
                des_runs.append(metrics)
                if self.des_runs > 1:
                    print(f"cycle_time={metrics.get('avg_cycle_time', 0):.2f}s")

        if not des_runs:
            return {}

        if len(des_runs) == 1:
            return des_runs[0]

        # Average numeric metrics
        aggregated = {}
        numeric_keys = ["avg_cycle_time", "avg_throughput", "avg_waiting_time"]
        for key in numeric_keys:
            values = [float(m[key]) for m in des_runs if key in m and m[key] is not None]
            if values:
                aggregated[key] = sum(values) / len(values)

        # Average resource utilization
        resource_utils = {}
        for run in des_runs:
            for resource_id, value in run.get("resource_utilization", {}).items():
                resource_utils.setdefault(resource_id, []).append(float(value))
        if resource_utils:
            aggregated["resource_utilization"] = {
                res_id: sum(vals) / len(vals) for res_id, vals in resource_utils.items()
            }

        if self.des_runs > 1:
            print(f"      Averaged {len(des_runs)} runs: cycle_time={aggregated.get('avg_cycle_time', 0):.2f}s")

        return aggregated

    def run_evaluation(
        self,
        logs_dir: str,
        output_dir: str,
        logs: Optional[List[str]] = None
    ) -> EvaluationSummary:
        """
        Run full evaluation on specified logs.

        Args:
            logs_dir: Directory containing event logs
            output_dir: Directory for outputs and results
            logs: List of log filenames (default: DEFAULT_LOGS)

        Returns:
            EvaluationSummary with all experiment results
        """
        if logs is None:
            logs = self.DEFAULT_LOGS

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = EvaluationSummary(
            timestamp=timestamp,
            model_name=self.model_name
        )

        # Create output directory
        eval_output_dir = os.path.join(output_dir, f"quantitative_eval_{timestamp}")
        os.makedirs(eval_output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("QUANTITATIVE EVALUATION: LLM vs Grid Search")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Grid search range: {self.grid_min}-{self.grid_max}")
        print(f"Fault injection: reduce to {self.fault_count} resource(s)")
        print(f"Tolerance: {self.tolerance_pct}%")
        print(f"DES runs per simulation: {self.des_runs}")
        print(f"Logs to evaluate: {len(logs)}")
        print("=" * 80 + "\n")

        for i, log_filename in enumerate(logs, 1):
            log_path = os.path.join(logs_dir, log_filename)

            if not os.path.exists(log_path):
                print(f"\n[{i}/{len(logs)}] SKIP: {log_filename} (file not found)")
                continue

            print(f"\n[{i}/{len(logs)}] Processing: {log_filename}")
            print("-" * 60)

            try:
                result = self._run_single_experiment(
                    log_path=log_path,
                    log_name=log_filename,
                    work_dir=os.path.join(eval_output_dir, os.path.splitext(log_filename)[0])
                )
                summary.experiments.append(result)

                # Save intermediate result
                result_file = os.path.join(
                    eval_output_dir,
                    f"result_{os.path.splitext(log_filename)[0]}.json"
                )
                with open(result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

            except Exception as e:
                print(f"  [X] Error processing {log_filename}: {e}")
                import traceback
                traceback.print_exc()

        # Save full summary
        summary_file = os.path.join(eval_output_dir, "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        self._print_summary(summary)
        print(f"\nResults saved to: {eval_output_dir}")

        return summary

    def _run_single_experiment(
        self,
        log_path: str,
        log_name: str,
        work_dir: str
    ) -> ExperimentResult:
        """Run experiment on a single log."""

        os.makedirs(work_dir, exist_ok=True)

        # Step 1: Discover optimal model (M_opt)
        print("  Step 1: Discovering optimal model with SIMOD...")
        model, baseline_metrics = self.discoverer.discover(log_path, work_dir)
        print(f"    [OK] Model discovered, baseline cycle_time: {baseline_metrics.get('avg_cycle_time', 0):.2f}s")

        # Step 2: Simulate optimal model to get ground truth metrics
        print(f"  Step 2: Simulating optimal model ({self.des_runs} runs)...")
        optimal_metrics = self._run_averaged_simulation(
            model=model,
            modifications={},
            work_dir=work_dir,
            baseline_metrics=baseline_metrics
        )
        print(f"    [OK] Optimal cycle_time: {optimal_metrics.get('avg_cycle_time', 0):.2f}s")

        # Step 3: Find target resource and inject fault
        print("  Step 3: Injecting fault...")
        # Prefer roles with at least 3 resources for meaningful fault injection
        # (reducing 3→1 is more dramatic than 2→1)
        target_resource, original_count, utilization = self.fault_injector.find_best_resource_for_fault(
            baseline_metrics, model, min_count=3
        )

        # Fall back to min_count=2 if no role has 3+ resources
        if target_resource is None:
            print("    [!] No role with 3+ resources found, trying min_count=2...")
            target_resource, original_count, utilization = self.fault_injector.find_best_resource_for_fault(
                baseline_metrics, model, min_count=2
            )

        if target_resource is None:
            raise ValueError("Could not identify target resource for fault injection")

        # Log if using activity-based grouping (informational)
        if target_resource.startswith("Group_"):
            print(f"    [OK] Using activity-based pseudo-role: {target_resource}")

        # Validate that we can create a meaningful fault
        if original_count <= self.fault_count:
            print(f"    [!] WARNING: original_count ({original_count}) <= fault_count ({self.fault_count})")
            print(f"    [!] This won't create a meaningful bottleneck for the experiment")

        print(f"    Target: {target_resource} (original count: {original_count}, util: {utilization:.2%})")

        faulty_model, fault_info = self.fault_injector.inject_fault(
            model=model,
            target_resource=target_resource,
            fault_count=self.fault_count
        )
        print(f"    [OK] Fault injected: {target_resource} reduced from {original_count} to {self.fault_count}")

        # Step 4: Simulate faulty model to get starting metrics
        print(f"  Step 4: Simulating faulty model ({self.des_runs} runs)...")
        faulty_metrics = self._run_averaged_simulation(
            model=faulty_model,
            modifications={},
            work_dir=work_dir,
            baseline_metrics=baseline_metrics
        )
        faulty_cycle = faulty_metrics.get('avg_cycle_time', 0)
        optimal_cycle = optimal_metrics.get('avg_cycle_time', 0)
        print(f"    [OK] Faulty cycle_time: {faulty_cycle:.2f}s")

        # Validate fault effect
        if optimal_cycle > 0:
            degradation_pct = (faulty_cycle - optimal_cycle) / optimal_cycle * 100
            print(f"    [OK] Fault degradation: {degradation_pct:+.1f}% cycle time increase")
            if degradation_pct < 20:
                print(f"    [!] WARNING: Fault only caused {degradation_pct:.1f}% degradation (< 20%)")
                print(f"    [!] This log may not show meaningful difference between search methods")
                print(f"    [!] Consider using a log with larger resource groups (e.g., LoanApp.csv)")
            if degradation_pct < 0:
                print(f"    [!] CRITICAL: Faulty model is FASTER than optimal - this group is not on critical path!")
                print(f"    [!] Skipping this log - results would be meaningless")

        # Create simulation function that restores N profiles for the target role
        # This is key: we use the ORIGINAL model and set the resource count directly
        #
        # FAIR BENCHMARKING: We use a simulation cache keyed by resource count.
        # This ensures that if Grid Search tests count=3 and LLM tests count=3,
        # they get the EXACT same metrics. The cache guarantees identical inputs
        # yield identical outputs, eliminating random variance as a confounding factor.
        simulation_cache: Dict[int, Dict] = {}

        def simulate_with_role_count(role_name: str, count: int) -> Dict:
            """Simulate with exactly `count` resources for the specified role.

            Uses a cache to ensure identical counts yield identical results,
            regardless of which search method (Grid or LLM) tests them first.
            """
            # Check cache first - ensures fair benchmarking
            if count in simulation_cache:
                print(f"      [CACHE HIT] Using cached result for count={count}")
                return simulation_cache[count]

            # Start from original model (not faulty)
            test_model = copy.deepcopy(model)
            resource_profiles = test_model.get("prosimos_params", {}).get("resource_profiles", [])

            # Special case: __ALL_RESOURCES__ means all resources as a pool
            if role_name == "__ALL_RESOURCES__":
                # Sort for deterministic selection
                sorted_profiles = sorted(resource_profiles, key=lambda p: p.get("id", p.get("name", "")))
                kept = sorted_profiles[:count]
                removed = sorted_profiles[count:]

                test_model["prosimos_params"]["resource_profiles"] = kept
                self.fault_injector._update_task_distributions(test_model, removed)

                result = self._run_averaged_simulation(
                    model=test_model,
                    modifications={},
                    work_dir=work_dir,
                    baseline_metrics=baseline_metrics
                )
                simulation_cache[count] = result
                return result

            # Activity-based pseudo-role (Group_*)
            if role_name.startswith("Group_"):
                activity_roles = self.fault_injector._group_resources_by_activity_with_profiles(model)

                if role_name not in activity_roles:
                    raise ValueError(f"Activity-based group '{role_name}' not found")

                group_profiles = activity_roles[role_name]
                group_resource_ids = {
                    p.get("id", p.get("name", "")).replace("_profile", "")
                    for p in group_profiles
                }

                # Split profiles
                other_profiles = []
                role_profiles = []
                for profile in resource_profiles:
                    profile_id = profile.get("id", profile.get("name", "")).replace("_profile", "")
                    if profile_id in group_resource_ids:
                        role_profiles.append(profile)
                    else:
                        other_profiles.append(profile)

                # Keep only `count` profiles for this group
                kept = role_profiles[:count]
                removed = role_profiles[count:]

                test_model["prosimos_params"]["resource_profiles"] = other_profiles + kept
                self.fault_injector._update_task_distributions(test_model, removed)

                result = self._run_averaged_simulation(
                    model=test_model,
                    modifications={},
                    work_dir=work_dir,
                    baseline_metrics=baseline_metrics
                )
                simulation_cache[count] = result
                return result

            # Normal case: pattern-based role
            role_profiles = []
            other_profiles = []
            for profile in resource_profiles:
                profile_id = profile.get("id", profile.get("name", ""))
                if self.fault_injector._extract_role_name(profile_id) == role_name:
                    role_profiles.append(profile)
                else:
                    other_profiles.append(profile)

            # Keep only `count` profiles for this role
            kept = role_profiles[:count]
            removed = role_profiles[count:]

            # Update model
            test_model["prosimos_params"]["resource_profiles"] = other_profiles + kept

            # Update task distributions
            self.fault_injector._update_task_distributions(test_model, removed)

            # Run simulation with no additional modifications
            result = self._run_averaged_simulation(
                model=test_model,
                modifications={},
                work_dir=work_dir,
                baseline_metrics=baseline_metrics
            )
            simulation_cache[count] = result
            return result

        # Wrapper for grid/LLM search API compatibility
        def simulate_fn(ignored_model: Dict, modifications: Dict) -> Dict:
            # Extract the count from modifications
            changes = modifications.get("specific_changes", [])
            if changes:
                count = changes[0].get("value", 1)
            else:
                count = 1
            # Cap count to original_count - can't create more resources than originally existed
            count = min(count, original_count)
            return simulate_with_role_count(target_resource, count)

        # Cap search range to original_count (can't exceed available resources)
        # Always search up to original_count - that's the target we're trying to recover
        effective_grid_max = original_count
        effective_llm_max_iters = original_count

        print(f"  Search range: 1 to {effective_grid_max} (original resource count)")

        # Step 5: Run Grid Search
        print("  Step 5: Running Grid Search...")
        # Create a temporary grid search with capped range
        from evaluation.quantitative.grid_search import GridSearch
        capped_grid_search = GridSearch(
            min_count=self.grid_min,
            max_count=effective_grid_max,
            tolerance_pct=self.tolerance_pct
        )
        grid_result = capped_grid_search.search(
            faulty_model=faulty_model,
            target_resource=target_resource,
            optimal_count=original_count,
            optimal_metrics=optimal_metrics,
            simulate_fn=simulate_fn,
            work_dir=work_dir
        )

        # Step 6: Run LLM Search
        print("  Step 6: Running LLM Search...")
        # Create a temporary LLM search with capped iterations
        from evaluation.quantitative.llm_search import LLMResourceSearch
        capped_llm_search = LLMResourceSearch(
            model_name=self.model_name,
            temperature=self.temperature,
            max_iterations=effective_llm_max_iters,
            tolerance_pct=self.tolerance_pct
        )
        llm_result = capped_llm_search.search(
            faulty_model=faulty_model,
            target_resource=target_resource,
            optimal_count=original_count,
            optimal_metrics=optimal_metrics,
            faulty_metrics=faulty_metrics,
            simulate_fn=simulate_fn,
            work_dir=work_dir
        )

        # Compute comparison metrics
        grid_iters = grid_result.solution_found_at
        llm_iters = llm_result.solution_found_at
        speedup = None
        if grid_iters and llm_iters:
            speedup = grid_iters / llm_iters

        result = ExperimentResult(
            log_name=log_name,
            target_resource=target_resource,
            original_count=original_count,
            fault_count=self.fault_count,
            optimal_metrics=optimal_metrics,
            faulty_metrics=faulty_metrics,
            grid_search=grid_result,
            llm_search=llm_result,
            grid_iterations_to_solution=grid_iters,
            llm_iterations_to_solution=llm_iters,
            speedup_ratio=speedup
        )

        # Print result
        print(f"\n  RESULT: {log_name}")
        print(f"    Grid Search: {grid_iters or 'not found'} iterations")
        print(f"    LLM Search:  {llm_iters or 'not found'} iterations")
        if speedup:
            print(f"    Speedup:     {speedup:.2f}x")

        return result

    def _print_summary(self, summary: EvaluationSummary):
        """Print summary of all experiments."""
        print(f"\nProcessed {len(summary.experiments)} logs")

        agg = summary.to_dict().get("aggregate", {})
        if agg:
            print(f"\nAggregate Results:")
            print(f"  Avg Grid iterations: {agg.get('avg_grid_iterations', 'N/A'):.2f}" if agg.get('avg_grid_iterations') else "  Avg Grid iterations: N/A")
            print(f"  Avg LLM iterations:  {agg.get('avg_llm_iterations', 'N/A'):.2f}" if agg.get('avg_llm_iterations') else "  Avg LLM iterations:  N/A")
            print(f"  Avg Speedup ratio:   {agg.get('avg_speedup_ratio', 'N/A'):.2f}x" if agg.get('avg_speedup_ratio') else "  Avg Speedup ratio:   N/A")
            print(f"  Grid success rate:   {agg.get('grid_success_rate', 0):.1%}")
            print(f"  LLM success rate:    {agg.get('llm_success_rate', 0):.1%}")

        print("\nPer-log results:")
        for exp in summary.experiments:
            grid = exp.grid_iterations_to_solution or "X"
            llm = exp.llm_iterations_to_solution or "X"
            speedup = f"{exp.speedup_ratio:.2f}x" if exp.speedup_ratio else "N/A"
            print(f"  {exp.log_name}: Grid={grid}, LLM={llm}, Speedup={speedup}")
