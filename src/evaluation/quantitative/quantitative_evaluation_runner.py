"""
Comprehensive Evaluation Runner for All Improvement Types.

Orchestrates the full evaluation pipeline across all three improvement types:
1. Resource Allocation - Adding resources to bottleneck roles
2. Calendar Adjustment - Extending working hours for constrained resources
3. Shift Responsibility - Redistributing workload between roles

Key Features:
- Supports 5 repetitions per experiment with mean ± std reporting
- Fair benchmarking with simulation cache
- Dual-simulator validation (DES + ABS)
- Intervention cost tracking

Based on the evaluation framework proposal for journal paper submission.
"""

import os
import json
import copy
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from simulator.simod_discoverer import SIMODDiscoverer
from simulator.prosimos_runner import ProsimosRunner
from simulator.agent_simulator_runner import AgentSimulatorRunner

# Resource allocation components
from evaluation.quantitative.resource_fault_injector import FaultInjector
from evaluation.quantitative.resource_grid_search import GridSearch, GridSearchHistory
from evaluation.quantitative.resource_llm_search import LLMResourceSearch, LLMSearchHistory

# Calendar components (new)
from evaluation.quantitative.calendar_fault_injector import CalendarFaultInjector
from evaluation.quantitative.calendar_grid_search import CalendarGridSearch, CalendarGridSearchHistory
from evaluation.quantitative.calendar_llm_search import CalendarLLMSearch, CalendarLLMSearchHistory

# Shift responsibility components
from evaluation.quantitative.shift_responsibility_fault_injector import ShiftResponsibilityFaultInjector
from evaluation.quantitative.shift_responsibility_grid_search import (
    ShiftResponsibilityGridSearch, ShiftResponsibilityGridSearchHistory
)
from evaluation.quantitative.shift_responsibility_llm_search import (
    ShiftResponsibilityLLMSearch, ShiftResponsibilityLLMHistory
)


class ImprovementType(Enum):
    """Types of process improvements to evaluate."""
    RESOURCE_ALLOCATION = "resource_allocation"
    CALENDAR_ADJUSTMENT = "calendar_adjustment"
    SHIFT_RESPONSIBILITY = "shift_responsibility"


class InterventionCost(Enum):
    """Cost levels for intervention types (lower is better)."""
    LOW = 1       # Shift Responsibility - redistribute existing workforce
    MEDIUM = 2    # Calendar Extension - increase working hours (overtime)
    HIGH = 3      # Resource Addition - hire new staff


@dataclass
class SingleRunResult:
    """Result of a single experiment run."""
    run_id: int
    baseline_iterations: Optional[int]
    llm_iterations: Optional[int]
    speedup_ratio: Optional[float]
    baseline_found_solution: bool
    llm_found_solution: bool
    llm_accuracy: bool  # Did LLM find optimal/correct configuration?
    convergence_quality: Optional[float]  # Final KPI improvement %
    # Dual-simulator validation scores
    des_validation_score: Optional[float] = None  # DES (Prosimos) validation
    abs_validation_score: Optional[float] = None  # ABS (AgentSimulator) validation
    dual_validation_passed: bool = False  # Both S_DES > 0 AND S_ABS > 0
    # Trajectory data for proximity plots
    baseline_trajectory: Optional[List[Dict]] = None  # Per-iteration data for grid search
    llm_trajectory: Optional[List[Dict]] = None  # Per-iteration data for LLM search


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    log_name: str
    improvement_type: ImprovementType
    fault_description: str
    ground_truth: str
    intervention_cost: InterventionCost


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs."""
    config: ExperimentConfig
    num_runs: int

    # Iteration statistics
    baseline_iterations_mean: Optional[float] = None
    baseline_iterations_std: Optional[float] = None
    llm_iterations_mean: Optional[float] = None
    llm_iterations_std: Optional[float] = None

    # Speedup statistics
    speedup_ratio_mean: Optional[float] = None
    speedup_ratio_std: Optional[float] = None

    # Success rates
    baseline_success_rate: float = 0.0
    llm_success_rate: float = 0.0
    llm_accuracy_rate: float = 0.0  # % of runs where LLM found optimal

    # Convergence quality
    convergence_quality_mean: Optional[float] = None
    convergence_quality_std: Optional[float] = None

    # Individual run results
    runs: List[SingleRunResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.config.experiment_id,
            "log_name": self.config.log_name,
            "improvement_type": self.config.improvement_type.value,
            "fault_description": self.config.fault_description,
            "ground_truth": self.config.ground_truth,
            "intervention_cost": self.config.intervention_cost.value,
            "num_runs": self.num_runs,
            "baseline_iterations": {
                "mean": self.baseline_iterations_mean,
                "std": self.baseline_iterations_std
            },
            "llm_iterations": {
                "mean": self.llm_iterations_mean,
                "std": self.llm_iterations_std
            },
            "speedup_ratio": {
                "mean": self.speedup_ratio_mean,
                "std": self.speedup_ratio_std
            },
            "success_rates": {
                "baseline": self.baseline_success_rate,
                "llm": self.llm_success_rate,
                "llm_accuracy": self.llm_accuracy_rate
            },
            "convergence_quality": {
                "mean": self.convergence_quality_mean,
                "std": self.convergence_quality_std
            },
            "runs": [
                {
                    "run_id": r.run_id,
                    "baseline_iterations": r.baseline_iterations,
                    "llm_iterations": r.llm_iterations,
                    "speedup_ratio": r.speedup_ratio,
                    "baseline_found": r.baseline_found_solution,
                    "llm_found": r.llm_found_solution,
                    "llm_accuracy": r.llm_accuracy,
                    "baseline_trajectory": r.baseline_trajectory,
                    "llm_trajectory": r.llm_trajectory
                }
                for r in self.runs
            ]
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report across all experiments."""
    timestamp: str
    model_name: str
    num_repetitions: int
    results: List[AggregatedResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "num_repetitions": self.num_repetitions,
            "total_experiments": len(self.results),
            "results_by_type": self._group_by_type(),
            "aggregate_statistics": self._compute_aggregate(),
            "detailed_results": [r.to_dict() for r in self.results]
        }

    def _group_by_type(self) -> Dict:
        """Group results by improvement type."""
        grouped = {}
        for result in self.results:
            type_name = result.config.improvement_type.value
            if type_name not in grouped:
                grouped[type_name] = []
            grouped[type_name].append(result.config.experiment_id)
        return grouped

    def _compute_aggregate(self) -> Dict:
        """Compute aggregate statistics across all experiments."""
        if not self.results:
            return {}

        speedups = [r.speedup_ratio_mean for r in self.results if r.speedup_ratio_mean]
        llm_rates = [r.llm_success_rate for r in self.results]
        baseline_rates = [r.baseline_success_rate for r in self.results]

        return {
            "overall_avg_speedup": statistics.mean(speedups) if speedups else None,
            "overall_llm_success_rate": statistics.mean(llm_rates) if llm_rates else None,
            "overall_baseline_success_rate": statistics.mean(baseline_rates) if baseline_rates else None,
            "by_improvement_type": self._compute_by_type()
        }

    def _compute_by_type(self) -> Dict:
        """Compute statistics per improvement type."""
        by_type = {}
        for imp_type in ImprovementType:
            type_results = [r for r in self.results if r.config.improvement_type == imp_type]
            if not type_results:
                continue

            speedups = [r.speedup_ratio_mean for r in type_results if r.speedup_ratio_mean]
            by_type[imp_type.value] = {
                "num_experiments": len(type_results),
                "avg_speedup": statistics.mean(speedups) if speedups else None,
                "avg_llm_success_rate": statistics.mean([r.llm_success_rate for r in type_results])
            }
        return by_type


class ComprehensiveEvaluator:
    """
    Main evaluator for comprehensive experiments across all improvement types.

    Implements the evaluation framework for journal paper submission.
    """

    # Default experiment configurations for comprehensive BPM research paper evaluation
    # VALIDATED: Only logs with pattern-based roles (2+ resources per role) are used
    # for resource allocation. Calendar and shift responsibility use different criteria.
    #
    # Validated logs for RESOURCE ALLOCATION:
    #   - LoanApp.csv: Clerk(8), Loan Officer(4), AML Investigator(2), Appraiser(2), Senior Officer(2)
    #   - cvs_pharmacy.csv: Technician(3) - only 1 valid role
    #   - Confidential_1000.csv: DEF(3), GHI(3), JKL(3), MNO(3)
    #
    # EXCLUDED from resource allocation (no multi-resource roles):
    #   - BPI_Challenge_2012_A.csv: numeric IDs only
    #   - env_permit.csv: each resource unique (Resource01, Resource02...)
    #   - helpdesk.csv: each resource unique (Value 1, Value 2...)
    #   - p2p.csv: named individuals, each unique
    #
    DEFAULT_EXPERIMENTS = [
        # ==================== RESOURCE ALLOCATION (E1.1 - E1.8) ====================
        # 8 validated experiments across 3 logs with pattern-based roles
        #
        # LoanApp.csv - Financial domain (4 experiments)
        # Roles: Clerk(8), Loan Officer(4), AML Investigator(2), Appraiser(2)
        ExperimentConfig(
            experiment_id="E1.1",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            fault_description="Clerk: 8->1 (high-volume role)",
            ground_truth="count=8",
            intervention_cost=InterventionCost.HIGH
        ),
        ExperimentConfig(
            experiment_id="E1.2",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            fault_description="Loan Officer: 4->1",
            ground_truth="count=4",
            intervention_cost=InterventionCost.HIGH
        ),
        ExperimentConfig(
            experiment_id="E1.3",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            fault_description="AML Investigator: 2->1",
            ground_truth="count=2",
            intervention_cost=InterventionCost.HIGH
        ),
        ExperimentConfig(
            experiment_id="E1.4",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            fault_description="Appraiser: 2->1",
            ground_truth="count=2",
            intervention_cost=InterventionCost.HIGH
        ),
        # LoanApp.csv - Senior Officer (5th role with 2 resources)
        ExperimentConfig(
            experiment_id="E1.5",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.RESOURCE_ALLOCATION,
            fault_description="Senior Officer: 2->1",
            ground_truth="count=2",
            intervention_cost=InterventionCost.HIGH
        ),

        # ==================== CALENDAR ADJUSTMENT (E2.1 - E2.5) ====================
        # 5 experiments using logs WITHOUT resourceId column (SIMOD-compatible)
        # LoanApp.csv - Financial domain (2 experiments)
        ExperimentConfig(
            experiment_id="E2.1",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.CALENDAR_ADJUSTMENT,
            fault_description="Clerk: 8h->2h",
            ground_truth="+6 hours",
            intervention_cost=InterventionCost.MEDIUM
        ),
        ExperimentConfig(
            experiment_id="E2.2",
            log_name="LoanApp.csv",
            improvement_type=ImprovementType.CALENDAR_ADJUSTMENT,
            fault_description="Loan Officer: 8h->2h",
            ground_truth="+6 hours",
            intervention_cost=InterventionCost.MEDIUM
        ),
        # p2p.csv - Procurement domain (1 experiment)
        # Named individuals: Karel de Groot, Magdalena Predutta, etc.
        ExperimentConfig(
            experiment_id="E2.3",
            log_name="p2p.csv",
            improvement_type=ImprovementType.CALENDAR_ADJUSTMENT,
            fault_description="Karel de Groot: 8h->2h",
            ground_truth="+6 hours",
            intervention_cost=InterventionCost.MEDIUM
        ),
        # helpdesk.csv - IT Support domain (1 experiment)
        # Resources: Value 2, Value 5, etc.
        ExperimentConfig(
            experiment_id="E2.4",
            log_name="helpdesk.csv",
            improvement_type=ImprovementType.CALENDAR_ADJUSTMENT,
            fault_description="Value 2: 8h->2h",
            ground_truth="+6 hours",
            intervention_cost=InterventionCost.MEDIUM
        ),
        # env_permit.csv - Government domain (1 experiment)
        # Resources: Resource01, Resource02, etc.
        ExperimentConfig(
            experiment_id="E2.5",
            log_name="env_permit.csv",
            improvement_type=ImprovementType.CALENDAR_ADJUSTMENT,
            fault_description="Resource01: 8h->2h",
            ground_truth="+6 hours",
            intervention_cost=InterventionCost.MEDIUM
        ),

        # ==================== SHIFT RESPONSIBILITY (E3.1 - E3.5) ====================
        # 5 experiments using logs WITH overlapping activities between resources
        # Shift responsibility requires resources that can perform the same activities
        #
        # helpdesk.csv - IT support domain (2 experiments)
        # 22 resources (Value 1-22) with overlapping activities across support tiers
        ExperimentConfig(
            experiment_id="E3.1",
            log_name="helpdesk.csv",
            improvement_type=ImprovementType.SHIFT_RESPONSIBILITY,
            fault_description="Overload Value 1 resource (50% shift)",
            ground_truth="Shift ~50% to other support resources",
            intervention_cost=InterventionCost.LOW
        ),
        ExperimentConfig(
            experiment_id="E3.2",
            log_name="helpdesk.csv",
            improvement_type=ImprovementType.SHIFT_RESPONSIBILITY,
            fault_description="Overload Value 2 resource",
            ground_truth="Shift to Value 3 or other resources",
            intervention_cost=InterventionCost.LOW
        ),
        # p2p.csv - Procurement domain (3 experiments)
        # Named individuals with overlapping activities
        # Group 1: Karel, Magdalena, Francois share quotation/order activities
        ExperimentConfig(
            experiment_id="E3.3",
            log_name="p2p.csv",
            improvement_type=ImprovementType.SHIFT_RESPONSIBILITY,
            fault_description="Overload Karel de Groot",
            ground_truth="Shift to Magdalena Predutta",
            intervention_cost=InterventionCost.LOW
        ),
        ExperimentConfig(
            experiment_id="E3.4",
            log_name="p2p.csv",
            improvement_type=ImprovementType.SHIFT_RESPONSIBILITY,
            fault_description="Overload Francois de Perrier",
            ground_truth="Shift to Karel de Groot",
            intervention_cost=InterventionCost.LOW
        ),
        # Group 2: Pedro, Karalda share invoice activities
        ExperimentConfig(
            experiment_id="E3.5",
            log_name="p2p.csv",
            improvement_type=ImprovementType.SHIFT_RESPONSIBILITY,
            fault_description="Overload Pedro Alvares",
            ground_truth="Shift to Karalda Nimwada",
            intervention_cost=InterventionCost.LOW
        ),
    ]

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        use_sudo: bool = True,
        num_repetitions: int = 5,
        tolerance_pct: float = 2.0,
        resource_tolerance_pct: float = 30.0,  # Higher tolerance for resource (more variance)
        calendar_tolerance_pct: float = 5.0,   # Calendar has moderate variance
        shift_tolerance_pct: float = 5.0,        # Within 5% of optimal (symmetric, above or below)
        des_runs: int = 5,
        abs_runs: int = 5,
        use_dual_validation: bool = True,
        skip_grid_search: bool = False
    ):
        """
        Initialize the comprehensive evaluator.

        Args:
            model_name: LLM model for guided search
            temperature: LLM temperature (0.0 for determinism)
            use_sudo: Use sudo for Docker commands
            num_repetitions: Number of repetitions per experiment (default: 5)
            tolerance_pct: Default KPI tolerance for convergence (default: 2%)
            resource_tolerance_pct: Tolerance for resource allocation (default: 30% due to high variance)
            calendar_tolerance_pct: Tolerance for calendar adjustment (default: 5%)
            shift_tolerance_pct: Tolerance for shift responsibility (default: 5% - symmetric above/below optimal)
            des_runs: Number of DES (Prosimos) runs to average per simulation
            abs_runs: Number of ABS (AgentSimulator) runs per simulation
            use_dual_validation: Whether to validate with both DES and ABS
            skip_grid_search: Skip running grid search, use theoretical iterations instead
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_sudo = use_sudo
        self.num_repetitions = num_repetitions
        self.tolerance_pct = tolerance_pct
        self.resource_tolerance_pct = resource_tolerance_pct
        self.calendar_tolerance_pct = calendar_tolerance_pct
        self.shift_tolerance_pct = shift_tolerance_pct
        self.des_runs = des_runs
        self.abs_runs = abs_runs
        self.use_dual_validation = use_dual_validation
        self.skip_grid_search = skip_grid_search

        # Initialize simulators
        self.discoverer = SIMODDiscoverer(use_sudo=use_sudo)
        self.prosimos = ProsimosRunner(use_sudo=use_sudo)
        self.agent_simulator = AgentSimulatorRunner()

        # Check ABS availability
        if use_dual_validation and not self.agent_simulator.available:
            print("  [!] WARNING: AgentSimulator not available, dual validation disabled")
            self.use_dual_validation = False

        # Initialize fault injectors
        self.resource_fault_injector = FaultInjector(fault_resource_count=1)
        self.calendar_fault_injector = CalendarFaultInjector(fault_hours_reduction=6.0)
        # Use 50% shift to create workload imbalance that requires ~50% correction
        # This places the optimal fix in the middle of the search space (not trivially at 5% or 10%)
        self.shift_fault_injector = ShiftResponsibilityFaultInjector(fault_shift_fraction=0.5)

    def _parse_target_role_from_config(self, config: ExperimentConfig) -> str:
        """
        Parse the target role name from the experiment config's fault_description.

        Expected formats:
          - "Clerk: 8->1 (high-volume role)" -> "Clerk"
          - "Loan Officer: 4->1" -> "Loan Officer"
          - "AML Investigator: 2->1" -> "AML Investigator"
          - "Technician: 8h->2h" -> "Technician"
          - "Overload Clerk role (50% shift)" -> "Clerk"
          - "Overload Karel de Groot" -> "Karel de Groot"

        Args:
            config: The experiment configuration

        Returns:
            The target role name extracted from fault_description
        """
        desc = config.fault_description

        # Handle "Overload X role" or "Overload X" format (shift responsibility)
        if desc.startswith("Overload "):
            role = desc.replace("Overload ", "").strip()
            # Remove trailing "role" and any parenthetical info
            role = role.split(" (")[0].strip()
            if role.endswith(" role"):
                role = role[:-5].strip()
            return role

        # Handle "Role: N->M" or "Role: Nh->Mh" format (resource/calendar)
        if ":" in desc:
            role = desc.split(":")[0].strip()
            return role

        raise ValueError(f"Cannot parse target role from fault_description: {desc}")

    def run_full_evaluation(
        self,
        logs_dir: str,
        output_dir: str,
        experiments: Optional[List[ExperimentConfig]] = None
    ) -> EvaluationReport:
        """
        Run full evaluation across all experiment configurations.

        Args:
            logs_dir: Directory containing event logs
            output_dir: Directory for outputs and results
            experiments: List of experiment configs (default: DEFAULT_EXPERIMENTS)

        Returns:
            EvaluationReport with all aggregated results
        """
        if experiments is None:
            experiments = self.DEFAULT_EXPERIMENTS

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = EvaluationReport(
            timestamp=timestamp,
            model_name=self.model_name,
            num_repetitions=self.num_repetitions
        )

        eval_output_dir = os.path.join(output_dir, f"impr_type_eval_{timestamp}")
        os.makedirs(eval_output_dir, exist_ok=True)

        self._print_header(experiments)

        for i, config in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {config.experiment_id}: {config.improvement_type.value}")
            print(f"{'='*80}")
            print(f"Log: {config.log_name}")
            print(f"Fault: {config.fault_description}")
            print(f"Ground truth: {config.ground_truth}")
            print(f"Intervention cost: Level {config.intervention_cost.value}")

            log_path = os.path.join(logs_dir, config.log_name)
            if not os.path.exists(log_path):
                print(f"  [SKIP] Log not found: {log_path}")
                continue

            exp_dir = os.path.join(eval_output_dir, config.experiment_id)
            os.makedirs(exp_dir, exist_ok=True)

            try:
                result = self._run_experiment_with_repetitions(
                    config=config,
                    log_path=log_path,
                    work_dir=exp_dir
                )
                report.results.append(result)

                # Save intermediate result
                result_file = os.path.join(exp_dir, "result.json")
                with open(result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

                self._print_experiment_summary(result)

            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

        # Save full report
        report_file = os.path.join(eval_output_dir, "evaluation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        self._print_final_summary(report)
        print(f"\nResults saved to: {eval_output_dir}")

        return report

    def _run_experiment_with_repetitions(
        self,
        config: ExperimentConfig,
        log_path: str,
        work_dir: str
    ) -> AggregatedResult:
        """Run a single experiment configuration with multiple repetitions."""

        result = AggregatedResult(config=config, num_runs=self.num_repetitions)

        # Store log_path for ABS simulations
        self._current_log_path = log_path

        # Step 1: Discover model (once, shared across repetitions)
        print(f"\n  Discovering model with SIMOD...")
        model, simod_metrics = self.discoverer.discover(log_path, work_dir)
        print(f"    [OK] SIMOD metrics cycle_time: {simod_metrics.get('avg_cycle_time', 0):.2f}s")

        # Step 1.5: Simulate baseline with appropriate simulator
        # For shift_responsibility: use ABS (AgentSimulator) as primary
        # For resource/calendar: use DES (Prosimos) as primary
        if config.improvement_type == ImprovementType.SHIFT_RESPONSIBILITY:
            print(f"\n  Simulating baseline model with AgentSimulator (ABS)...")
            baseline_metrics = self._simulate_baseline_abs(log_path, work_dir, simod_metrics)
        else:
            print(f"\n  Simulating baseline model with Prosimos (DES)...")
            baseline_metrics = self._simulate_baseline(model, work_dir, simod_metrics, log_path)
        print(f"    [OK] Simulated baseline cycle_time: {baseline_metrics.get('avg_cycle_time', 0):.2f}s")

        # Step 2: Inject fault based on improvement type
        print(f"\n  Injecting fault...")
        faulty_model, fault_info, optimal_metrics = self._inject_fault(
            config, model, baseline_metrics, work_dir, log_path
        )

        # Step 3: Run repetitions
        print(f"\n  Running {self.num_repetitions} repetitions...")
        for run_id in range(1, self.num_repetitions + 1):
            print(f"\n  --- Repetition {run_id}/{self.num_repetitions} ---")
            run_result = self._run_single_repetition(
                config=config,
                model=model,
                faulty_model=faulty_model,
                fault_info=fault_info,
                optimal_metrics=optimal_metrics,
                baseline_metrics=baseline_metrics,
                work_dir=os.path.join(work_dir, f"run_{run_id}"),
                log_path=log_path
            )
            result.runs.append(run_result)

        # Step 4: Aggregate statistics
        self._aggregate_results(result)

        return result

    def _simulate_baseline(
        self,
        model: Dict,
        work_dir: str,
        simod_metrics: Dict,
        log_path: str
    ) -> Dict:
        """
        Simulate the baseline model with Prosimos to get simulator-aligned metrics.

        This mirrors the main pipeline approach where baseline KPIs come from
        simulating the discovered model, not from SIMOD extraction directly.
        """
        baseline_dir = os.path.join(work_dir, "baseline_simulation")
        os.makedirs(baseline_dir, exist_ok=True)

        # No modifications - simulate the original model
        empty_modifications = {"parameter_type": None, "specific_changes": []}

        # Run DES simulation(s) and average
        des_metrics_list = []
        for run_idx in range(self.des_runs):
            try:
                metrics = self.prosimos.simulate(
                    model=model,
                    modifications=empty_modifications,
                    work_dir=baseline_dir,
                    baseline_metrics=simod_metrics  # Used for case budget calculation
                )
                if metrics:
                    des_metrics_list.append(metrics)
            except Exception as e:
                print(f"      [!] Baseline DES run {run_idx + 1} failed: {e}")

        if des_metrics_list:
            baseline_metrics = self._average_metrics(des_metrics_list)
            baseline_metrics["simulation_source"] = "Prosimos (DES)"
            return baseline_metrics

        # Fallback to SIMOD metrics if simulation fails
        print("      [!] Baseline simulation failed, using SIMOD metrics as fallback")
        return simod_metrics

    def _simulate_baseline_abs(
        self,
        log_path: str,
        work_dir: str,
        simod_metrics: Dict
    ) -> Dict:
        """
        Simulate the baseline model with AgentSimulator (ABS) for shift responsibility.

        For shift responsibility evaluation, ABS is the primary simulator because it better
        captures role-based behavior and workload distribution.
        """
        baseline_dir = os.path.join(work_dir, "baseline_simulation_abs")
        os.makedirs(baseline_dir, exist_ok=True)

        # No modifications - simulate the original model
        empty_modifications = {"parameter_type": None, "specific_changes": []}

        if not self.agent_simulator.available:
            print("      [!] AgentSimulator not available, falling back to SIMOD metrics")
            return simod_metrics

        # Run ABS simulation(s) and average
        abs_metrics_list = []
        for run_idx in range(self.abs_runs):
            try:
                abs_result = self.agent_simulator.simulate(
                    log_path=log_path,
                    modifications=empty_modifications,
                    output_dir=os.path.join(baseline_dir, f"abs_run_{run_idx}"),
                    num_simulations=1
                )
                if abs_result and abs_result[0]:
                    abs_metrics_list.append(abs_result[0])
            except Exception as e:
                print(f"      [!] Baseline ABS run {run_idx + 1} failed: {e}")

        if abs_metrics_list:
            baseline_metrics = self._average_metrics(abs_metrics_list)
            baseline_metrics["simulation_source"] = "AgentSimulator (ABS)"
            # Store ABS-specific metrics
            baseline_metrics["abs_cycle_time"] = baseline_metrics.get("avg_cycle_time", 0)
            baseline_metrics["abs_throughput"] = baseline_metrics.get("avg_throughput", 0)
            return baseline_metrics

        # Fallback to SIMOD metrics if simulation fails
        print("      [!] Baseline ABS simulation failed, using SIMOD metrics as fallback")
        return simod_metrics

    def _inject_fault(
        self,
        config: ExperimentConfig,
        model: Dict,
        baseline_metrics: Dict,
        work_dir: str,
        log_path: Optional[str] = None
    ) -> Tuple[Dict, Dict, Dict]:
        """Inject fault based on improvement type using the SPECIFIC target from config."""

        # Parse the target role from the config's fault_description
        target = self._parse_target_role_from_config(config)
        print(f"    Target from config: '{target}'")

        if config.improvement_type == ImprovementType.RESOURCE_ALLOCATION:
            # Get the actual count for this specific role
            original_count = self.resource_fault_injector.get_role_count(model, target)
            if original_count is None:
                raise ValueError(f"Role '{target}' not found in model. Available roles: {[r['role_name'] for r in self.resource_fault_injector.get_all_resources(model)]}")
            if original_count < 2:
                raise ValueError(f"Role '{target}' has only {original_count} resource(s), need at least 2 for fault injection")

            print(f"    Role '{target}' has {original_count} resources, reducing to 1")

            faulty_model, fault_info = self.resource_fault_injector.inject_fault(
                model, target, fault_count=1
            )

            # Simulate optimal model (original, before fault)
            optimal_metrics = self._simulate_model(model, {}, work_dir, baseline_metrics, log_path, experiment_type="resource")
            fault_info["original_count"] = original_count
            fault_info["target_resource"] = target

        elif config.improvement_type == ImprovementType.CALENDAR_ADJUSTMENT:
            # For calendar, we use the target role name to find matching calendars
            # The calendar injector needs the calendar ID, which typically matches the resource profile
            calendar_target = self.calendar_fault_injector.find_calendar_for_role(model, target)
            if calendar_target is None:
                raise ValueError(f"No calendar found for role '{target}'")

            original_hours = self.calendar_fault_injector.get_calendar_hours(model, calendar_target)
            print(f"    Calendar '{calendar_target}' has {original_hours:.1f}h/week, reducing by ~{6*5}h/week")

            faulty_model, fault_info = self.calendar_fault_injector.inject_fault(
                model, calendar_target, hours_reduction=6.0
            )

            optimal_metrics = self._simulate_model(model, {}, work_dir, baseline_metrics, log_path, experiment_type="calendar")

        elif config.improvement_type == ImprovementType.SHIFT_RESPONSIBILITY:
            # For shift responsibility, target is the overloaded role
            # We need to find a suitable target role to shift work to
            high_role = target  # The role we're overloading (from config)

            # Find a role that shares activities with the overloaded role
            low_role, shared_activities = self.shift_fault_injector.find_target_role_for_overload(
                model, high_role
            )
            if low_role is None:
                raise ValueError(f"No suitable target role found for overloading '{high_role}'")

            print(f"    Shifting work from '{low_role}' to '{high_role}' to create bottleneck")

            # Shift work TO the high_role to overload it
            faulty_model, fault_info = self.shift_fault_injector.inject_fault(
                model, source_role=low_role, target_role=high_role, shift_fraction=0.8
            )

            optimal_metrics = self._simulate_model(model, {}, work_dir, baseline_metrics, log_path, experiment_type="shift_responsibility")

        else:
            raise ValueError(f"Unknown improvement type: {config.improvement_type}")

        # Derive experiment type for faulty model simulation
        exp_type_map = {
            ImprovementType.RESOURCE_ALLOCATION: "resource",
            ImprovementType.CALENDAR_ADJUSTMENT: "calendar",
            ImprovementType.SHIFT_RESPONSIBILITY: "shift_responsibility",
        }
        exp_type = exp_type_map.get(config.improvement_type)

        # Simulate faulty model
        faulty_metrics = self._simulate_model(faulty_model, {}, work_dir, baseline_metrics, log_path, experiment_type=exp_type)
        fault_info["faulty_metrics"] = faulty_metrics
        fault_info["optimal_metrics"] = optimal_metrics

        # Validate that fault actually degraded performance
        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)
        degradation_pct = ((faulty_cycle - optimal_cycle) / optimal_cycle * 100) if optimal_cycle > 0 else 0

        print(f"    [Fault Validation] Optimal cycle_time: {optimal_cycle:.2f}s")
        print(f"    [Fault Validation] Faulty cycle_time: {faulty_cycle:.2f}s")
        print(f"    [Fault Validation] Degradation: {degradation_pct:+.1f}%")

        if faulty_cycle <= optimal_cycle:
            print(f"    [WARNING] Fault did not degrade performance! Faulty ({faulty_cycle:.2f}s) <= Optimal ({optimal_cycle:.2f}s)")

        return faulty_model, fault_info, optimal_metrics

    def _run_single_repetition(
        self,
        config: ExperimentConfig,
        model: Dict,
        faulty_model: Dict,
        fault_info: Dict,
        optimal_metrics: Dict,
        baseline_metrics: Dict,
        work_dir: str,
        log_path: Optional[str] = None
    ) -> SingleRunResult:
        """Run a single repetition of the experiment."""

        os.makedirs(work_dir, exist_ok=True)

        # Derive experiment type from config
        exp_type_map = {
            ImprovementType.RESOURCE_ALLOCATION: "resource",
            ImprovementType.CALENDAR_ADJUSTMENT: "calendar",
            ImprovementType.SHIFT_RESPONSIBILITY: "shift_responsibility",
        }
        exp_type = exp_type_map.get(config.improvement_type)

        # Create simulation cache for fair benchmarking
        simulation_cache: Dict[str, Dict] = {}

        def cached_simulate(model_to_sim: Dict, modifications: Dict) -> Dict:
            """Simulate with caching for fair benchmarking."""
            cache_key = json.dumps(modifications, sort_keys=True)
            if cache_key in simulation_cache:
                return simulation_cache[cache_key]

            result = self._simulate_model(model_to_sim, modifications, work_dir, baseline_metrics, log_path, experiment_type=exp_type)
            simulation_cache[cache_key] = result
            return result

        # Run baseline and LLM search based on improvement type
        if config.improvement_type == ImprovementType.RESOURCE_ALLOCATION:
            baseline_result, llm_result = self._run_resource_searches(
                faulty_model, fault_info, optimal_metrics,
                fault_info.get("faulty_metrics", {}), cached_simulate, work_dir
            )
        elif config.improvement_type == ImprovementType.CALENDAR_ADJUSTMENT:
            baseline_result, llm_result = self._run_calendar_searches(
                faulty_model, fault_info, optimal_metrics,
                fault_info.get("faulty_metrics", {}), cached_simulate, work_dir
            )
        elif config.improvement_type == ImprovementType.SHIFT_RESPONSIBILITY:
            baseline_result, llm_result = self._run_shift_responsibility_searches(
                faulty_model, fault_info, optimal_metrics,
                fault_info.get("faulty_metrics", {}), cached_simulate, work_dir,
                original_model=model
            )
        else:
            raise ValueError(f"Unknown improvement type: {config.improvement_type}")

        # Extract results
        baseline_iters = getattr(baseline_result, 'solution_found_at', None)
        llm_iters = getattr(llm_result, 'solution_found_at', None)

        speedup = baseline_iters / llm_iters if baseline_iters and llm_iters else None

        # Extract trajectory data for proximity plots
        baseline_trajectory = None
        llm_trajectory = None
        if hasattr(baseline_result, 'to_dict'):
            baseline_dict = baseline_result.to_dict()
            baseline_trajectory = baseline_dict.get('iterations', [])
        if hasattr(llm_result, 'to_dict'):
            llm_dict = llm_result.to_dict()
            llm_trajectory = llm_dict.get('iterations', [])

        return SingleRunResult(
            run_id=len(simulation_cache),  # Approximate run ID
            baseline_iterations=baseline_iters,
            llm_iterations=llm_iters,
            speedup_ratio=speedup,
            baseline_found_solution=baseline_iters is not None,
            llm_found_solution=llm_iters is not None,
            llm_accuracy=llm_iters is not None,  # Simplified accuracy check
            convergence_quality=None,  # Would need to compute from final metrics
            baseline_trajectory=baseline_trajectory,
            llm_trajectory=llm_trajectory
        )

    def _run_resource_searches(
        self, faulty_model, fault_info, optimal_metrics, faulty_metrics, simulate_fn, work_dir
    ):
        """Run resource allocation searches."""
        target = fault_info.get("target_resource")
        original_count = fault_info.get("original_count", 10)

        # Grid search: theoretical iterations = original_count - 1 (from 1 to original_count)
        if self.skip_grid_search:
            theoretical_iterations = original_count - 1
            grid_result = GridSearchHistory(
                target_resource=target,
                target_count=original_count,
                max_search_range=original_count,
                solution_found_at=theoretical_iterations
            )
            print(f"    [SKIP] Grid search - theoretical iterations: {theoretical_iterations}")
        else:
            grid_search = GridSearch(min_count=1, max_count=original_count, tolerance_pct=self.resource_tolerance_pct)
            grid_result = grid_search.search(
                faulty_model, target, original_count, optimal_metrics, simulate_fn, work_dir
            )

        llm_search = LLMResourceSearch(
            model_name=self.model_name, temperature=self.temperature,
            max_iterations=original_count, tolerance_pct=self.resource_tolerance_pct
        )
        llm_result = llm_search.search(
            faulty_model, target, original_count, optimal_metrics, faulty_metrics, simulate_fn, work_dir
        )

        return grid_result, llm_result

    def _run_calendar_searches(
        self, faulty_model, fault_info, optimal_metrics, faulty_metrics, simulate_fn, work_dir
    ):
        """Run calendar adjustment searches."""
        target = fault_info.get("target_calendar")
        hours_to_add = fault_info.get("expected_fix_hours", 4.0)
        # Support both old and new fault_info keys
        faulty_hours = fault_info.get("fault_hours_daily", fault_info.get("fault_hours", 4.0))

        # Grid search: theoretical iterations = hours_to_add (1-hour steps)
        if self.skip_grid_search:
            theoretical_iterations = int(hours_to_add)
            grid_result = CalendarGridSearchHistory(
                target_calendar=target,
                target_hours=hours_to_add,
                max_search_range=hours_to_add + 1,
                solution_found_at=theoretical_iterations
            )
            print(f"    [SKIP] Grid search - theoretical iterations: {theoretical_iterations}")
        else:
            grid_search = CalendarGridSearch(
                min_hours=1.0, max_hours=hours_to_add + 1, tolerance_pct=self.calendar_tolerance_pct
            )
            grid_result = grid_search.search(
                faulty_model, target, hours_to_add, optimal_metrics, faulty_hours, simulate_fn, work_dir
            )

        llm_search = CalendarLLMSearch(
            model_name=self.model_name, temperature=self.temperature,
            max_iterations=10, tolerance_pct=self.calendar_tolerance_pct
        )
        llm_result = llm_search.search(
            faulty_model, target, hours_to_add, optimal_metrics, faulty_metrics,
            faulty_hours, simulate_fn, work_dir
        )

        return grid_result, llm_result

    def _run_shift_responsibility_searches(
        self, faulty_model, fault_info, optimal_metrics, faulty_metrics, simulate_fn, work_dir,
        original_model: Dict = None
    ):
        """Run shift responsibility searches.

        The new approach enumerates compatible role pairs instead of shift percentages:
        - Grid Search: tries all (source, target) pairs with overlapping activities
        - LLM Search: proposes pairs based on utilization analysis

        Args:
            faulty_model: Model with injected workload imbalance
            fault_info: Information about the injected fault
            optimal_metrics: Target metrics to match
            faulty_metrics: Current faulty metrics
            simulate_fn: Cached simulation function
            work_dir: Working directory
            original_model: Original model (for finding all compatible pairs)
        """
        from_role = fault_info.get("source_role")  # The overloaded role
        to_role = fault_info.get("target_role")    # The underloaded role

        optimal_cycle_time = optimal_metrics.get("avg_cycle_time", 0)

        # Use faulty_model if original not provided
        model_for_pairs = original_model if original_model else faulty_model

        # Grid search baseline: use unique resources count (not N×(N-1) pairs)
        # This represents trying each resource as a potential overload source
        if self.skip_grid_search:
            # Count unique resources for realistic baseline
            grid_search_temp = ShiftResponsibilityGridSearch(tolerance_pct=self.shift_tolerance_pct)
            all_pairs = grid_search_temp._find_compatible_pairs(model_for_pairs)

            # Count unique resources (roles) involved in any pair
            unique_roles = set()
            for p in all_pairs:
                unique_roles.add(p[0])
                unique_roles.add(p[1])

            # Use unique resources count as baseline (more realistic than all pairs)
            theoretical_iterations = len(unique_roles)
            theoretical_iterations = max(1, theoretical_iterations)

            grid_result = ShiftResponsibilityGridSearchHistory(
                target_from_role=from_role,
                target_to_role=to_role,
                target_cycle_time=optimal_cycle_time,
                all_compatible_pairs=[(p[0], p[1]) for p in all_pairs],
                solution_found_at=theoretical_iterations
            )
            print(f"    [SKIP] Grid search - theoretical iterations: {theoretical_iterations} (unique resources)")
        else:
            grid_search = ShiftResponsibilityGridSearch(tolerance_pct=self.shift_tolerance_pct)
            grid_result = grid_search.search(
                faulty_model, from_role, to_role,
                optimal_metrics, faulty_metrics, simulate_fn, work_dir,
                model=model_for_pairs
            )

        # LLM search uses same role pair space
        num_pairs = len(grid_result.all_compatible_pairs) if hasattr(grid_result, 'all_compatible_pairs') else 10
        llm_search = ShiftResponsibilityLLMSearch(
            model_name=self.model_name, temperature=self.temperature,
            max_iterations=num_pairs, tolerance_pct=self.shift_tolerance_pct
        )
        llm_result = llm_search.search(
            faulty_model, from_role, to_role,
            optimal_metrics, faulty_metrics, simulate_fn, work_dir,
            model=model_for_pairs
        )

        return grid_result, llm_result

    def _simulate_model(
        self,
        model: Dict,
        modifications: Dict,
        work_dir: str,
        baseline_metrics: Dict,
        log_path: Optional[str] = None,
        experiment_type: Optional[str] = None
    ) -> Dict:
        """
        Run simulation with dual-simulator validation (DES + ABS).

        Per the evaluation framework proposal:
        - DES (Prosimos): Captures structural aspects (queues, capacity, waiting times)
        - ABS (AgentSimulator): Captures behavioral aspects (handoffs, routing)

        Combined Validation Score:
        - For Resource/Calendar: S_final = (S_DES + S_ABS) / 2
        - Accepted if: S_final > threshold AND S_DES > 0 AND S_ABS > 0

        Args:
            model: Simulation model
            modifications: Parameter modifications
            work_dir: Working directory
            baseline_metrics: Baseline metrics for comparison
            log_path: Optional path to event log (for ABS)
            experiment_type: Type of experiment (e.g., "calendar") - ABS is skipped for calendar

        Returns:
            Dictionary with combined metrics and validation scores
        """
        result = {}

        # Run DES (Prosimos) simulation - average over des_runs
        des_metrics_list = []
        for run_idx in range(self.des_runs):
            if self.des_runs > 1:
                print(f"      DES run {run_idx + 1}/{self.des_runs}...", end=" ")
            try:
                metrics = self.prosimos.simulate(
                    model=model,
                    modifications=modifications,
                    work_dir=work_dir,
                    baseline_metrics=baseline_metrics
                )
                if metrics:
                    des_metrics_list.append(metrics)
                    if self.des_runs > 1:
                        print(f"cycle_time={metrics.get('avg_cycle_time', 0):.2f}s")
            except Exception as e:
                print(f"      [!] DES run {run_idx + 1} failed: {e}")

        # Average DES results
        if des_metrics_list:
            result = self._average_metrics(des_metrics_list)
            if self.des_runs > 1:
                print(f"      DES averaged ({len(des_metrics_list)} runs): "
                      f"cycle_time={result.get('avg_cycle_time', 0):.2f}s")

        # Calculate DES validation score (improvement over baseline)
        des_baseline_cycle = baseline_metrics.get("avg_cycle_time", 0)
        des_current_cycle = result.get("avg_cycle_time", 0)
        if des_baseline_cycle > 0:
            des_improvement = (des_baseline_cycle - des_current_cycle) / des_baseline_cycle
            result["des_validation_score"] = des_improvement
        else:
            result["des_validation_score"] = 0.0

        # Run ABS (AgentSimulator) if enabled and available
        # Skip ABS for calendar experiments - ABS doesn't support calendar changes
        param_type = modifications.get("parameter_type") if modifications else None
        skip_abs = param_type == "calendar" or experiment_type == "calendar"

        if self.use_dual_validation and self.agent_simulator.available and log_path and not skip_abs:
            abs_metrics_list = []
            for run_idx in range(self.abs_runs):
                if self.abs_runs > 1:
                    print(f"      ABS run {run_idx + 1}/{self.abs_runs}...", end=" ")
                try:
                    abs_result = self.agent_simulator.simulate(
                        log_path=log_path,
                        modifications=modifications,
                        output_dir=os.path.join(work_dir, f"abs_run_{run_idx}"),
                        num_simulations=1
                    )
                    if abs_result and abs_result[0]:
                        abs_metrics_list.append(abs_result[0])
                        if self.abs_runs > 1:
                            print(f"throughput={abs_result[0].get('avg_throughput', 0):.2f}")
                except Exception as e:
                    print(f"      [!] ABS run {run_idx + 1} failed: {e}")

            # Average ABS results and compute validation score
            if abs_metrics_list:
                abs_averaged = self._average_metrics(abs_metrics_list)
                abs_baseline_throughput = baseline_metrics.get("avg_throughput", 0)
                abs_current_throughput = abs_averaged.get("avg_throughput", 0)
                abs_current_cycle_time = abs_averaged.get("avg_cycle_time", 0)

                # Store ABS metrics directly for shift responsibility searches
                result["abs_throughput"] = abs_current_throughput
                result["abs_cycle_time"] = abs_current_cycle_time  # Primary KPI for shift responsibility

                if abs_baseline_throughput > 0:
                    # For throughput, higher is better
                    abs_improvement = (abs_current_throughput - abs_baseline_throughput) / abs_baseline_throughput
                    result["abs_validation_score"] = abs_improvement
                else:
                    result["abs_validation_score"] = 0.0

                if self.abs_runs > 1:
                    print(f"      ABS averaged ({len(abs_metrics_list)} runs): "
                          f"cycle_time={abs_current_cycle_time:.2f}s, throughput={abs_current_throughput:.2f}")
            else:
                result["abs_validation_score"] = None
                result["abs_throughput"] = None
                result["abs_cycle_time"] = None
        elif skip_abs:
            # Calendar modifications use DES-only validation
            result["abs_validation_score"] = None
            result["abs_throughput"] = None
            result["abs_cycle_time"] = None
        else:
            result["abs_validation_score"] = None
            result["abs_throughput"] = None
            result["abs_cycle_time"] = None

        # Combined validation (per proposal)
        des_score = result.get("des_validation_score", 0)
        abs_score = result.get("abs_validation_score")

        if abs_score is not None:
            result["combined_validation_score"] = (des_score + abs_score) / 2
            result["dual_validation_passed"] = des_score > 0 and abs_score > 0
        else:
            result["combined_validation_score"] = des_score
            result["dual_validation_passed"] = des_score > 0

        return result

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average numeric metrics across multiple simulation runs."""
        if not metrics_list:
            return {}

        if len(metrics_list) == 1:
            return metrics_list[0]

        aggregated = {}
        numeric_keys = ["avg_cycle_time", "avg_throughput", "avg_waiting_time"]

        for key in numeric_keys:
            values = [float(m[key]) for m in metrics_list if key in m and m[key] is not None]
            if values:
                aggregated[key] = statistics.mean(values)

        # Average resource utilization
        resource_utils = {}
        for run in metrics_list:
            for resource_id, value in run.get("resource_utilization", {}).items():
                resource_utils.setdefault(resource_id, []).append(float(value))

        if resource_utils:
            aggregated["resource_utilization"] = {
                res_id: statistics.mean(vals) for res_id, vals in resource_utils.items()
            }

        return aggregated

    def _aggregate_results(self, result: AggregatedResult):
        """Compute aggregate statistics from individual runs."""
        runs = result.runs

        # Extract successful iterations
        baseline_iters = [r.baseline_iterations for r in runs if r.baseline_iterations]
        llm_iters = [r.llm_iterations for r in runs if r.llm_iterations]
        speedups = [r.speedup_ratio for r in runs if r.speedup_ratio]

        # Compute statistics
        if baseline_iters:
            result.baseline_iterations_mean = statistics.mean(baseline_iters)
            result.baseline_iterations_std = statistics.stdev(baseline_iters) if len(baseline_iters) > 1 else 0

        if llm_iters:
            result.llm_iterations_mean = statistics.mean(llm_iters)
            result.llm_iterations_std = statistics.stdev(llm_iters) if len(llm_iters) > 1 else 0

        if speedups:
            result.speedup_ratio_mean = statistics.mean(speedups)
            result.speedup_ratio_std = statistics.stdev(speedups) if len(speedups) > 1 else 0

        # Success rates
        result.baseline_success_rate = len(baseline_iters) / len(runs) if runs else 0
        result.llm_success_rate = len(llm_iters) / len(runs) if runs else 0
        result.llm_accuracy_rate = sum(1 for r in runs if r.llm_accuracy) / len(runs) if runs else 0

    def _print_header(self, experiments: List[ExperimentConfig]):
        """Print evaluation header."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION: LLM vs Baseline Search")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Repetitions per experiment: {self.num_repetitions}")
        print(f"DES runs per simulation: {self.des_runs}")
        print(f"ABS runs per simulation: {self.abs_runs}")
        print(f"Dual-simulator validation: {'Enabled' if self.use_dual_validation else 'Disabled'}")
        print(f"Skip grid search: {'Yes (using theoretical iterations)' if self.skip_grid_search else 'No'}")
        print(f"Tolerance by type: Resource={self.resource_tolerance_pct}%, Calendar={self.calendar_tolerance_pct}%, Shift={self.shift_tolerance_pct}%")
        print(f"\nExperiments to run:")
        for config in experiments:
            print(f"  {config.experiment_id}: {config.improvement_type.value} - {config.log_name}")
        print("=" * 80)

    def _print_experiment_summary(self, result: AggregatedResult):
        """Print summary for a single experiment."""
        print(f"\n  SUMMARY for {result.config.experiment_id}:")
        print(f"    Baseline: {result.baseline_iterations_mean:.1f} ± {result.baseline_iterations_std:.1f} iterations"
              if result.baseline_iterations_mean else "    Baseline: N/A")
        print(f"    LLM:      {result.llm_iterations_mean:.1f} ± {result.llm_iterations_std:.1f} iterations"
              if result.llm_iterations_mean else "    LLM:      N/A")
        print(f"    Speedup:  {result.speedup_ratio_mean:.2f}× ± {result.speedup_ratio_std:.2f}"
              if result.speedup_ratio_mean else "    Speedup:  N/A")
        print(f"    LLM success rate: {result.llm_success_rate:.0%}")

    def _print_final_summary(self, report: EvaluationReport):
        """Print final summary across all experiments."""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        agg = report.to_dict().get("aggregate_statistics", {})
        print(f"\nOverall Statistics:")
        print(f"  Average speedup: {agg.get('overall_avg_speedup', 'N/A'):.2f}×"
              if agg.get('overall_avg_speedup') else "  Average speedup: N/A")
        print(f"  LLM success rate: {agg.get('overall_llm_success_rate', 0):.0%}")
        print(f"  Baseline success rate: {agg.get('overall_baseline_success_rate', 0):.0%}")

        print("\nBy Improvement Type:")
        by_type = agg.get("by_improvement_type", {})
        for type_name, stats in by_type.items():
            print(f"  {type_name}:")
            print(f"    Experiments: {stats.get('num_experiments', 0)}")
            print(f"    Avg speedup: {stats.get('avg_speedup', 'N/A'):.2f}×"
                  if stats.get('avg_speedup') else "    Avg speedup: N/A")


def run_comprehensive_evaluation(
    logs_dir: str = "data/logs",
    output_dir: str = "output",
    model_name: str = "gpt-4.1",
    num_repetitions: int = 5,
    skip_grid_search: bool = False
):
    """
    Convenience function to run the comprehensive evaluation.

    Args:
        logs_dir: Directory containing event logs
        output_dir: Directory for outputs
        model_name: LLM model to use
        num_repetitions: Number of repetitions per experiment
        skip_grid_search: Skip running grid search, use theoretical iterations instead
    """
    evaluator = ComprehensiveEvaluator(
        model_name=model_name,
        num_repetitions=num_repetitions,
        skip_grid_search=skip_grid_search
    )
    return evaluator.run_full_evaluation(logs_dir, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive evaluation")
    parser.add_argument("--logs-dir", default="data/logs", help="Directory with event logs")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="gpt-4.1", help="LLM model name")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per experiment")
    parser.add_argument("--skip-grid-search", action="store_true",
                        help="Skip grid search, use theoretical iterations instead")

    args = parser.parse_args()

    run_comprehensive_evaluation(
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        num_repetitions=args.repetitions,
        skip_grid_search=args.skip_grid_search
    )
