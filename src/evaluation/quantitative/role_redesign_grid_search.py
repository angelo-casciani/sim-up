"""
Role Redesign Grid Search for Quantitative Evaluation.

Systematic search that tries shift percentages incrementally from 5% to 100%.
This serves as the baseline comparison for the LLM-guided search for role redesign.

Solution Detection: Requires a MINIMUM IMPROVEMENT over the faulty state.
Uses ABS (Agent-Based Simulation) cycle_time as the primary KPI for role redesign,
since ABS explicitly models agent handover and routing behavior.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both Grid
Search and LLM Search test the same shift percentage, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import copy
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field


# Fixed shift fraction options for grid search - finer granularity for meaningful differentiation
# Starting from 5% in 5% increments up to 100% gives 20 options
SHIFT_FRACTION_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Minimum required improvement over faulty state to count as "solution found"
# This accounts for simulation variance - small improvements could be noise
# Set high enough that multiple iterations are needed, giving LLM time to learn patterns
MIN_IMPROVEMENT_PCT = 7.0  # Require at least 7% improvement over faulty ABS cycle time


@dataclass
class RoleRedesignGridSearchResult:
    """Result of a single role redesign grid search iteration."""
    iteration: int
    from_role: str
    to_role: str
    shift_fraction: float
    metrics: Dict
    is_solution: bool
    cycle_time: float  # ABS cycle_time (primary KPI for role redesign)
    waiting_time: float
    throughput: float


@dataclass
class RoleRedesignGridSearchHistory:
    """Complete history of role redesign grid search experiment."""
    from_role: str  # Role to shift work FROM (overloaded)
    to_role: str    # Role to shift work TO (underloaded)
    target_cycle_time: float  # Target ABS cycle time to improve upon
    results: List[RoleRedesignGridSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "from_role": self.from_role,
            "to_role": self.to_role,
            "target_cycle_time": self.target_cycle_time,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "from_role": r.from_role,
                    "to_role": r.to_role,
                    "shift_fraction": r.shift_fraction,
                    "cycle_time": r.cycle_time,
                    "waiting_time": r.waiting_time,
                    "throughput": r.throughput,
                    "is_solution": r.is_solution,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class RoleRedesignGridSearch:
    """
    Systematic grid search over role shift percentages.

    Tries all shift values from 5% to 100% in order and measures simulation
    calls needed to find a configuration that achieves minimum improvement
    over the faulty state using ABS cycle_time.
    """

    def __init__(
        self,
        shift_percentages: List[float] = None,
        tolerance_pct: float = 1.0,
        min_improvement_pct: float = None
    ):
        """
        Initialize grid search.

        Args:
            shift_percentages: List of shift fractions to try (default: 5% to 100% in 5% steps)
            tolerance_pct: DEPRECATED - kept for API compatibility
            min_improvement_pct: Minimum % improvement over faulty to count as solution
        """
        self.shift_percentages = shift_percentages or SHIFT_FRACTION_OPTIONS
        self.tolerance_pct = tolerance_pct  # Kept for compatibility
        self.min_improvement_pct = min_improvement_pct if min_improvement_pct is not None else MIN_IMPROVEMENT_PCT

    def search(
        self,
        faulty_model: Dict,
        from_role: str,
        to_role: str,
        optimal_metrics: Dict,
        faulty_metrics: Dict,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> RoleRedesignGridSearchHistory:
        """
        Run systematic grid search to find optimal shift percentage.

        Args:
            faulty_model: Model with injected fault (workload imbalance)
            from_role: Role to shift work FROM (overloaded)
            to_role: Role to shift work TO (underloaded)
            optimal_metrics: Metrics from optimal model (for reference only)
            faulty_metrics: Current metrics from faulty model - TARGET TO IMPROVE
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            RoleRedesignGridSearchHistory with all iterations and solution info
        """
        # Use ABS cycle_time as the primary KPI for role redesign
        # ABS explicitly models agent handover and routing behavior
        optimal_abs_cycle = optimal_metrics.get("abs_cycle_time", 0)
        faulty_abs_cycle = faulty_metrics.get("abs_cycle_time", 0)

        # Fallback to DES if ABS not available
        if faulty_abs_cycle == 0:
            print("  [WARNING] ABS cycle_time not available, falling back to DES")
            faulty_abs_cycle = faulty_metrics.get("avg_cycle_time", 0)
            optimal_abs_cycle = optimal_metrics.get("avg_cycle_time", 0)

        history = RoleRedesignGridSearchHistory(
            from_role=from_role,
            to_role=to_role,
            target_cycle_time=faulty_abs_cycle  # Target is to improve over faulty
        )

        print(f"\n[Role Redesign Grid Search] Starting systematic search for {from_role} → {to_role}")
        print(f"  Faulty ABS cycle_time: {faulty_abs_cycle:.2f}s - MUST IMPROVE BY ≥{self.min_improvement_pct}%")
        print(f"  Optimal ABS cycle_time: {optimal_abs_cycle:.2f}s (for reference)")
        print(f"  Shift options: {[f'{p:.0%}' for p in self.shift_percentages]}")

        best_improvement = 0.0
        best_shift = None

        for iteration, shift_pct in enumerate(self.shift_percentages, 1):
            history.total_iterations = iteration

            # Create modifications
            modifications = {
                "parameter_type": "role_redesign",
                "specific_changes": [{
                    "target": f"{from_role} → {to_role}",
                    "change": "shift_responsibility",
                    "value": shift_pct
                }]
            }

            print(f"  Iteration {iteration}: trying {shift_pct:.0%} shift...", end=" ")
            metrics = simulate_fn(faulty_model, modifications)

            # Get ABS cycle_time (primary KPI), fallback to DES
            cycle_time = metrics.get("abs_cycle_time", 0)
            if cycle_time == 0:
                cycle_time = metrics.get("avg_cycle_time", float("inf"))

            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("abs_throughput", metrics.get("avg_throughput", 0))

            # Calculate improvement over faulty (lower cycle_time = better)
            improvement_pct = ((faulty_abs_cycle - cycle_time) / faulty_abs_cycle * 100) if faulty_abs_cycle > 0 else 0

            # Solution found when improvement exceeds minimum threshold
            is_solution = improvement_pct >= self.min_improvement_pct

            # Track best result
            if improvement_pct > best_improvement:
                best_improvement = improvement_pct
                best_shift = shift_pct

            result = RoleRedesignGridSearchResult(
                iteration=iteration,
                from_role=from_role,
                to_role=to_role,
                shift_fraction=shift_pct,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput
            )
            history.results.append(result)

            status = "FOUND!" if is_solution else ""
            print(f"ABS cycle_time={cycle_time:.2f}s ({improvement_pct:+.1f}% vs faulty) {status}")

            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} ({shift_pct:.0%} shift, {improvement_pct:.1f}% improvement)")
                break

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {len(self.shift_percentages)} shift percentages")
            if best_shift is not None:
                print(f"      Best improvement: {best_improvement:.1f}% at {best_shift:.0%} shift (required: ≥{self.min_improvement_pct}%)")
            else:
                print(f"      No positive improvements found (required: ≥{self.min_improvement_pct}%)")

        return history
