"""
Role Redesign Grid Search for Quantitative Evaluation.

Systematic search that tries shift percentages incrementally from 10% to 50%.
This serves as the baseline comparison for the LLM-guided search for role redesign.

Solution Detection: Based on KPI improvement (cycle_time within tolerance of optimal),
NOT on matching the exact injected shift fraction. This is consistent with how
calendar and resource grid searches work.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both Grid
Search and LLM Search test the same shift percentage, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import copy
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field


# Fixed shift fraction options for grid search
SHIFT_FRACTION_OPTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]


@dataclass
class RoleRedesignGridSearchResult:
    """Result of a single role redesign grid search iteration."""
    iteration: int
    from_role: str
    to_role: str
    shift_fraction: float
    metrics: Dict
    is_solution: bool
    cycle_time: float
    waiting_time: float
    throughput: float


@dataclass
class RoleRedesignGridSearchHistory:
    """Complete history of role redesign grid search experiment."""
    from_role: str  # Role to shift work FROM (overloaded)
    to_role: str    # Role to shift work TO (underloaded)
    target_cycle_time: float  # Optimal cycle time to reach
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

    Tries all shift values from 10% to 50% in order and measures simulation
    calls needed to find a configuration that brings KPIs within tolerance
    of optimal.
    """

    def __init__(
        self,
        shift_percentages: List[float] = None,
        tolerance_pct: float = 5.0
    ):
        """
        Initialize grid search.

        Args:
            shift_percentages: List of shift fractions to try (default: [0.1, 0.2, 0.3, 0.4, 0.5])
            tolerance_pct: Tolerance % for considering KPIs as "matching" optimal
        """
        self.shift_percentages = shift_percentages or SHIFT_FRACTION_OPTIONS
        self.tolerance_pct = tolerance_pct

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
            optimal_metrics: Metrics from optimal model (target to match)
            faulty_metrics: Current metrics from faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            RoleRedesignGridSearchHistory with all iterations and solution info
        """
        optimal_cycle_time = optimal_metrics.get("avg_cycle_time", 0)
        faulty_cycle_time = faulty_metrics.get("avg_cycle_time", 0)

        history = RoleRedesignGridSearchHistory(
            from_role=from_role,
            to_role=to_role,
            target_cycle_time=optimal_cycle_time
        )

        print(f"\n[Role Redesign Grid Search] Starting systematic search for {from_role} → {to_role}")
        print(f"  Faulty cycle time: {faulty_cycle_time:.2f}s")
        print(f"  Target cycle time: {optimal_cycle_time:.2f}s (tolerance: {self.tolerance_pct}%)")
        print(f"  Shift options: {[f'{p:.0%}' for p in self.shift_percentages]}")

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

            cycle_time = metrics.get("avg_cycle_time", float("inf"))
            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("avg_throughput", 0)

            # Solution found when cycle_time is within tolerance of optimal
            is_solution = self._is_within_tolerance(cycle_time, optimal_cycle_time)

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

            # Calculate improvement for logging
            improvement_pct = ((faulty_cycle_time - cycle_time) / faulty_cycle_time * 100) if faulty_cycle_time > 0 else 0
            status = "FOUND!" if is_solution else ""
            print(f"cycle_time={cycle_time:.2f}s ({improvement_pct:+.1f}% vs faulty) {status}")

            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} ({shift_pct:.0%} shift)")
                break

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {len(self.shift_percentages)} shift percentages")

        return history

    def _is_within_tolerance(self, value: float, target: float) -> bool:
        """Check if value is within tolerance of target (or better).

        For cycle_time, lower is better, so we accept values that are:
        - At most tolerance_pct% HIGHER than target (worse)
        - Any amount LOWER than target (better or equal)
        """
        if target == 0:
            return value == 0
        # Accept if value is better (lower) than target
        if value <= target:
            return True
        # Otherwise check if it's within tolerance (not too much worse)
        pct_diff = (value - target) / target * 100
        return pct_diff <= self.tolerance_pct
