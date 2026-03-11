"""
Grid Search for Quantitative Evaluation.

Systematic search that tries all resource combinations from 1 to max_count.
This serves as the baseline comparison for the LLM-guided search.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both Grid
Search and LLM Search test the same resource count (e.g., count=3), they
receive identical simulation results, eliminating random variance as a
confounding factor in the comparison.
"""

import copy
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class GridSearchResult:
    """Result of a single grid search iteration."""
    iteration: int
    resource_count: int
    metrics: Dict
    is_solution: bool
    cycle_time: float
    waiting_time: float
    throughput: float


@dataclass
class GridSearchHistory:
    """Complete history of grid search experiment."""
    target_resource: str
    target_count: int  # The optimal count we're searching for
    max_search_range: int
    results: List[GridSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_resource": self.target_resource,
            "target_count": self.target_count,
            "max_search_range": self.max_search_range,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "resource_count": r.resource_count,
                    "cycle_time": r.cycle_time,
                    "waiting_time": r.waiting_time,
                    "throughput": r.throughput,
                    "is_solution": r.is_solution,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class GridSearch:
    """
    Systematic grid search over resource counts.

    Tries all values from 1 to max_count for a target resource
    and measures simulation calls needed to find the optimal configuration.
    """

    def __init__(
        self,
        min_count: int = 1,
        max_count: int = 10,
        tolerance_pct: float = 2.0
    ):
        """
        Initialize grid search.

        Args:
            min_count: Minimum resource count to try
            max_count: Maximum resource count to try
            tolerance_pct: Tolerance % for considering KPIs as "matching" optimal
        """
        self.min_count = min_count
        self.max_count = max_count
        self.tolerance_pct = tolerance_pct

    def search(
        self,
        faulty_model: Dict,
        target_resource: str,
        optimal_count: int,
        optimal_metrics: Dict,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> GridSearchHistory:
        """
        Run systematic grid search to find optimal resource count.

        Args:
            faulty_model: Model with injected fault (resource count = 1)
            target_resource: Name of the resource to optimize
            optimal_count: The known optimal count (ground truth)
            optimal_metrics: Metrics from optimal model (target to match)
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            GridSearchHistory with all iterations and solution info
        """
        history = GridSearchHistory(
            target_resource=target_resource,
            target_count=optimal_count,
            max_search_range=self.max_count
        )

        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)

        print(f"\n[Grid Search] Starting systematic search for {target_resource}")
        print(f"  Target optimal count: {optimal_count}")
        print(f"  Search range: {self.min_count} to {self.max_count}")
        print(f"  Target cycle time: {optimal_cycle:.2f}s (tolerance: {self.tolerance_pct}%)")

        for count in range(self.min_count, self.max_count + 1):
            iteration = count - self.min_count + 1
            history.total_iterations = iteration

            # Create modifications for this count
            modifications = {
                "parameter_type": "resources",
                "specific_changes": [{
                    "target": target_resource,
                    "change": "increase_count",
                    "value": count
                }]
            }

            # Run simulation
            print(f"  Iteration {iteration}: trying count={count}...", end=" ")
            metrics = simulate_fn(faulty_model, modifications)

            cycle_time = metrics.get("avg_cycle_time", float("inf"))
            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("avg_throughput", 0)

            # Check if KPIs are within tolerance of optimal
            is_solution = self._is_within_tolerance(cycle_time, optimal_cycle)

            result = GridSearchResult(
                iteration=iteration,
                resource_count=count,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput
            )
            history.results.append(result)

            status = "FOUND!" if is_solution else ""
            print(f"cycle_time={cycle_time:.2f}s {status}")

            # Record first hit and stop searching
            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} (count={count})")
                break

        if history.solution_found_at is None:
            print(f"  [X] No solution found in search range [{self.min_count}-{self.max_count}]")

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
