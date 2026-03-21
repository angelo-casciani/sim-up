"""
Calendar Grid Search for Quantitative Evaluation.

Systematic search that tries adding hours incrementally from 1 to max_hours.
This serves as the baseline comparison for the LLM-guided search for calendar improvements.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both Grid
Search and LLM Search test the same hours value, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import copy
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class CalendarGridSearchResult:
    """Result of a single calendar grid search iteration."""
    iteration: int
    hours_added: float
    total_hours: float
    metrics: Dict
    is_solution: bool
    cycle_time: float
    waiting_time: float
    throughput: float


@dataclass
class CalendarGridSearchHistory:
    """Complete history of calendar grid search experiment."""
    target_calendar: str
    target_hours: float  # The optimal hours to add (ground truth)
    max_search_range: float
    results: List[CalendarGridSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_calendar": self.target_calendar,
            "target_hours": self.target_hours,
            "max_search_range": self.max_search_range,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "hours_added": r.hours_added,
                    "total_hours": r.total_hours,
                    "cycle_time": r.cycle_time,
                    "waiting_time": r.waiting_time,
                    "throughput": r.throughput,
                    "is_solution": r.is_solution,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class CalendarGridSearch:
    """
    Systematic grid search over calendar hours.

    Tries all values from 1 to max_hours for hours to add
    and measures simulation calls needed to find the optimal configuration.
    """

    def __init__(
        self,
        min_hours: float = 1.0,
        max_hours: float = 8.0,
        step_hours: float = 1.0,
        tolerance_pct: float = 5.0
    ):
        """
        Initialize calendar grid search.

        Args:
            min_hours: Minimum hours to add
            max_hours: Maximum hours to add
            step_hours: Step size for hour increments
            tolerance_pct: Tolerance % for considering KPIs as "matching" optimal
        """
        self.min_hours = min_hours
        self.max_hours = max_hours
        self.step_hours = step_hours
        self.tolerance_pct = tolerance_pct

    def search(
        self,
        faulty_model: Dict,
        target_calendar: str,
        optimal_hours_to_add: float,
        optimal_metrics: Dict,
        faulty_hours: float,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> CalendarGridSearchHistory:
        """
        Run systematic grid search to find optimal hours to add.

        Args:
            faulty_model: Model with injected fault (reduced hours)
            target_calendar: Name of the calendar to optimize
            optimal_hours_to_add: The known optimal hours to add (ground truth)
            optimal_metrics: Metrics from optimal model (target to match)
            faulty_hours: Current hours in faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            CalendarGridSearchHistory with all iterations and solution info
        """
        # Cap search range to optimal hours (can't exceed what was reduced)
        effective_max = min(self.max_hours, optimal_hours_to_add + 1)

        history = CalendarGridSearchHistory(
            target_calendar=target_calendar,
            target_hours=optimal_hours_to_add,
            max_search_range=effective_max
        )

        print(f"\n[Calendar Grid Search] Starting systematic search for {target_calendar}")
        print(f"  Target optimal hours to add: {optimal_hours_to_add:.1f}")
        print(f"  Current faulty hours/day: {faulty_hours:.1f}")
        print(f"  Search range: {self.min_hours:.1f}h to {effective_max:.1f}h (step: {self.step_hours}h)")
        print(f"  Solution criterion: must reach >= {optimal_hours_to_add:.1f}h (ground truth)")

        hours = self.min_hours
        iteration = 0
        while hours <= effective_max:
            iteration += 1
            history.total_iterations = iteration

            # Create modifications for this hours value
            modifications = {
                "parameter_type": "calendar",
                "specific_changes": [{
                    "target": target_calendar,
                    "change": "increase_hours",
                    "value": hours
                }]
            }

            # Run simulation
            total_hours = faulty_hours + hours
            print(f"  Iteration {iteration}: trying +{hours:.1f}h ({total_hours:.1f}h total)...", end=" ")
            metrics = simulate_fn(faulty_model, modifications)

            cycle_time = metrics.get("avg_cycle_time", float("inf"))
            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("avg_throughput", 0)

            # Solution criterion: must reach the ground truth hours (not just KPI tolerance)
            is_solution = hours >= optimal_hours_to_add

            result = CalendarGridSearchResult(
                iteration=iteration,
                hours_added=hours,
                total_hours=total_hours,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput
            )
            history.results.append(result)

            status = "FOUND!" if is_solution else ""
            print(f"waiting_time={waiting_time:.2f}s {status}")

            # Record first hit and stop searching
            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} (+{hours:.1f}h)")
                break

            hours += self.step_hours

        if history.solution_found_at is None:
            print(f"  [X] No solution found in search range [{self.min_hours:.1f}h-{effective_max:.1f}h]")

        return history

    def _is_within_tolerance(self, value: float, target: float) -> bool:
        """Check if value is within tolerance of target (or better).

        For waiting_time, lower is better, so we accept values that are:
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
