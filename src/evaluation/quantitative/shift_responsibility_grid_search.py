"""
Shift Responsibility Grid Search for Quantitative Evaluation.

Systematic search that tries all compatible (source, target) role pairs.
This serves as the baseline comparison for the LLM-guided search.

Search Space:
- Enumerate all role pairs (source, target) that share overlapping activities
- For each pair, try shifting responsibility from source to target
- Solution: KPIs improve (validated improvement in simulation)

This approach matches what the actual pipeline supports - coarse role-to-role
shift_responsibility actions without fine-grained percentage parameterization.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both Grid
Search and LLM Search test the same role pair, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import copy
from typing import Dict, List, Callable, Optional, Set, Tuple
from dataclasses import dataclass, field


# Tolerance for considering KPIs as "improved"
TOLERANCE_PCT = 5.0  # Within 5% of optimal cycle time


@dataclass
class ShiftResponsibilityGridSearchResult:
    """Result of a single shift responsibility grid search iteration."""
    iteration: int
    from_role: str
    to_role: str
    metrics: Dict
    is_solution: bool
    cycle_time: float  # ABS cycle_time (primary KPI for shift responsibility)
    waiting_time: float
    throughput: float
    shared_activities: List[str]


@dataclass
class ShiftResponsibilityGridSearchHistory:
    """Complete history of shift responsibility grid search experiment."""
    target_from_role: str  # Ground truth: shift FROM this role
    target_to_role: str    # Ground truth: shift TO this role
    target_cycle_time: float  # Target ABS cycle time (optimal)
    all_compatible_pairs: List[Tuple[str, str]]  # All (source, target) pairs enumerated
    results: List[ShiftResponsibilityGridSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_from_role": self.target_from_role,
            "target_to_role": self.target_to_role,
            "target_cycle_time": self.target_cycle_time,
            "all_compatible_pairs": self.all_compatible_pairs,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "from_role": r.from_role,
                    "to_role": r.to_role,
                    "cycle_time": r.cycle_time,
                    "waiting_time": r.waiting_time,
                    "throughput": r.throughput,
                    "is_solution": r.is_solution,
                    "shared_activities": r.shared_activities,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class ShiftResponsibilityGridSearch:
    """
    Systematic grid search over compatible role pairs.

    Enumerates all (source, target) role pairs that share overlapping activities
    and tries shift_responsibility for each pair until finding a validated improvement.
    """

    def __init__(
        self,
        tolerance_pct: float = None
    ):
        """
        Initialize grid search.

        Args:
            tolerance_pct: Tolerance % for considering cycle_time as "matching" optimal (default: 5%)
        """
        self.tolerance_pct = tolerance_pct if tolerance_pct is not None else TOLERANCE_PCT

    def _get_role_task_mapping(self, model: Dict) -> Dict[str, Set[str]]:
        """
        Extract role -> tasks mapping from model's task_resource_distribution.

        Returns dict: role_name -> set of task_ids that role can perform
        """
        import re

        role_tasks = {}
        task_distributions = model.get("prosimos_params", {}).get("task_resource_distribution", [])

        for task in task_distributions:
            task_id = task.get("task_id", "")
            resources = task.get("resources", [])

            for res in resources:
                res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                # Extract role name (e.g., "Clerk-000001" -> "Clerk")
                role = re.sub(r'-\d+$', '', res_id).strip()

                if role not in role_tasks:
                    role_tasks[role] = set()
                role_tasks[role].add(task_id)

        return role_tasks

    def _find_compatible_pairs(self, model: Dict) -> List[Tuple[str, str, Set[str]]]:
        """
        Find all compatible (source, target) role pairs that share overlapping activities.

        Returns list of (source_role, target_role, shared_activities) tuples.
        Each pair appears twice (A->B and B->A) since shifting can go either direction.
        """
        role_tasks = self._get_role_task_mapping(model)
        pairs = []

        roles = list(role_tasks.keys())
        for i, role1 in enumerate(roles):
            for role2 in roles:
                if role1 == role2:
                    continue

                shared = role_tasks[role1] & role_tasks[role2]
                if shared:
                    # role1 -> role2 (shift from role1 to role2)
                    pairs.append((role1, role2, shared))

        return pairs

    def search(
        self,
        faulty_model: Dict,
        from_role: str,
        to_role: str,
        optimal_metrics: Dict,
        faulty_metrics: Dict,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str,
        model: Dict = None
    ) -> ShiftResponsibilityGridSearchHistory:
        """
        Run systematic grid search over all compatible role pairs.

        Args:
            faulty_model: Model with injected fault (workload imbalance)
            from_role: Ground truth - role to shift work FROM (overloaded)
            to_role: Ground truth - role to shift work TO (underloaded)
            optimal_metrics: Metrics from optimal model - TARGET TO MATCH
            faulty_metrics: Current metrics from faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation
            model: Original model (for finding all compatible pairs)

        Returns:
            ShiftResponsibilityGridSearchHistory with all iterations and solution info
        """
        # Use faulty_model if original model not provided
        if model is None:
            model = faulty_model

        # Get ABS cycle_time as primary KPI
        optimal_cycle = optimal_metrics.get("abs_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("abs_cycle_time", 0)

        # Fallback to DES if ABS not available
        if optimal_cycle == 0:
            print("  [WARNING] ABS cycle_time not available, falling back to DES")
            optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
            faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)

        # Find all compatible role pairs
        all_pairs = self._find_compatible_pairs(model)

        # Convert to list of (from, to) tuples for history
        pair_tuples = [(p[0], p[1]) for p in all_pairs]

        history = ShiftResponsibilityGridSearchHistory(
            target_from_role=from_role,
            target_to_role=to_role,
            target_cycle_time=optimal_cycle,
            all_compatible_pairs=pair_tuples
        )

        print(f"\n[Shift Responsibility Grid Search] Starting systematic search")
        print(f"  Ground truth: {from_role} → {to_role}")
        print(f"  Faulty ABS cycle_time: {faulty_cycle:.2f}s")
        print(f"  Optimal ABS cycle_time: {optimal_cycle:.2f}s (target, ±{self.tolerance_pct}%)")
        print(f"  Compatible role pairs to enumerate: {len(all_pairs)}")

        best_diff_pct = float("inf")
        best_pair = None

        for iteration, (source, target, shared) in enumerate(all_pairs, 1):
            history.total_iterations = iteration

            # Create modifications for this role pair
            modifications = {
                "parameter_type": "shift_responsibility",
                "specific_changes": [{
                    "target": f"{source} → {target}",
                    "change": "shift_responsibility",
                    "value": 1.0  # Full shift (coarse action)
                }]
            }

            print(f"  Iteration {iteration}: trying {source} → {target}...", end=" ")
            metrics = simulate_fn(faulty_model, modifications)

            # Get ABS cycle_time (primary KPI), fallback to DES
            cycle_time = metrics.get("abs_cycle_time", 0)
            if cycle_time == 0:
                cycle_time = metrics.get("avg_cycle_time", float("inf"))

            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("abs_throughput", metrics.get("avg_throughput", 0))

            # Calculate difference from optimal (symmetric)
            diff_pct = abs(cycle_time - optimal_cycle) / optimal_cycle * 100 if optimal_cycle > 0 else float("inf")

            # Solution found when within tolerance of optimal
            is_solution = diff_pct <= self.tolerance_pct

            # Track best result
            if diff_pct < best_diff_pct:
                best_diff_pct = diff_pct
                best_pair = (source, target)

            result = ShiftResponsibilityGridSearchResult(
                iteration=iteration,
                from_role=source,
                to_role=target,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput,
                shared_activities=list(shared)
            )
            history.results.append(result)

            # Show result
            direction = "+" if cycle_time > optimal_cycle else "-"
            status = "FOUND!" if is_solution else ""
            print(f"cycle_time={cycle_time:.2f}s ({direction}{diff_pct:.1f}% vs optimal) {status}")

            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} ({source} → {target})")
                break

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {len(all_pairs)} role pairs")
            if best_pair:
                print(f"      Best: {best_diff_pct:.1f}% from optimal at {best_pair[0]} → {best_pair[1]}")

        return history
