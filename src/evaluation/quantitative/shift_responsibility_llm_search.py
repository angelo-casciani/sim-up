"""
Shift Responsibility LLM-Guided Search for Quantitative Evaluation.

This module uses an LLM to propose which role pair to shift responsibility between,
based on utilization analysis and role-task mapping.

Search Space:
- LLM analyzes role utilizations and overlapping activities
- Proposes a (source, target) role pair to try shift_responsibility
- Solution: KPIs improve (validated improvement in simulation)

This approach matches what the actual pipeline supports - coarse role-to-role
shift_responsibility actions without fine-grained percentage parameterization.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both LLM
Search and Grid Search test the same role pair, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import json
import re
from typing import Dict, List, Callable, Optional, Set, Tuple
from dataclasses import dataclass, field

from llm import HypothesisGenerator


# Tolerance for considering KPIs as "improved"
TOLERANCE_PCT = 5.0  # Within 5% of optimal cycle time


@dataclass
class ShiftResponsibilityLLMResult:
    """Result of a single shift responsibility LLM search iteration."""
    iteration: int
    from_role: str
    to_role: str
    metrics: Dict
    is_solution: bool
    cycle_time: float  # ABS cycle_time (primary KPI)
    waiting_time: float
    throughput: float
    llm_rationale: str
    shared_activities: List[str]


@dataclass
class ShiftResponsibilityLLMHistory:
    """Complete history of shift responsibility LLM-guided search experiment."""
    target_from_role: str  # Ground truth: shift FROM this role
    target_to_role: str    # Ground truth: shift TO this role
    target_cycle_time: float  # Target ABS cycle time (optimal)
    model_name: str
    all_compatible_pairs: List[Tuple[str, str]]  # All possible pairs
    results: List[ShiftResponsibilityLLMResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_from_role": self.target_from_role,
            "target_to_role": self.target_to_role,
            "target_cycle_time": self.target_cycle_time,
            "model_name": self.model_name,
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
                    "llm_rationale": r.llm_rationale,
                    "shared_activities": r.shared_activities,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class ShiftResponsibilityLLMSearch:
    """
    LLM-guided search for shift responsibility optimization.

    Uses an LLM to intelligently propose which role pair to shift work between,
    based on utilization analysis and task overlap information.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_iterations: int = 10,
        tolerance_pct: float = None
    ):
        """
        Initialize LLM-guided shift responsibility search.

        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            max_iterations: Maximum search iterations
            tolerance_pct: Tolerance % for considering cycle_time as "matching" optimal (default: 5%)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.tolerance_pct = tolerance_pct if tolerance_pct is not None else TOLERANCE_PCT

        self._llm = HypothesisGenerator(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _get_role_task_mapping(self, model: Dict) -> Dict[str, Set[str]]:
        """Extract role -> tasks mapping from model."""
        role_tasks = {}
        task_distributions = model.get("prosimos_params", {}).get("task_resource_distribution", [])

        for task in task_distributions:
            task_id = task.get("task_id", "")
            resources = task.get("resources", [])

            for res in resources:
                res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                role = re.sub(r'-\d+$', '', res_id).strip()

                if role not in role_tasks:
                    role_tasks[role] = set()
                role_tasks[role].add(task_id)

        return role_tasks

    def _find_compatible_pairs(self, model: Dict) -> List[Tuple[str, str, Set[str]]]:
        """Find all compatible (source, target) role pairs."""
        role_tasks = self._get_role_task_mapping(model)
        pairs = []

        roles = list(role_tasks.keys())
        for role1 in roles:
            for role2 in roles:
                if role1 == role2:
                    continue

                shared = role_tasks[role1] & role_tasks[role2]
                if shared:
                    pairs.append((role1, role2, shared))

        return pairs

    def _calculate_role_utilization(self, metrics: Dict, model: Dict) -> Dict[str, float]:
        """Calculate average utilization per role."""
        utilization = metrics.get("resource_utilization", {})
        role_utils = {}

        for res_id, util_val in utilization.items():
            res_norm = res_id.replace("_profile", "").replace("_calendar", "")
            role = re.sub(r'-\d+$', '', res_norm).strip()

            if role not in role_utils:
                role_utils[role] = []
            role_utils[role].append(float(util_val))

        # Average across resources in each role
        return {
            role: sum(vals) / len(vals) if vals else 0.5
            for role, vals in role_utils.items()
        }

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
    ) -> ShiftResponsibilityLLMHistory:
        """
        Run LLM-guided search to find the optimal role pair for shift_responsibility.

        Args:
            faulty_model: Model with injected fault (workload imbalance)
            from_role: Ground truth - role to shift work FROM (overloaded)
            to_role: Ground truth - role to shift work TO (underloaded)
            optimal_metrics: Metrics from optimal model - TARGET TO MATCH
            faulty_metrics: Current metrics from faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation
            model: Original model (for finding compatible pairs)

        Returns:
            ShiftResponsibilityLLMHistory with all iterations and solution info
        """
        if model is None:
            model = faulty_model

        # Get ABS cycle_time as primary KPI
        optimal_cycle = optimal_metrics.get("abs_cycle_time", 0)
        faulty_cycle = faulty_metrics.get("abs_cycle_time", 0)

        if optimal_cycle == 0:
            print("  [WARNING] ABS cycle_time not available, falling back to DES")
            optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)
            faulty_cycle = faulty_metrics.get("avg_cycle_time", 0)

        # Find all compatible role pairs
        all_pairs = self._find_compatible_pairs(model)
        pair_tuples = [(p[0], p[1]) for p in all_pairs]

        # Build pair -> shared activities lookup
        pair_to_shared = {(p[0], p[1]): p[2] for p in all_pairs}

        history = ShiftResponsibilityLLMHistory(
            target_from_role=from_role,
            target_to_role=to_role,
            target_cycle_time=optimal_cycle,
            model_name=self.model_name,
            all_compatible_pairs=pair_tuples
        )

        print(f"\n[Shift Responsibility LLM Search] Starting guided search")
        print(f"  Model: {self.model_name}")
        print(f"  Ground truth: {from_role} → {to_role}")
        print(f"  Faulty ABS cycle_time: {faulty_cycle:.2f}s")
        print(f"  Optimal ABS cycle_time: {optimal_cycle:.2f}s (target, ±{self.tolerance_pct}%)")
        print(f"  Compatible role pairs available: {len(all_pairs)}")

        current_metrics = faulty_metrics
        previous_attempts = []
        tried_pairs = set()
        best_diff_pct = float("inf")
        best_pair = None

        # Cap iterations to number of available pairs
        max_iters = min(self.max_iterations, len(all_pairs))

        for iteration in range(1, max_iters + 1):
            history.total_iterations = iteration

            # Calculate current role utilizations
            role_utils = self._calculate_role_utilization(current_metrics, model)

            # Get available pairs (not yet tried)
            available_pairs = [p for p in pair_tuples if p not in tried_pairs]

            if not available_pairs:
                print(f"  [X] No more role pairs to try")
                break

            # Generate proposal from LLM
            proposal = self._generate_proposal(
                role_utils=role_utils,
                available_pairs=available_pairs,
                pair_to_shared=pair_to_shared,
                previous_attempts=previous_attempts,
                faulty_cycle=faulty_cycle,
                optimal_cycle=optimal_cycle
            )

            source = proposal.get("from_role", available_pairs[0][0])
            target = proposal.get("to_role", available_pairs[0][1])
            rationale = proposal.get("rationale", "")

            # Ensure pair is valid and available
            if (source, target) not in available_pairs:
                # Find closest valid pair
                for avail_source, avail_target in available_pairs:
                    if avail_source.lower() == source.lower():
                        source, target = avail_source, avail_target
                        break
                else:
                    source, target = available_pairs[0]

            tried_pairs.add((source, target))
            shared = pair_to_shared.get((source, target), set())

            print(f"  Iteration {iteration}: LLM proposes {source} → {target}...", end=" ")

            # Create modifications
            modifications = {
                "parameter_type": "shift_responsibility",
                "specific_changes": [{
                    "target": f"{source} → {target}",
                    "change": "shift_responsibility",
                    "value": 1.0
                }]
            }

            # Run simulation
            metrics = simulate_fn(faulty_model, modifications)

            # Get cycle_time
            cycle_time = metrics.get("abs_cycle_time", 0)
            if cycle_time == 0:
                cycle_time = metrics.get("avg_cycle_time", float("inf"))

            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("abs_throughput", metrics.get("avg_throughput", 0))

            # Calculate difference from optimal
            diff_pct = abs(cycle_time - optimal_cycle) / optimal_cycle * 100 if optimal_cycle > 0 else float("inf")

            # Solution found when within tolerance
            is_solution = diff_pct <= self.tolerance_pct

            # Track best
            if diff_pct < best_diff_pct:
                best_diff_pct = diff_pct
                best_pair = (source, target)

            result = ShiftResponsibilityLLMResult(
                iteration=iteration,
                from_role=source,
                to_role=target,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput,
                llm_rationale=rationale,
                shared_activities=list(shared)
            )
            history.results.append(result)

            # Track for next iteration
            previous_attempts.append({
                "from_role": source,
                "to_role": target,
                "cycle_time": cycle_time,
                "diff_pct": diff_pct,
                "improved": cycle_time < faulty_cycle
            })

            # Show result
            direction = "+" if cycle_time > optimal_cycle else "-"
            status = "FOUND!" if is_solution else ""
            print(f"cycle_time={cycle_time:.2f}s ({direction}{diff_pct:.1f}% vs optimal) {status}")

            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} ({source} → {target})")
                break

            current_metrics = metrics

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {history.total_iterations} iterations")
            if best_pair:
                print(f"      Best: {best_diff_pct:.1f}% from optimal at {best_pair[0]} → {best_pair[1]}")

        return history

    def _generate_proposal(
        self,
        role_utils: Dict[str, float],
        available_pairs: List[Tuple[str, str]],
        pair_to_shared: Dict[Tuple[str, str], Set[str]],
        previous_attempts: List[Dict],
        faulty_cycle: float,
        optimal_cycle: float
    ) -> Dict:
        """Generate role pair proposal from LLM."""

        # Sort roles by utilization
        sorted_roles = sorted(role_utils.items(), key=lambda x: x[1], reverse=True)
        overloaded = [f"{r}({u:.2f})" for r, u in sorted_roles if u > 0.7]
        underloaded = [f"{r}({u:.2f})" for r, u in sorted_roles if u < 0.3]

        # Format available pairs with their shared activity counts
        pairs_info = []
        for source, target in available_pairs[:15]:  # Limit for prompt size
            shared = pair_to_shared.get((source, target), set())
            source_util = role_utils.get(source, 0.5)
            target_util = role_utils.get(target, 0.5)
            pairs_info.append(
                f"  {source} ({source_util:.2f}) → {target} ({target_util:.2f}): {len(shared)} shared tasks"
            )

        system_prompt = f"""You are a process optimization expert analyzing workload distribution between roles.

TASK: Select the best role pair to shift responsibility FROM source TO target.

CONTEXT:
- The process has a workload imbalance causing poor performance
- Goal: Find which role pair to shift work between to improve cycle time
- Current faulty cycle_time: {faulty_cycle:.2f}s
- Optimal target cycle_time: {optimal_cycle:.2f}s (within ±{self.tolerance_pct}%)

ROLE UTILIZATIONS (higher = more overloaded):
{chr(10).join(f"  {role}: {util:.2%}" for role, util in sorted_roles)}

OVERLOADED ROLES (>70%): {overloaded if overloaded else 'none'}
UNDERLOADED ROLES (<30%): {underloaded if underloaded else 'none'}

AVAILABLE ROLE PAIRS (source → target, with shared task count):
{chr(10).join(pairs_info)}
{f'... and {len(available_pairs) - 15} more pairs' if len(available_pairs) > 15 else ''}

STRATEGY:
- Shift work FROM overloaded roles TO underloaded roles
- Prefer pairs with more shared tasks (more flexibility)
- Higher utilization difference = more potential improvement

Return ONLY a JSON object:
{{
  "from_role": "<source role name exactly as shown>",
  "to_role": "<target role name exactly as shown>",
  "rationale": "Brief explanation"
}}"""

        user_prompt = ""
        if previous_attempts:
            user_prompt = "PREVIOUS ATTEMPTS:\n"
            for attempt in previous_attempts:
                improved = "✓ improved" if attempt["improved"] else "✗ no improvement"
                user_prompt += f"  {attempt['from_role']} → {attempt['to_role']}: cycle_time={attempt['cycle_time']:.2f}s ({attempt['diff_pct']:.1f}% from optimal) {improved}\n"
            user_prompt += "\nBased on the results, propose the next role pair to try.\n"
        else:
            user_prompt = "This is the first attempt. Propose the most promising role pair based on utilization analysis.\n"

        user_prompt += "\nReturn ONLY the JSON object."

        try:
            response = self._llm._call_llm(system_prompt, user_prompt)

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                proposal = json.loads(response[start_idx:end_idx])
                return {
                    "from_role": proposal.get("from_role", ""),
                    "to_role": proposal.get("to_role", ""),
                    "rationale": proposal.get("rationale", "")
                }
        except Exception as e:
            print(f"\n    [!] LLM parsing error: {e}")

        # Fallback: pick pair with highest utilization difference
        if available_pairs:
            best_pair = None
            best_diff = -1
            for source, target in available_pairs:
                diff = role_utils.get(source, 0.5) - role_utils.get(target, 0.5)
                if diff > best_diff:
                    best_diff = diff
                    best_pair = (source, target)

            if best_pair:
                return {
                    "from_role": best_pair[0],
                    "to_role": best_pair[1],
                    "rationale": "Fallback: highest utilization difference"
                }

        return {
            "from_role": available_pairs[0][0] if available_pairs else "",
            "to_role": available_pairs[0][1] if available_pairs else "",
            "rationale": "Fallback: first available pair"
        }
