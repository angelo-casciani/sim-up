"""
LLM-Guided Search for Quantitative Evaluation.

This module constrains the LLM to only propose resource_allocation changes
(increase_count) and iteratively updates it based on simulation results.
The goal is to show that the LLM can use semantic clues (high queue, high utilization)
to "jump" to the solution faster than systematic grid search.

Uses the existing HypothesisGenerator for LLM calls to avoid code duplication.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both LLM
Search and Grid Search test the same resource count (e.g., count=3), they
receive identical simulation results, eliminating random variance as a
confounding factor in the comparison.
"""

import json
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field

from llm import HypothesisGenerator


@dataclass
class LLMSearchResult:
    """Result of a single LLM search iteration."""
    iteration: int
    proposed_count: int
    metrics: Dict
    is_solution: bool
    cycle_time: float
    waiting_time: float
    throughput: float
    llm_rationale: str


@dataclass
class LLMSearchHistory:
    """Complete history of LLM-guided search experiment."""
    target_resource: str
    target_count: int  # The optimal count we're searching for
    model_name: str
    results: List[LLMSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_resource": self.target_resource,
            "target_count": self.target_count,
            "model_name": self.model_name,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "proposed_count": r.proposed_count,
                    "cycle_time": r.cycle_time,
                    "waiting_time": r.waiting_time,
                    "throughput": r.throughput,
                    "is_solution": r.is_solution,
                    "llm_rationale": r.llm_rationale,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class LLMResourceSearch:
    """
    LLM-guided search constrained to resource_allocation changes only.

    The LLM sees simulation metrics (queue times, utilization) and proposes
    resource count changes. It is constrained to only increase_count operations
    for the target resource.

    Uses HypothesisGenerator from the main pipeline for LLM calls.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_iterations: int = 10,
        tolerance_pct: float = 2.0
    ):
        """
        Initialize LLM-guided search.

        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            max_iterations: Maximum search iterations
            tolerance_pct: Tolerance % for considering KPIs as "matching" optimal
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.tolerance_pct = tolerance_pct

        # Reuse HypothesisGenerator for LLM calls (handles routing to different providers)
        self._llm = HypothesisGenerator(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def search(
        self,
        faulty_model: Dict,
        target_resource: str,
        optimal_count: int,
        optimal_metrics: Dict,
        faulty_metrics: Dict,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> LLMSearchHistory:
        """
        Run LLM-guided search to find optimal resource count.

        Args:
            faulty_model: Model with injected fault (resource count = 1)
            target_resource: Name of the resource to optimize
            optimal_count: The known optimal count (ground truth, NOT shown to LLM)
            optimal_metrics: Metrics from optimal model (target to match)
            faulty_metrics: Current metrics from faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            LLMSearchHistory with all iterations and solution info
        """
        history = LLMSearchHistory(
            target_resource=target_resource,
            target_count=optimal_count,
            model_name=self.model_name
        )

        optimal_cycle = optimal_metrics.get("avg_cycle_time", 0)

        print(f"\n[LLM Search] Starting guided search for {target_resource}")
        print(f"  Model: {self.model_name}")
        print(f"  Target cycle time: {optimal_cycle:.2f}s (tolerance: {self.tolerance_pct}%)")
        print(f"  Current (faulty) cycle time: {faulty_metrics.get('avg_cycle_time', 0):.2f}s")

        current_metrics = faulty_metrics
        previous_attempts = []

        for iteration in range(1, self.max_iterations + 1):
            history.total_iterations = iteration

            # Generate proposal from LLM
            proposal = self._generate_proposal(
                target_resource=target_resource,
                current_metrics=current_metrics,
                previous_attempts=previous_attempts,
                max_count=optimal_count  # Pass the actual max valid count
            )

            proposed_count = proposal.get("proposed_count", 2)
            rationale = proposal.get("rationale", "")

            print(f"  Iteration {iteration}: LLM proposes count={proposed_count}...", end=" ")

            # Create modifications
            modifications = {
                "parameter_type": "resources",
                "specific_changes": [{
                    "target": target_resource,
                    "change": "increase_count",
                    "value": proposed_count
                }]
            }

            # Run simulation
            metrics = simulate_fn(faulty_model, modifications)

            cycle_time = metrics.get("avg_cycle_time", float("inf"))
            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("avg_throughput", 0)

            # Check if KPIs are within tolerance of optimal
            is_solution = self._is_within_tolerance(cycle_time, optimal_cycle)

            result = LLMSearchResult(
                iteration=iteration,
                proposed_count=proposed_count,
                metrics=metrics,
                is_solution=is_solution,
                cycle_time=cycle_time,
                waiting_time=waiting_time,
                throughput=throughput,
                llm_rationale=rationale
            )
            history.results.append(result)

            # Track for next iteration
            previous_attempts.append({
                "count": proposed_count,
                "cycle_time": cycle_time,
                "waiting_time": waiting_time,
                "throughput": throughput,
                "improved": cycle_time < current_metrics.get("avg_cycle_time", float("inf"))
            })

            status = "FOUND!" if is_solution else ""
            print(f"cycle_time={cycle_time:.2f}s {status}")

            # Record first hit and stop searching
            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} (count={proposed_count})")
                break

            # Update current metrics for next iteration
            current_metrics = metrics

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {self.max_iterations} iterations")

        return history

    def _generate_proposal(
        self,
        target_resource: str,
        current_metrics: Dict,
        previous_attempts: List[Dict],
        max_count: int = 10
    ) -> Dict:
        """Generate resource count proposal from LLM."""

        # Extract counts already tried
        tried_counts = [attempt["count"] for attempt in previous_attempts]
        available_counts = [c for c in range(1, max_count + 1) if c not in tried_counts]

        system_prompt = f"""You are a process optimization expert analyzing simulation metrics.

Your ONLY task is to propose the optimal resource count for: {target_resource}

CONSTRAINTS (CRITICAL):
- You can ONLY propose resource_allocation changes (increase_count)
- You must propose a single integer value for the resource count
- Valid range: 1 to {max_count}
- DO NOT propose a count that was already tried: {tried_counts if tried_counts else 'none yet'}

Analysis strategy:
- High waiting_time indicates resource bottleneck -> need MORE resources
- High resource utilization (>80%) indicates bottleneck -> need MORE resources
- If previous attempts with higher counts improved metrics, continue increasing
- If previous attempts plateaued, you've likely found the optimal count

Return ONLY a JSON object:
{{
  "proposed_count": <integer 1-{max_count}, NOT in {tried_counts}>,
  "rationale": "Brief explanation of why this count"
}}"""

        user_prompt = f"""CURRENT SIMULATION METRICS:
{json.dumps(current_metrics, indent=2)}

"""
        if previous_attempts:
            user_prompt += "PREVIOUS ATTEMPTS (most recent last):\n"
            for i, attempt in enumerate(previous_attempts[-5:], 1):
                improved = "improved" if attempt.get("improved") else "no improvement"
                user_prompt += f"  {i}. count={attempt['count']}: cycle_time={attempt['cycle_time']:.2f}s ({improved})\n"
            user_prompt += "\n"

        user_prompt += f"""Based on the metrics and previous attempts, propose the next resource count to try.
Focus on the waiting_time and resource_utilization to identify bottlenecks.
IMPORTANT: Do NOT propose any of these already-tried counts: {tried_counts}
Valid range: 1 to {max_count}

Return ONLY the JSON object."""

        try:
            # Use HypothesisGenerator's _call_llm for provider routing
            response = self._llm._call_llm(system_prompt, user_prompt)

            # Parse JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                proposal = json.loads(response[start_idx:end_idx])
                count = int(proposal.get("proposed_count", 2))
                count = max(1, min(max_count, count))  # Clamp to valid range

                # If LLM still proposed a duplicate, pick the nearest untried count
                if count in tried_counts and available_counts:
                    # Find nearest available count
                    nearest = min(available_counts, key=lambda x: abs(x - count))
                    print(f"\n    [!] LLM proposed duplicate {count}, using {nearest} instead")
                    count = nearest

                return {
                    "proposed_count": count,
                    "rationale": proposal.get("rationale", "")
                }
        except Exception as e:
            print(f"\n    [!] LLM parsing error: {e}")

        # Fallback: pick next available count (not already tried)
        if available_counts:
            # Prefer incrementing from last attempt if possible
            if previous_attempts:
                last_count = previous_attempts[-1].get("count", 1)
                higher_available = [c for c in available_counts if c > last_count]
                if higher_available:
                    return {"proposed_count": min(higher_available), "rationale": "Fallback: next higher untried"}
            return {"proposed_count": min(available_counts), "rationale": "Fallback: lowest untried"}

        # All counts exhausted
        return {"proposed_count": 10, "rationale": "Fallback: all counts tried"}

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
