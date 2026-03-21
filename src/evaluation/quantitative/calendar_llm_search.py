"""
Calendar LLM-Guided Search for Quantitative Evaluation.

This module constrains the LLM to only propose calendar changes
(increase_hours) and iteratively updates it based on simulation results.
The goal is to show that the LLM can use semantic clues (high utilization,
high waiting times) to "jump" to the solution faster than systematic grid search.

Uses the existing HypothesisGenerator for LLM calls to avoid code duplication.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both LLM
Search and Grid Search test the same hours value, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import json
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field

from llm import HypothesisGenerator


@dataclass
class CalendarLLMSearchResult:
    """Result of a single calendar LLM search iteration."""
    iteration: int
    proposed_hours: float
    total_hours: float
    metrics: Dict
    is_solution: bool
    cycle_time: float
    waiting_time: float
    throughput: float
    llm_rationale: str


@dataclass
class CalendarLLMSearchHistory:
    """Complete history of calendar LLM-guided search experiment."""
    target_calendar: str
    target_hours: float  # The optimal hours to add (ground truth)
    model_name: str
    results: List[CalendarLLMSearchResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "target_calendar": self.target_calendar,
            "target_hours": self.target_hours,
            "model_name": self.model_name,
            "solution_found_at": self.solution_found_at,
            "total_iterations": self.total_iterations,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "proposed_hours": r.proposed_hours,
                    "total_hours": r.total_hours,
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


class CalendarLLMSearch:
    """
    LLM-guided search constrained to calendar changes only.

    The LLM sees simulation metrics (waiting times, utilization) and proposes
    hours to add. It is constrained to only increase_hours operations
    for the target calendar.

    Uses HypothesisGenerator from the main pipeline for LLM calls.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_iterations: int = 10,
        tolerance_pct: float = 5.0
    ):
        """
        Initialize LLM-guided calendar search.

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
        target_calendar: str,
        optimal_hours_to_add: float,
        optimal_metrics: Dict,
        faulty_metrics: Dict,
        faulty_hours: float,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> CalendarLLMSearchHistory:
        """
        Run LLM-guided search to find optimal hours to add.

        Args:
            faulty_model: Model with injected fault (reduced hours)
            target_calendar: Name of the calendar to optimize
            optimal_hours_to_add: The known optimal hours (ground truth, NOT shown to LLM)
            optimal_metrics: Metrics from optimal model (target to match)
            faulty_metrics: Current metrics from faulty model
            faulty_hours: Current hours/day in faulty model
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            CalendarLLMSearchHistory with all iterations and solution info
        """
        history = CalendarLLMSearchHistory(
            target_calendar=target_calendar,
            target_hours=optimal_hours_to_add,
            model_name=self.model_name
        )

        print(f"\n[Calendar LLM Search] Starting guided search for {target_calendar}")
        print(f"  Model: {self.model_name}")
        print(f"  Current faulty hours/day: {faulty_hours:.1f}")
        print(f"  Solution criterion: must reach >= {optimal_hours_to_add:.1f}h (ground truth)")
        print(f"  Current (faulty) waiting_time: {faulty_metrics.get('avg_waiting_time', 0):.2f}s")

        current_metrics = faulty_metrics
        previous_attempts = []

        for iteration in range(1, self.max_iterations + 1):
            history.total_iterations = iteration

            # Generate proposal from LLM
            proposal = self._generate_proposal(
                target_calendar=target_calendar,
                current_metrics=current_metrics,
                faulty_hours=faulty_hours,
                previous_attempts=previous_attempts,
                max_hours=8.0  # Fixed max - do NOT leak optimal_hours_to_add
            )

            proposed_hours = proposal.get("proposed_hours", 1.0)
            rationale = proposal.get("rationale", "")

            total_hours = faulty_hours + proposed_hours
            print(f"  Iteration {iteration}: LLM proposes +{proposed_hours:.1f}h ({total_hours:.1f}h total)...", end=" ")

            # Create modifications
            modifications = {
                "parameter_type": "calendar",
                "specific_changes": [{
                    "target": target_calendar,
                    "change": "increase_hours",
                    "value": proposed_hours
                }]
            }

            # Run simulation
            metrics = simulate_fn(faulty_model, modifications)

            cycle_time = metrics.get("avg_cycle_time", float("inf"))
            waiting_time = metrics.get("avg_waiting_time", float("inf"))
            throughput = metrics.get("avg_throughput", 0)

            # Solution criterion: must reach the ground truth hours (not just KPI tolerance)
            is_solution = proposed_hours >= optimal_hours_to_add

            result = CalendarLLMSearchResult(
                iteration=iteration,
                proposed_hours=proposed_hours,
                total_hours=total_hours,
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
                "hours_added": proposed_hours,
                "total_hours": total_hours,
                "waiting_time": waiting_time,
                "cycle_time": cycle_time,
                "throughput": throughput,
                "improved": waiting_time < current_metrics.get("avg_waiting_time", float("inf"))
            })

            status = "FOUND!" if is_solution else ""
            print(f"waiting_time={waiting_time:.2f}s {status}")

            # Record first hit and stop searching
            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} (+{proposed_hours:.1f}h)")
                break

            # Update current metrics for next iteration
            current_metrics = metrics

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {self.max_iterations} iterations")

        return history

    def _generate_proposal(
        self,
        target_calendar: str,
        current_metrics: Dict,
        faulty_hours: float,
        previous_attempts: List[Dict],
        max_hours: float = 8.0
    ) -> Dict:
        """Generate hours-to-add proposal from LLM."""

        # Extract hours already tried
        tried_hours = [round(attempt["hours_added"], 1) for attempt in previous_attempts]
        available_hours = [h for h in [i * 0.5 for i in range(2, int(max_hours * 2) + 1)]
                          if round(h, 1) not in tried_hours]

        system_prompt = f"""You are a process optimization expert analyzing simulation metrics.

Your ONLY task is to propose optimal working hours to ADD for calendar: {target_calendar}

Current situation:
- The resource has {faulty_hours:.1f} working hours per day
- High waiting_time indicates the resource is overloaded and needs more availability

CONSTRAINTS (CRITICAL):
- You can ONLY propose calendar changes (increase_hours)
- You must propose hours to ADD (not total hours)
- Valid range: 1.0 to {max_hours:.1f} hours to add
- DO NOT propose hours already tried: {tried_hours if tried_hours else 'none yet'}

Analysis strategy:
- High waiting_time indicates resource availability bottleneck -> need MORE hours
- High resource utilization (>80%) indicates they're overworked -> need MORE hours
- If previous attempts with more hours improved metrics, continue increasing
- If previous attempts plateaued, you've likely found the optimal amount

Return ONLY a JSON object:
{{
  "proposed_hours": <float 1.0-{max_hours:.1f}, NOT in {tried_hours}>,
  "rationale": "Brief explanation of why this amount"
}}"""

        user_prompt = f"""CURRENT SIMULATION METRICS:
{json.dumps(current_metrics, indent=2)}

Current working hours: {faulty_hours:.1f}h/day
"""
        if previous_attempts:
            user_prompt += "\nPREVIOUS ATTEMPTS (most recent last):\n"
            for i, attempt in enumerate(previous_attempts[-5:], 1):
                improved = "improved" if attempt.get("improved") else "no improvement"
                user_prompt += f"  {i}. +{attempt['hours_added']:.1f}h ({attempt['total_hours']:.1f}h total): waiting_time={attempt['waiting_time']:.2f}s ({improved})\n"
            user_prompt += "\n"

        user_prompt += f"""Based on the metrics and previous attempts, propose hours to add.
Focus on the waiting_time and resource_utilization to identify if more availability is needed.
IMPORTANT: Do NOT propose any of these already-tried values: {tried_hours}
Valid range: 1.0 to {max_hours:.1f} hours

Return ONLY the JSON object."""

        try:
            # Use HypothesisGenerator's _call_llm for provider routing
            response = self._llm._call_llm(system_prompt, user_prompt)

            # Parse JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                proposal = json.loads(response[start_idx:end_idx])
                hours = float(proposal.get("proposed_hours", 2.0))
                hours = max(1.0, min(max_hours, hours))  # Clamp to valid range
                hours = round(hours * 2) / 2  # Round to nearest 0.5

                # If LLM still proposed a duplicate, pick the nearest untried
                if round(hours, 1) in tried_hours and available_hours:
                    nearest = min(available_hours, key=lambda x: abs(x - hours))
                    print(f"\n    [!] LLM proposed duplicate {hours:.1f}h, using {nearest:.1f}h instead")
                    hours = nearest

                return {
                    "proposed_hours": hours,
                    "rationale": proposal.get("rationale", "")
                }
        except Exception as e:
            print(f"\n    [!] LLM parsing error: {e}")

        # Fallback: pick next available hours (not already tried)
        if available_hours:
            # Prefer incrementing from last attempt if possible
            if previous_attempts:
                last_hours = previous_attempts[-1].get("hours_added", 1.0)
                higher_available = [h for h in available_hours if h > last_hours]
                if higher_available:
                    return {"proposed_hours": min(higher_available), "rationale": "Fallback: next higher untried"}
            return {"proposed_hours": min(available_hours), "rationale": "Fallback: lowest untried"}

        # All hours exhausted
        return {"proposed_hours": max_hours, "rationale": "Fallback: all values tried"}

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
