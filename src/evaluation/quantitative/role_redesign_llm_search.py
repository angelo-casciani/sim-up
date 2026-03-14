"""
Role Redesign LLM-Guided Search for Quantitative Evaluation.

This module uses an LLM to propose role redesign changes (shift_responsibility)
and iteratively refines based on simulation feedback.

Solution Detection: Requires a MINIMUM IMPROVEMENT over the faulty state.
Uses ABS (Agent-Based Simulation) cycle_time as the primary KPI for role redesign,
since ABS explicitly models agent handover and routing behavior.

Fair Benchmarking: The simulate_fn provided by evaluation_runner uses a
simulation cache with fixed random seeds. This ensures that if both LLM
Search and Grid Search test the same configuration, they receive identical
simulation results, eliminating random variance as a confounding factor.
"""

import json
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field

from llm import HypothesisGenerator


# Fixed shift fraction options for search space - finer granularity for meaningful differentiation
# Starting from 5% in 5% increments up to 100% gives 20 options
SHIFT_FRACTION_OPTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

# Minimum required improvement over faulty state to count as "solution found"
# This accounts for simulation variance - small improvements could be noise
# Set high enough that multiple iterations are needed, giving LLM time to learn patterns
MIN_IMPROVEMENT_PCT = 7.0  # Require at least 7% improvement over faulty ABS cycle time


@dataclass
class RoleRedesignLLMResult:
    """Result of a single role redesign LLM search iteration."""
    iteration: int
    from_role: str
    to_role: str
    shift_fraction: float
    metrics: Dict
    is_solution: bool
    cycle_time: float  # ABS cycle_time (primary KPI for role redesign)
    waiting_time: float
    throughput: float
    llm_rationale: str


@dataclass
class RoleRedesignLLMHistory:
    """Complete history of role redesign LLM-guided search experiment."""
    from_role: str  # Role to shift work FROM (overloaded)
    to_role: str    # Role to shift work TO (underloaded)
    target_cycle_time: float  # Target ABS cycle time to improve upon
    model_name: str
    results: List[RoleRedesignLLMResult] = field(default_factory=list)
    solution_found_at: Optional[int] = None
    total_iterations: int = 0

    def to_dict(self) -> Dict:
        return {
            "from_role": self.from_role,
            "to_role": self.to_role,
            "target_cycle_time": self.target_cycle_time,
            "model_name": self.model_name,
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
                    "llm_rationale": r.llm_rationale,
                    "resource_utilization": r.metrics.get("resource_utilization", {})
                }
                for r in self.results
            ]
        }


class RoleRedesignLLMSearch:
    """
    LLM-guided search for role redesign optimization.

    Uses an LLM to intelligently propose shift percentages based on
    simulation feedback, aiming to find a configuration that achieves
    minimum improvement over faulty state faster than grid search.
    Uses ABS cycle_time as the primary KPI.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_iterations: int = 10,
        tolerance_pct: float = 1.0,
        min_improvement_pct: float = None
    ):
        """
        Initialize LLM-guided role redesign search.

        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            max_iterations: Maximum search iterations (default: 10, capped by available options)
            tolerance_pct: DEPRECATED - kept for API compatibility
            min_improvement_pct: Minimum % improvement over faulty to count as solution
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = min(max_iterations, len(SHIFT_FRACTION_OPTIONS))
        self.tolerance_pct = tolerance_pct  # Kept for compatibility
        self.min_improvement_pct = min_improvement_pct if min_improvement_pct is not None else MIN_IMPROVEMENT_PCT

        self._llm = HypothesisGenerator(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def search(
        self,
        faulty_model: Dict,
        from_role: str,
        to_role: str,
        optimal_metrics: Dict,
        faulty_metrics: Dict,
        simulate_fn: Callable[[Dict, Dict], Dict],
        work_dir: str
    ) -> RoleRedesignLLMHistory:
        """
        Run LLM-guided search to find optimal shift percentage.

        Args:
            faulty_model: Model with injected fault (workload imbalance)
            from_role: Role to shift work FROM (overloaded)
            to_role: Role to shift work TO (underloaded)
            optimal_metrics: Metrics from optimal model (for reference only)
            faulty_metrics: Current metrics from faulty model - TARGET TO IMPROVE
            simulate_fn: Function(model, modifications) -> metrics
            work_dir: Working directory for simulation

        Returns:
            RoleRedesignLLMHistory with all iterations and solution info
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

        history = RoleRedesignLLMHistory(
            from_role=from_role,
            to_role=to_role,
            target_cycle_time=faulty_abs_cycle,  # Target is to improve over faulty
            model_name=self.model_name
        )

        print(f"\n[Role Redesign LLM Search] Starting guided search")
        print(f"  Model: {self.model_name}")
        print(f"  Role pair: {from_role} → {to_role}")
        print(f"  Faulty ABS cycle_time: {faulty_abs_cycle:.2f}s - MUST IMPROVE BY ≥{self.min_improvement_pct}%")
        print(f"  Optimal ABS cycle_time: {optimal_abs_cycle:.2f}s (for reference)")
        print(f"  Shift options: {[f'{x:.0%}' for x in SHIFT_FRACTION_OPTIONS]}")

        current_metrics = faulty_metrics
        previous_attempts = []
        tried_fractions = []
        best_improvement = 0.0
        best_shift = None

        for iteration in range(1, self.max_iterations + 1):
            history.total_iterations = iteration

            # Generate proposal from LLM
            proposal = self._generate_proposal(
                from_role=from_role,
                to_role=to_role,
                current_metrics=current_metrics,
                previous_attempts=previous_attempts,
                tried_fractions=tried_fractions,
                faulty_abs_cycle=faulty_abs_cycle,
                optimal_abs_cycle=optimal_abs_cycle
            )

            shift_fraction = proposal.get("shift_fraction", 0.2)
            rationale = proposal.get("rationale", "")

            print(f"  Iteration {iteration}: LLM proposes {from_role} → {to_role} ({shift_fraction:.0%})...", end=" ")

            # Create modifications
            modifications = {
                "parameter_type": "role_redesign",
                "specific_changes": [{
                    "target": f"{from_role} → {to_role}",
                    "change": "shift_responsibility",
                    "value": shift_fraction
                }]
            }

            # Run simulation
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
                best_shift = shift_fraction

            result = RoleRedesignLLMResult(
                iteration=iteration,
                from_role=from_role,
                to_role=to_role,
                shift_fraction=shift_fraction,
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
                "shift_fraction": shift_fraction,
                "cycle_time": cycle_time,
                "waiting_time": waiting_time,
                "improvement_pct": improvement_pct
            })
            tried_fractions.append(shift_fraction)

            status = "FOUND!" if is_solution else ""
            print(f"ABS cycle_time={cycle_time:.2f}s ({improvement_pct:+.1f}% vs faulty) {status}")

            if is_solution and history.solution_found_at is None:
                history.solution_found_at = iteration
                print(f"  [OK] Solution found at iteration {iteration} ({shift_fraction:.0%} shift, {improvement_pct:.1f}% improvement)")
                break

            current_metrics = metrics

        if history.solution_found_at is None:
            print(f"  [X] No solution found in {self.max_iterations} iterations")
            if best_shift is not None:
                print(f"      Best improvement: {best_improvement:.1f}% at {best_shift:.0%} shift (required: ≥{self.min_improvement_pct}%)")
            else:
                print(f"      No positive improvements found (required: ≥{self.min_improvement_pct}%)")

        return history

    def _generate_proposal(
        self,
        from_role: str,
        to_role: str,
        current_metrics: Dict,
        previous_attempts: List[Dict],
        tried_fractions: List[float],
        faulty_abs_cycle: float,
        optimal_abs_cycle: float
    ) -> Dict:
        """Generate shift fraction proposal from LLM."""

        available_fractions = [f for f in SHIFT_FRACTION_OPTIONS if f not in tried_fractions]

        # Sort previous attempts by shift fraction for pattern recognition
        sorted_attempts = sorted(previous_attempts, key=lambda x: x["shift_fraction"])

        system_prompt = f"""You are a process optimization expert analyzing workload distribution.

TASK: Find the optimal shift fraction to rebalance workload between roles.

CONTEXT:
- Work should be shifted FROM '{from_role}' (overloaded) TO '{to_role}' (underloaded)
- The role pair is already determined - you only need to find the right PERCENTAGE to shift
- Faulty ABS cycle_time: {faulty_abs_cycle:.2f}s (BAD - too high due to imbalance)
- You need to achieve ≥{self.min_improvement_pct}% IMPROVEMENT over faulty cycle_time
- Target cycle_time: ≤{faulty_abs_cycle * (1 - self.min_improvement_pct/100):.2f}s (to achieve {self.min_improvement_pct}% improvement)

VALID SHIFT OPTIONS: {[f'{x:.0%}' for x in SHIFT_FRACTION_OPTIONS]}
ALREADY TRIED: {[f'{x:.0%}' for x in tried_fractions] if tried_fractions else 'none yet'}
AVAILABLE OPTIONS: {[f'{x:.0%}' for x in available_fractions] if available_fractions else 'NONE LEFT'}

ANALYSIS STRATEGY:
- Look at the pattern: shifting work redistributes load, which should reduce cycle_time
- If a lower shift didn't improve enough, try a HIGHER shift
- If a higher shift made things worse (over-correction), try a MODERATE shift
- Look for the shift % that gives the BEST improvement over faulty

Return ONLY a JSON object:
{{
  "shift_fraction": <float from {available_fractions}>,
  "rationale": "Brief explanation based on the pattern"
}}"""

        # Get current ABS cycle_time
        current_cycle_time = current_metrics.get("abs_cycle_time", 0)
        if current_cycle_time == 0:
            current_cycle_time = current_metrics.get("avg_cycle_time", 0)

        current_improvement = ((faulty_abs_cycle - current_cycle_time) / faulty_abs_cycle * 100) if faulty_abs_cycle > 0 else 0

        user_prompt = f"""CURRENT METRICS:
- ABS Cycle time: {current_cycle_time:.2f}s
- Current improvement: {current_improvement:+.1f}% vs faulty
- Required improvement: ≥{self.min_improvement_pct}%

"""
        if sorted_attempts:
            user_prompt += "PREVIOUS ATTEMPTS (sorted by shift %):\n"
            user_prompt += f"{'Shift %':<10} {'Cycle Time':<15} {'Improvement':<15}\n"
            user_prompt += "-" * 40 + "\n"
            for attempt in sorted_attempts:
                imp = attempt.get("improvement_pct", 0)
                user_prompt += f"{attempt['shift_fraction']:.0%}{'':<6} {attempt['cycle_time']:.2f}s{'':<8} {imp:+.1f}%\n"
            user_prompt += "\n"

            # Highlight best result (highest improvement)
            best = max(sorted_attempts, key=lambda x: x.get("improvement_pct", 0))
            user_prompt += f"BEST SO FAR: {best['shift_fraction']:.0%} with {best.get('improvement_pct', 0):+.1f}% improvement\n\n"

        user_prompt += f"""Based on the pattern, propose the next shift fraction to try.
AVAILABLE OPTIONS: {[f'{x:.0%}' for x in available_fractions]}

Return ONLY the JSON object."""

        try:
            response = self._llm._call_llm(system_prompt, user_prompt)

            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                proposal = json.loads(response[start_idx:end_idx])
                shift = float(proposal.get("shift_fraction", 0.2))

                # Snap to nearest valid option
                shift = min(SHIFT_FRACTION_OPTIONS, key=lambda x: abs(x - shift))

                # If already tried, pick nearest untried
                if shift in tried_fractions and available_fractions:
                    shift = min(available_fractions, key=lambda x: abs(x - shift))

                return {
                    "shift_fraction": shift,
                    "rationale": proposal.get("rationale", "")
                }
        except Exception as e:
            print(f"\n    [!] LLM parsing error: {e}")

        # Fallback: pick next untried fraction
        if available_fractions:
            return {
                "shift_fraction": available_fractions[0],
                "rationale": "Fallback: first untried option"
            }

        return {
            "shift_fraction": SHIFT_FRACTION_OPTIONS[-1],
            "rationale": "Fallback: all tried, returning max"
        }
