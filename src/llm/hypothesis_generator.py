import json
from typing import Dict, List
from dotenv import load_dotenv
import re

load_dotenv()

from gen_ai_hub.proxy.native.openai import chat


class HypothesisGenerator:
    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    
    def generate(
        self,
        baseline_metrics: Dict,
        previous_attempts: List[Dict] = None,
        target_hints: Dict = None
    ) -> Dict:
        """
        Generate improvement hypothesis using LLM.
        
        Args:
            baseline_metrics: Baseline process metrics
            previous_attempts: List of previous attempts with outcomes
            target_hints: Dict with known target names (resources, activities, calendars)
            
        Returns:
            Hypothesis dict with description, type, changes, and expected impact
        """
        if previous_attempts is None:
            previous_attempts = []
        if target_hints is None:
            target_hints = {}

        strategy_hints = self._build_strategy_hints(baseline_metrics, previous_attempts, target_hints)
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(
            baseline_metrics,
            previous_attempts,
            target_hints,
            strategy_hints
        )
        
        try:
            response = chat.completions.create(
                model_name=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse LLM response and extract JSON
            response_text = response.choices[0].message.content
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    hypothesis = json.loads(json_str)
                else:
                    hypothesis = self._parse_hypothesis_text(response_text)
            except json.JSONDecodeError:
                hypothesis = self._parse_hypothesis_text(response_text)
            
            hypothesis = self._sanitize_hypothesis(hypothesis, strategy_hints, target_hints)

            if self._validate_hypothesis(hypothesis):
                return hypothesis
            else:
                print("Warning: Invalid hypothesis structure, using fallback")
                return self._get_fallback_hypothesis(strategy_hints, target_hints)
                
        except Exception as e:
            print(f"Error generating hypothesis: {e}")
            return self._get_fallback_hypothesis(strategy_hints, target_hints)
    

    def _get_system_prompt(self) -> str:
        return """You are a process improvement specialist analyzing business process metrics.

Your task is to generate actionable, predictive hypotheses to improve process performance.

    Proposal strategy:
    - Propose ONE change only (single specific change entry).
        - Use only this safe intervention palette:
            * role_redesign (shift_responsibility or create_specialist_role)
            * activity_duration_optimization (reduce_duration 5-25%)
            * calendar_adjustment (increase_hours 1-2)
            * resource_allocation (add or increase_count by 1-2)
        - Do NOT propose arrival_rate_adjustment or routing_adjustment.
        - Do NOT propose remove/decrease_count/increase_duration/decrease_hours.
    - Avoid hypotheses that are likely to increase cycle time or waiting time, or decrease throughput.
    - Prefer changes that improve at least two of: cycle time, throughput, waiting time.

For each hypothesis, provide:
- Clear description of the improvement
- Type: resource_allocation, role_redesign, activity_duration_optimization, or calendar_adjustment
- Specific changes to simulation parameters
- Expected impact on metrics

Return a JSON object with the following structure:
{
  "description": "Clear description of the improvement",
    "improvement_type": "resource_allocation|role_redesign|activity_duration_optimization|calendar_adjustment",
  "rationale": "Why this improvement would help",
  "changes": {
        "parameter_type": "resources|role_redesign|activity_durations|calendar",
    "specific_changes": [
      {
                "target": "Resource_1|role_name|ActivityName|calendar_name|gateway_id",
                                "change": "add|increase_count|create_specialist_role|shift_responsibility|reduce_duration|increase_hours",
        "value": numeric_value
      }
    ]
  },
  "expected_impact": {
    "cycle_time": "percentage_change",
    "throughput": "percentage_change",
    "cost": "percentage_change"
  }
}"""
    

    def _get_user_prompt(
        self,
        baseline_metrics: Dict,
        previous_attempts: List[Dict],
        target_hints: Dict,
        strategy_hints: Dict
    ) -> str:
        prompt = f"""Analyze the current process metrics and generate ONE new improvement hypothesis:

BASELINE METRICS:
{json.dumps(baseline_metrics, indent=2)}

"""
        if target_hints:
            resources = ", ".join(target_hints.get("resources", []))
            activities = ", ".join(target_hints.get("activities", []))
            calendars = ", ".join(target_hints.get("calendars", []))
            gateways = ", ".join(target_hints.get("gateways", []))
            role_redesign_hints = target_hints.get("role_redesign_hints", "")
            prompt += "KNOWN TARGETS (use these exact names if you reference them):\n"
            if resources:
                prompt += f"- Resources: {resources}\n"
            if activities:
                prompt += f"- Activities: {activities}\n"
            if calendars:
                prompt += f"- Calendars: {calendars}\n"
            if gateways:
                prompt += f"- Gateways: {gateways}\n"
            if role_redesign_hints:
                prompt += f"- Role redesign guidance: {role_redesign_hints}\n"
            prompt += "\n"

        allowed_types = strategy_hints.get("allowed_parameter_types", [])
        disallowed_types = strategy_hints.get("disallowed_parameter_types", [])
        high_priority_targets = strategy_hints.get("high_priority_targets", [])
        prompt += "ITERATION STRATEGY CONSTRAINTS (must follow):\n"
        if allowed_types:
            prompt += f"- Allowed parameter_type values this iteration: {', '.join(allowed_types)}\n"
        if disallowed_types:
            prompt += f"- Disallowed parameter_type values: {', '.join(disallowed_types)}\n"
        if high_priority_targets:
            prompt += f"- Prefer these targets first: {', '.join(high_priority_targets[:6])}\n"
        prompt += (
            "- Throughput safety: expected throughput impact must be >= 0% (never propose throughput decrease).\n"
            "- Propose exactly one specific_changes item.\n\n"
        )

        if previous_attempts:
            prompt += "PREVIOUS ATTEMPTS (most recent last):\n"
            for attempt in previous_attempts[-3:]:
                hypothesis = attempt.get("hypothesis", {})
                outcome = attempt.get("improvement", {})
                unified = attempt.get("unified_metrics", {})
                des = attempt.get("des_metrics", {})
                abs_metrics = attempt.get("abs_metrics", {})
                prompt += f"- {hypothesis.get('description', 'Unknown')}\n"
                prompt += f"  Changes: {json.dumps(hypothesis.get('changes', {}))}\n"
                prompt += f"  Outcome: {outcome.get('summary', 'Unknown')}\n"
                if unified:
                    prompt += f"  Unified metrics: {json.dumps(unified)}\n"
                if des and abs_metrics:
                    prompt += f"  DES metrics: {json.dumps(des)}\n"
                    prompt += f"  ABS metrics: {json.dumps(abs_metrics)}\n"
            prompt += "\nBuild incrementally on the most recent attempt if it was promising; otherwise change direction.\n"
        else:
            prompt += "This is the first improvement attempt.\n"
        
        prompt += """
Focus on:
1. Resource bottlenecks (look for utilization > 0.8)
2. High waiting times (> 40% of cycle time)
3. Long cycle times (> 12000 seconds)

    Safety rules:
    - Use meaningful deltas (5-25% duration reduction, +1 to +2 resources, +1 to +2 hours calendar).
    - Avoid any change likely to worsen cycle time or waiting time.
    - Prefer improvements that help at least two of: cycle time, throughput, waiting time.
    - Prefer role_redesign over pure resource addition when overloaded and underutilized roles coexist.
    - If prior attempts fail repeatedly, switch intervention type among allowed types only.
    - Do not use placeholders like "Activity_with_highest_waiting_time"; choose from the known targets.

Return ONLY the JSON object, no additional text."""
        
        return prompt
    

    def _parse_hypothesis_text(self, text: str) -> Dict:
        """Parse hypothesis from plain text if JSON parsing fails."""
        hypothesis = {
            "description": "Add one additional Resource_1 to address high utilization",
            "improvement_type": "resource_allocation",
            "rationale": "Resource_1 shows high utilization indicating a bottleneck",
            "changes": {
                "parameter_type": "resources",
                "specific_changes": [{"target": "Resource_1", "change": "add", "value": 1}]
            },
            "expected_impact": {"cycle_time": "-8", "throughput": "6", "cost": "4"}
        }
        return hypothesis
    

    def _validate_hypothesis(self, hypothesis: Dict) -> bool:
        """Validate hypothesis structure."""
        required_keys = {"description", "improvement_type", "changes", "expected_impact"}
        if not required_keys.issubset(hypothesis.keys()):
            return False

        changes = hypothesis.get("changes", {})
        parameter_type = changes.get("parameter_type")
        specific_changes = changes.get("specific_changes", [])
        if parameter_type not in {"resources", "role_redesign", "activity_durations", "calendar"}:
            return False
        if not isinstance(specific_changes, list) or len(specific_changes) != 1:
            return False

        expected = hypothesis.get("expected_impact", {})
        throughput = self._extract_pct(expected.get("throughput", "0"))
        if throughput < 0:
            return False
        return True
    

    def _get_fallback_hypothesis(self, strategy_hints: Dict = None, target_hints: Dict = None) -> Dict:
        """Get fallback hypothesis when LLM call fails."""
        strategy_hints = strategy_hints or {}
        target_hints = target_hints or {}
        allowed = strategy_hints.get("allowed_parameter_types", ["resources"])
        primary_type = allowed[0] if allowed else "resources"

        target, change, value, improvement_type = self._build_safe_change(primary_type, target_hints, strategy_hints)
        return {
            "description": f"Apply a targeted {primary_type} intervention on {target} to reduce bottlenecks and preserve throughput.",
            "improvement_type": improvement_type,
            "rationale": "This is a safe, high-likelihood intervention constrained to validated target names and conservative positive deltas.",
            "changes": {
                "parameter_type": primary_type,
                "specific_changes": [{"target": target, "change": change, "value": value}]
            },
            "expected_impact": {"cycle_time": "-8%", "throughput": "+3%", "cost": "+2%"}
        }

    def _build_strategy_hints(self, baseline_metrics: Dict, previous_attempts: List[Dict], target_hints: Dict) -> Dict:
        """Build strict, high-likelihood intervention constraints for the current iteration."""
        baseline_metrics = baseline_metrics or {}
        target_hints = target_hints or {}

        cycle_time = float(baseline_metrics.get("avg_cycle_time", 0) or 0)
        waiting_time = float(baseline_metrics.get("avg_waiting_time", 0) or 0)
        waiting_ratio = (waiting_time / cycle_time) if cycle_time > 0 else 0

        util = baseline_metrics.get("resource_utilization", {}) or {}
        sorted_utils = sorted(((str(k), float(v)) for k, v in util.items()), key=lambda x: x[1], reverse=True)
        overloaded = [r for r, u in sorted_utils if u >= 0.80][:3]
        underutilized = [r for r, u in sorted_utils if u <= 0.25][:5]

        resources = target_hints.get("resources", [])
        resource_norms = {str(r).replace("_profile", "") for r in resources}
        mappable_overloaded = [r for r in overloaded if r in resource_norms]
        mappable_underutilized = [r for r in underutilized if r in resource_norms]

        allowed = []
        if mappable_overloaded and mappable_underutilized:
            allowed.append("role_redesign")
        if waiting_ratio >= 0.35:
            allowed.extend(["activity_durations", "calendar"])
        if overloaded:
            allowed.append("resources")

        if not allowed:
            allowed = ["activity_durations", "resources"]

        # Remove parameter types that failed repeatedly in latest attempts
        failed_recent = []
        for attempt in (previous_attempts or [])[-3:]:
            improvement = attempt.get("improvement", {})
            if improvement.get("is_improvement") is False:
                param_type = (
                    attempt.get("hypothesis", {})
                    .get("changes", {})
                    .get("parameter_type")
                )
                if param_type:
                    failed_recent.append(param_type)

        if len(failed_recent) >= 2:
            repeated = max(set(failed_recent), key=failed_recent.count)
            if failed_recent.count(repeated) >= 2 and repeated in allowed and len(allowed) > 1:
                allowed = [t for t in allowed if t != repeated] + [repeated]

        high_priority_targets = mappable_overloaded + mappable_underutilized
        if not high_priority_targets:
            high_priority_targets = overloaded + underutilized

        return {
            "allowed_parameter_types": list(dict.fromkeys(allowed)),
            "disallowed_parameter_types": ["arrival_rate", "gateway_probabilities"],
            "high_priority_targets": high_priority_targets,
            "overloaded_targets": overloaded,
            "underutilized_targets": underutilized,
            "mappable_overloaded_targets": mappable_overloaded,
            "mappable_underutilized_targets": mappable_underutilized,
            "waiting_ratio": waiting_ratio
        }

    def _sanitize_hypothesis(self, hypothesis: Dict, strategy_hints: Dict, target_hints: Dict) -> Dict:
        """Normalize hypothesis to a strict, safe and executable format."""
        strategy_hints = strategy_hints or {}
        target_hints = target_hints or {}
        hypothesis = hypothesis or {}

        allowed = strategy_hints.get("allowed_parameter_types", ["resources"])
        changes = hypothesis.get("changes", {}) if isinstance(hypothesis.get("changes", {}), dict) else {}
        parameter_type = changes.get("parameter_type")
        if parameter_type not in allowed:
            parameter_type = allowed[0]

        specific_changes = changes.get("specific_changes", [])
        if not isinstance(specific_changes, list) or not specific_changes:
            specific_changes = [{}]
        specific = specific_changes[0] if isinstance(specific_changes[0], dict) else {}

        target, change, value, improvement_type = self._build_safe_change(
            parameter_type,
            target_hints,
            strategy_hints,
            preferred_target=specific.get("target"),
            preferred_change=specific.get("change"),
            preferred_value=specific.get("value")
        )

        expected = hypothesis.get("expected_impact", {})
        if not isinstance(expected, dict):
            expected = {}
        throughput = self._extract_pct(expected.get("throughput", "0"))
        if throughput < 0:
            expected["throughput"] = "+2%"
        expected.setdefault("cycle_time", "-5%")
        expected.setdefault("cost", "+2%")

        description, rationale = self._build_consistent_narrative(
            parameter_type,
            target,
            change,
            value,
            fallback_description=hypothesis.get("description"),
            fallback_rationale=hypothesis.get("rationale")
        )

        return {
            "description": description,
            "improvement_type": improvement_type,
            "rationale": rationale,
            "changes": {
                "parameter_type": parameter_type,
                "specific_changes": [
                    {
                        "target": target,
                        "change": change,
                        "value": value
                    }
                ]
            },
            "expected_impact": expected
        }

    def _build_consistent_narrative(
        self,
        parameter_type: str,
        target: str,
        change: str,
        value,
        fallback_description: str = None,
        fallback_rationale: str = None
    ):
        """Build narrative text guaranteed to match executable change payload."""
        fallback_description = (fallback_description or "").strip()
        fallback_rationale = (fallback_rationale or "").strip()

        if parameter_type == "role_redesign":
            if "→" in str(target):
                src, dst = [part.strip() for part in str(target).split("→", 1)]
            else:
                src, dst = str(target), str(target)
            description = (
                f"Shift responsibilities from {src} to {dst} to rebalance workload and reduce bottlenecks."
            )
            rationale = (
                f"Executed change is {change} from {src} to {dst} (value={value}), targeting overloaded-to-underutilized redistribution."
            )
            return description, rationale

        if parameter_type == "activity_durations":
            description = f"Reduce execution duration for {target} by {value}% to improve flow efficiency."
            rationale = f"Executed change is {change} on activity {target} with value={value}."
            return description, rationale

        if parameter_type == "calendar":
            description = f"Adjust calendar {target} by {value} hour(s) to improve process capacity."
            rationale = f"Executed change is {change} on calendar {target} with value={value}."
            return description, rationale

        # resources/default
        description = f"Increase capacity for {target} using {change} (value={value}) to reduce resource pressure."
        rationale = f"Executed change is {change} on resource target {target} with value={value}."
        return description, rationale

    def _build_safe_change(
        self,
        parameter_type: str,
        target_hints: Dict,
        strategy_hints: Dict,
        preferred_target=None,
        preferred_change=None,
        preferred_value=None
    ):
        resources = target_hints.get("resources", [])
        activities = target_hints.get("activities", [])
        calendars = target_hints.get("calendars", [])
        top_targets = strategy_hints.get("high_priority_targets", [])
        overloaded_targets = strategy_hints.get("mappable_overloaded_targets", [])
        underutilized_targets = strategy_hints.get("mappable_underutilized_targets", [])

        def resolve_resource_name(name: str):
            if not name:
                return None
            if name in resources:
                return name
            normalized = str(name).replace("_profile", "")
            for candidate in resources:
                candidate_norm = str(candidate).replace("_profile", "")
                if candidate_norm == normalized:
                    return candidate
            for candidate in resources:
                if str(candidate).startswith(normalized):
                    return candidate
            return None

        if parameter_type == "role_redesign":
            from_role = None
            to_role = None
            for role in overloaded_targets:
                resolved = resolve_resource_name(role)
                if resolved and from_role is None:
                    from_role = resolved
            for role in underutilized_targets:
                resolved = resolve_resource_name(role)
                if resolved and resolved != from_role:
                    to_role = resolved
                    break
            if not from_role:
                from_role = resolve_resource_name("NOT_SET")
            if not from_role and resources:
                from_role = resources[0]
            if not to_role and len(resources) > 1:
                # Prefer a non-artificial target for receiving work
                non_artificial = [r for r in resources if not str(r).startswith("artificial_") and r != from_role]
                to_role = non_artificial[-1] if non_artificial else resources[-1]
            if not from_role:
                from_role = "NOT_SET"
            if not to_role:
                to_role = from_role
            target = f"{from_role} → {to_role}"
            change = preferred_change if preferred_change in {"shift_responsibility", "create_specialist_role"} else "shift_responsibility"
            value = 1
            return target, change, value, "role_redesign"

        if parameter_type == "activity_durations":
            target = preferred_target if preferred_target in activities else (activities[0] if activities else "Task_1")
            change = "reduce_duration"
            try:
                value = int(float(preferred_value))
            except Exception:
                value = 15
            value = max(5, min(25, value))
            return target, change, value, "activity_duration_optimization"

        if parameter_type == "calendar":
            target = preferred_target if preferred_target in calendars else (calendars[0] if calendars else "Default")
            change = "increase_hours"
            try:
                value = int(float(preferred_value))
            except Exception:
                value = 1
            value = max(1, min(2, value))
            return target, change, value, "calendar_adjustment"

        # Default: resources (safe positive capacity)
        target = preferred_target if preferred_target in resources else None
        if not target:
            for role in top_targets:
                if role in resources:
                    target = role
                    break
        if not target:
            target = resources[0] if resources else "Resource_1"

        change = preferred_change if preferred_change in {"add", "increase_count"} else "add"
        try:
            value = int(float(preferred_value))
        except Exception:
            value = 1
        value = max(1, min(2, value))
        return target, change, value, "resource_allocation"

    @staticmethod
    def _extract_pct(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        return 0.0
