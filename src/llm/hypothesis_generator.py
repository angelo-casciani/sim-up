import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import re

load_dotenv()

from gen_ai_hub.proxy.native.openai import chat

# Models that use Amazon Bedrock Converse API (Anthropic Claude models)
AMAZON_BEDROCK_MODELS = {'anthropic--claude-4.5-opus', 'anthropic--claude-4.5-sonnet', 'anthropic--claude-4.5-haiku',
                         'anthropic--claude-4-opus', 'anthropic--claude-4-sonnet'}

# OpenAI reasoning models (use max_completion_tokens, no temperature)
OPENAI_REASONING_MODELS = {'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o3', 'o3-mini', 'o1', 'o1-mini', 'o1-preview'}


class HypothesisGenerator:
    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _is_amazon_bedrock_model(self) -> bool:
        """Check if the model uses Amazon Bedrock Converse API."""
        model_lower = self.model_name.lower()
        return any(m in model_lower for m in AMAZON_BEDROCK_MODELS)

    def _is_reasoning_model(self) -> bool:
        """Check if the model is an OpenAI reasoning model (o-series, gpt-5 series)."""
        return self.model_name in OPENAI_REASONING_MODELS

    def _call_openai_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible chat API (works for GPT, Mistral models)."""
        response = chat.completions.create(
            model_name=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _call_openai_reasoning(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI reasoning model API (o-series, gpt-5 series - no temperature, use max_completion_tokens)."""
        response = chat.completions.create(
            model_name=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={"max_completion_tokens": self.max_tokens}
        )
        # Debug: check if content is None
        content = response.choices[0].message.content
        if content is None:
            # Try to get more info about the response
            print(f"[DEBUG] Response object: {response}")
            print(f"[DEBUG] Message object: {response.choices[0].message}")
            # Some reasoning models put content in 'reasoning_content' or similar
            if hasattr(response.choices[0].message, 'reasoning_content'):
                content = response.choices[0].message.reasoning_content
        return content or ""

    def _call_amazon_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """Call Amazon Bedrock Converse API for Anthropic models."""
        from gen_ai_hub.proxy.native.amazon.clients import Session

        bedrock = Session().client(model_name=self.model_name)
        conversation = [
            {
                "role": "user",
                "content": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ]
        response = bedrock.converse(
            messages=conversation,
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        return response["output"]["message"]["content"][0]["text"]

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Route LLM call to the appropriate provider based on model name."""
        if self._is_amazon_bedrock_model():
            print(f"[HypothesisGenerator] Using Amazon Bedrock Converse API for {self.model_name}")
            return self._call_amazon_bedrock(system_prompt, user_prompt)
        elif self._is_reasoning_model():
            print(f"[HypothesisGenerator] Using OpenAI Reasoning API for {self.model_name}")
            return self._call_openai_reasoning(system_prompt, user_prompt)
        else:
            print(f"[HypothesisGenerator] Using OpenAI Chat API for {self.model_name}")
            return self._call_openai_chat(system_prompt, user_prompt)
    
    
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
            # Route to appropriate provider based on model
            response_text = self._call_llm(system_prompt, user_prompt)

            # Parse LLM response and extract JSON
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

Validation oracle by change type (CRITICAL — use this to set realistic expected_impact):
- shift_responsibility (parameter_type=shift_responsibility): validated by ABS (AgentSimulator) ONLY.
  ABS explicitly models agent handover and routing behaviour. DES only checks throughput does not hard-fail.
  Main KPIs to improve: cycle_time (primary). ABS waiting_time is always 0 by design.
- resource_allocation (parameter_type=resources): validated by DES (Prosimos) ONLY.
  ABS has no queue model and cannot detect capacity improvements. Do NOT expect ABS to confirm.
  Main KPIs to improve: waiting_time (primary), cycle_time, throughput.
- calendar_adjustment (parameter_type=calendar): validated by DES.
  Extends working hours for resources, reducing waiting times and improving throughput.
  Main KPIs to improve: waiting_time (primary), cycle_time, throughput.

    ROLE SEMANTICS (CRITICAL for shift_responsibility):
    - Each role/resource profile can ONLY perform specific activities based on their domain expertise.
    - When proposing shift_responsibility (shift_responsibility), you MUST consider role compatibility:
      * Appraisers: Can ONLY do appraisal-related tasks (property valuation, assessment). Cannot be replaced by Applicants.
      * Loan Officers: Handle loan decisions, approvals, customer interactions related to loans.
      * Clerks: Handle administrative tasks, document processing, data entry.
      * AML Investigators: Handle anti-money laundering checks, compliance verification.
      * Senior Officers: Handle escalations, final approvals, complex decisions.
      * Applicants: External actors who submit applications, provide documents. They are NOT staff and CANNOT perform internal tasks.
    - NEVER propose shifting tasks FROM a specialist company role (appraiser, officer) to a customer/applicant.
    - ONLY shift responsibilities between roles that perform SIMILAR or COMPATIBLE activities within the organization.
    - If role_task_mapping is provided, use it to verify compatibility before proposing shifts.

    Proposal strategy:
    - Propose ONE change only (single specific change entry).
        - Use only this safe intervention palette (STRICT PRIORITY ORDER):
            1. shift_responsibility (shift_responsibility or create_specialist_role) — **MANDATORY FIRST CHOICE** when overloaded + underutilized roles coexist
              This is the LOWEST COST option: redistributes existing workforce without hiring.
              IMPORTANT: Only shift between roles with overlapping or compatible task assignments.
              Example valid shifts: clerk -> clerk, loan officer -> Senior Officer (similar domain)
              Example INVALID shifts: appraiser -> applicant (incompatible - Applicant cannot appraise)
            2. resource_allocation (add or increase_count) — SECOND CHOICE: add capacity to bottleneck roles
              - For utilization >= 95%: add 4-6 resources (critical bottleneck)
              - For utilization >= 90%: add 3-5 resources (severe bottleneck)
              - For utilization >= 80%: add 2-4 resources (high bottleneck)
              - Otherwise: add 1-2 resources
            3. calendar_adjustment (increase_hours) — LAST RESORT: extend working hours for overloaded resources
              - value = hours to add (1-4 hours typical)
              - target = resource name or "all" for global change
        - Do NOT propose arrival_rate_adjustment, routing_adjustment, or activity_duration_optimization.
        - Do NOT propose remove/decrease_count/increase_duration/decrease_hours.
    - Avoid hypotheses that are likely to increase cycle time or decrease throughput.
    - Prefer changes that improve at least two KPIs among: cycle_time, throughput, waiting_time.

Canonical taxonomy mapping (must be respected exactly):
- improvement_type=shift_responsibility       <-> changes.parameter_type=shift_responsibility
- improvement_type=resource_allocation  <-> changes.parameter_type=resources
- improvement_type=calendar_adjustment  <-> changes.parameter_type=calendar

For each hypothesis, provide:
- Clear description of the improvement
- Type: shift_responsibility, resource_allocation, or calendar_adjustment
- Specific changes to simulation parameters
- Expected impact on metrics

Strict output rules:
- Return ONLY valid JSON object.
- Include exactly one specific_changes entry.
- Keep description/rationale concise and consistent with the exact executed target/change/value.
- expected_impact must include: cycle_time, throughput, waiting_time.
  (cost is not tracked; omit it)
- throughput must be >= 0%.

Return a JSON object with the following structure:
{
  "description": "Clear description of the improvement",
    "improvement_type": "shift_responsibility|resource_allocation|calendar_adjustment",
  "rationale": "Why this improvement would help",
  "changes": {
        "parameter_type": "shift_responsibility|resources|calendar",
    "specific_changes": [
      {
                "target": "role_name|Resource_1|all",
                                "change": "shift_responsibility|create_specialist_role|add|increase_count|increase_hours",
        "value": numeric_value
      }
    ]
  },
  "expected_impact": {
    "cycle_time": "percentage_change",
    "throughput": "percentage_change",
        "waiting_time": "percentage_change"
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
            shift_responsibility_hints = target_hints.get("shift_responsibility_hints", "")
            prompt += "KNOWN TARGETS (use these exact names if you reference them):\n"
            if resources:
                prompt += f"- Resources: {resources}\n"
            if activities:
                prompt += f"- Activities: {activities}\n"
            if calendars:
                prompt += f"- Calendars: {calendars}\n"
            if gateways:
                prompt += f"- Gateways: {gateways}\n"
            if shift_responsibility_hints:
                prompt += f"- Shift responsibility guidance: {shift_responsibility_hints}\n"
            # Include role-task mapping if available
            role_task_mapping = target_hints.get("role_task_mapping", {})
            if role_task_mapping:
                prompt += "- Role-Task Mapping (each role can ONLY perform these tasks):\n"
                for role, tasks in role_task_mapping.items():
                    tasks_str = ", ".join(tasks[:5])  # Limit to first 5 tasks for brevity
                    if len(tasks) > 5:
                        tasks_str += f", ... (+{len(tasks)-5} more)"
                    prompt += f"    * {role}: {tasks_str}\n"
                prompt += "  IMPORTANT: Only shift responsibilities between roles with overlapping tasks!\n"
            prompt += "\n"

        allowed_types = strategy_hints.get("allowed_parameter_types", [])
        disallowed_types = strategy_hints.get("disallowed_parameter_types", [])
        high_priority_targets = strategy_hints.get("high_priority_targets", [])
        _oracle_map = {
            "shift_responsibility": "ABS primary (DES throughput safety)",
            "resources": "DES only",
        }
        prompt += "ITERATION STRATEGY CONSTRAINTS (must follow):\n"
        if allowed_types:
            prompt += f"- Allowed parameter_type values this iteration: {', '.join(allowed_types)}\n"
            # Enforce shift_responsibility priority when it's first in the list
            if allowed_types[0] == "shift_responsibility":
                prompt += (
                    "- **MANDATORY**: You MUST use shift_responsibility as your parameter_type. "
                    "Shift responsibility is the LOWEST COST option for the organization (no hiring needed). "
                    "Only use resource_allocation or calendar_adjustment if shift_responsibility has already been tried and failed.\n"
                )
            oracle_notes = [f"{t} -> {_oracle_map.get(t, 'OR')}" for t in allowed_types]
            prompt += f"- Validation oracle for each allowed type: {'; '.join(oracle_notes)}\n"
        if disallowed_types:
            prompt += f"- Disallowed parameter_type values: {', '.join(disallowed_types)}\n"
        recently_tried = strategy_hints.get("recently_tried", [])
        if recently_tried:
            prompt += f"- Do NOT propose these (type:target) combinations again — already attempted in a previous iteration: {', '.join(recently_tried)}\n"
            prompt += "  Each hypothesis is tested independently against the same fixed baseline; repeating an already-tried combination yields no new information.\n"
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
                des = attempt.get("des_metrics", {})
                abs_metrics = attempt.get("abs_metrics", {})
                prompt += f"- {hypothesis.get('description', 'Unknown')}\n"
                prompt += f"  Changes: {json.dumps(hypothesis.get('changes', {}))}\n"
                prompt += f"  Outcome: {outcome.get('summary', 'Unknown')}\n"
                prompt += f"  Decision oracle: {attempt.get('decision_rule', 'OR')}\n"
                prompt += f"  DES validated: {attempt.get('des_validated', False)}\n"
                prompt += f"  ABS validated: {attempt.get('abs_validated', False)}\n"
                if des and abs_metrics:
                    prompt += f"  DES metrics: {json.dumps(des)}\n"
                    prompt += f"  ABS metrics: {json.dumps(abs_metrics)}\n"
            prompt += "\nEach hypothesis is tested independently against the same fixed baseline. Choose the single best untried change.\n"
        else:
            prompt += "This is the first improvement attempt.\n"
        
        prompt += """
Focus on (PRIORITY ORDER - follow strictly):
1. **FIRST PRIORITY - shift_responsibility**: When overloaded + underutilized roles coexist, you MUST use shift_responsibility.
   This is the LOWEST COST intervention (redistributes existing workforce, no new hires needed).
   CRITICAL: Only shift between roles with COMPATIBLE task assignments. Check the role-task mapping!
   - Valid: shift between Clerks, between Loan Officers, or between roles sharing activities
   - INVALID: Appraiser -> Applicant (Applicant cannot do appraisals), AML Investigator -> Clerk (different expertise)
2. **SECOND PRIORITY - resource_allocation**: If shift_responsibility is not viable or already tried, add resources to bottlenecks.
3. **LAST RESORT - calendar_adjustment**: Only extend working hours if shift_responsibility AND resource_allocation are not viable.

    Safety rules:
    - Use meaningful deltas (+1 to +2 resources for additions, +1 to +4 hours for calendar).
    - Avoid any change likely to worsen cycle time or waiting time.
    - Prefer improvements that help at least two of: cycle_time, throughput, waiting_time.
    - **MANDATORY**: If shift_responsibility is in the allowed types, you MUST propose shift_responsibility (lowest organizational cost).
    - For shift_responsibility: VERIFY role compatibility before proposing. External actors (Applicant) cannot perform internal staff tasks.
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
            "expected_impact": {"cycle_time": "-8", "throughput": "6", "waiting_time": "-5"}
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
        if parameter_type not in {"resources", "shift_responsibility", "calendar"}:
            return False
        if not isinstance(specific_changes, list) or len(specific_changes) != 1:
            return False
        change_item = specific_changes[0] if specific_changes else {}
        if not isinstance(change_item, dict):
            return False
        required_change_keys = {"target", "change", "value"}
        if not required_change_keys.issubset(change_item.keys()):
            return False

        expected = hypothesis.get("expected_impact", {})
        if not isinstance(expected, dict):
            return False
        required_impact_keys = {"cycle_time", "throughput", "waiting_time"}
        if not required_impact_keys.issubset(expected.keys()):
            return False
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
            "expected_impact": {"cycle_time": "-8%", "throughput": "+3%", "waiting_time": "-5%"}
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
        # Use adaptive threshold: if no resources >= 80%, use >= 60% as "high utilization"
        # This ensures we always target the most loaded resources
        overloaded_strict = [r for r, u in sorted_utils if u >= 0.80][:3]
        overloaded_moderate = [r for r, u in sorted_utils if u >= 0.60][:5]
        overloaded = overloaded_strict if overloaded_strict else overloaded_moderate
        underutilized = [r for r, u in sorted_utils if u <= 0.25][:5]
        # Track utilization values for overloaded resources (for scaling increments)
        # Store both with and without _profile suffix for robust lookups
        overloaded_utilizations = {}
        for r, u in sorted_utils:
            if u >= 0.60:  # Track all moderately-loaded resources
                overloaded_utilizations[r] = u
                # Also store normalized version
                r_norm = r.replace("_profile", "")
                if r_norm != r:
                    overloaded_utilizations[r_norm] = u

        resources = target_hints.get("resources", [])
        resource_norms = {str(r).replace("_profile", "") for r in resources}
        mappable_overloaded = [r for r in overloaded if r in resource_norms]
        mappable_underutilized = [r for r in underutilized if r in resource_norms]

        # Priority order: shift_responsibility > resources > calendar
        # shift_responsibility is preferred when workload can be rebalanced
        # resources is next choice for adding capacity to bottlenecks
        # calendar is last resort for extending working hours
        allowed = []
        if mappable_overloaded and mappable_underutilized:
            allowed.append("shift_responsibility")
        if overloaded:
            allowed.append("resources")  # Resource allocation is second priority
            allowed.append("calendar")   # Calendar adjustment is last resort

        if not allowed:
            allowed = ["resources"]

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

        # High priority targets: overloaded first (for resource additions), then underutilized (for shift responsibility destination)
        # For resource additions, we should ONLY target overloaded resources
        high_priority_targets = mappable_overloaded + mappable_underutilized
        if not high_priority_targets:
            high_priority_targets = overloaded + underutilized
        # Ensure overloaded targets are at the front for resource allocation scenarios
        overloaded_priority_targets = [t for t in high_priority_targets if t in overloaded or t in mappable_overloaded]
        if not overloaded_priority_targets:
            overloaded_priority_targets = overloaded if overloaded else []

        # Track recently tried (parameter_type, target) tuples so the LLM doesn't repeat them.
        # Include BOTH failed AND validated+applied attempts:
        # - Failed: obviously shouldn't retry the same thing
        # - Validated+applied: already embedded in working log; re-applying doubles the effect
        recently_tried = []
        for attempt in (previous_attempts or [])[-3:]:
            h = attempt.get("hypothesis", {})
            pt = h.get("changes", {}).get("parameter_type", "")
            for sc in (h.get("changes", {}).get("specific_changes") or []):
                t = sc.get("target", "")
                if pt and t:
                    entry = f"{pt}:{t}"
                    if entry not in recently_tried:
                        recently_tried.append(entry)

        return {
            "allowed_parameter_types": list(dict.fromkeys(allowed)),
            "disallowed_parameter_types": ["arrival_rate", "gateway_probabilities"],
            "high_priority_targets": high_priority_targets,
            "overloaded_priority_targets": overloaded_priority_targets,  # For resource additions
            "overloaded_targets": overloaded,
            "underutilized_targets": underutilized,
            "mappable_overloaded_targets": mappable_overloaded,
            "mappable_underutilized_targets": mappable_underutilized,
            "overloaded_utilizations": overloaded_utilizations,
            "waiting_ratio": waiting_ratio,
            "recently_tried": recently_tried,
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
        expected.setdefault("waiting_time", "-5%")

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

        if parameter_type == "shift_responsibility":
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

        if parameter_type == "shift_responsibility":
            recently_tried = strategy_hints.get("recently_tried", [])
            role_task_mapping = target_hints.get("role_task_mapping", {})

            def _rd_entry(src, dst):
                return f"shift_responsibility:{src} → {dst}"

            def _is_external_actor(role_name: str) -> bool:
                """Check if a role represents an external actor who cannot perform internal tasks."""
                role_lower = str(role_name).lower()
                # External actors that cannot be assigned internal work
                external_patterns = ['applicant', 'customer', 'client', 'external', 'user']
                return any(p in role_lower for p in external_patterns)

            def _roles_are_compatible(src_role: str, dst_role: str) -> bool:
                """Check if shifting tasks from src to dst makes semantic sense."""
                # Never shift TO external actors (they can't do internal work)
                if _is_external_actor(dst_role):
                    return False
                # Never shift FROM external actors to internal (their tasks are theirs)
                if _is_external_actor(src_role):
                    return False
                # If we have task mapping, check for overlapping capabilities
                if role_task_mapping:
                    src_tasks = set(role_task_mapping.get(src_role, []))
                    dst_tasks = set(role_task_mapping.get(dst_role, []))
                    # Also try without _profile suffix
                    if not src_tasks:
                        src_norm = src_role.replace("_profile", "")
                        src_tasks = set(role_task_mapping.get(src_norm, []))
                    if not dst_tasks:
                        dst_norm = dst_role.replace("_profile", "")
                        dst_tasks = set(role_task_mapping.get(dst_norm, []))
                    # Roles should have at least some overlapping tasks to be compatible
                    # OR destination should be a generalist (can handle more tasks)
                    if src_tasks and dst_tasks:
                        overlap = src_tasks & dst_tasks
                        if overlap:
                            return True  # They share tasks, compatible
                        # Check if roles are in the same domain (by name similarity)
                        src_base = src_role.replace("_profile", "").split("-")[0].lower()
                        dst_base = dst_role.replace("_profile", "").split("-")[0].lower()
                        if src_base == dst_base:
                            return True  # Same role type, just different instances
                        return False  # Different tasks, different role types
                # Fallback: allow shift if we can't determine (trust LLM)
                return True

            # Build the candidate from/to lists from overloaded and underutilized mappable resources.
            # Fall back to full resource list if mappable lists are empty.
            from_candidates = [resolve_resource_name(r) for r in overloaded_targets]
            from_candidates = [r for r in from_candidates if r]
            to_candidates = [resolve_resource_name(r) for r in underutilized_targets]
            to_candidates = [r for r in to_candidates if r]
            # Filter out external actors from destination candidates
            to_candidates = [r for r in to_candidates if not _is_external_actor(r)]

            # Prefer the LLM's proposed target if it parses correctly, hasn't been tried, and is compatible.
            from_role = to_role = None
            if preferred_target and "→" in str(preferred_target):
                pref_src, pref_dst = [p.strip() for p in str(preferred_target).split("→", 1)]
                pref_src_r = resolve_resource_name(pref_src)
                pref_dst_r = resolve_resource_name(pref_dst)
                if (pref_src_r and pref_dst_r and pref_src_r != pref_dst_r
                        and _rd_entry(pref_src_r, pref_dst_r) not in recently_tried
                        and _roles_are_compatible(pref_src_r, pref_dst_r)):
                    from_role, to_role = pref_src_r, pref_dst_r
                elif pref_src_r and pref_dst_r and not _roles_are_compatible(pref_src_r, pref_dst_r):
                    print(f"  [!] Rejected incompatible role shift: {pref_src_r} → {pref_dst_r}")

            # Otherwise find the first overloaded→underutilized pair that is compatible and not in recently_tried.
            if not (from_role and to_role):
                for src in from_candidates:
                    for dst in to_candidates:
                        if (src != dst and _rd_entry(src, dst) not in recently_tried
                                and _roles_are_compatible(src, dst)):
                            from_role, to_role = src, dst
                            break
                    if from_role and to_role:
                        break

            # Hard fallbacks: just pick first valid compatible pair regardless of recently_tried.
            if not from_role:
                from_role = from_candidates[0] if from_candidates else (resources[0] if resources else "NOT_SET")
            if not to_role:
                non_artificial = [r for r in (to_candidates or resources)
                                  if (not str(r).startswith("artificial_")
                                      and r != from_role
                                      and not _is_external_actor(r)
                                      and _roles_are_compatible(from_role, r))]
                to_role = non_artificial[0] if non_artificial else (resources[-1] if resources else from_role)

            target = f"{from_role} → {to_role}"
            change = preferred_change if preferred_change in {"shift_responsibility", "create_specialist_role"} else "shift_responsibility"
            value = 1
            return target, change, value, "shift_responsibility"

        if parameter_type == "calendar":
            # Calendar adjustment: extend working hours for overloaded resources
            recently_tried = strategy_hints.get("recently_tried", [])

            # Prefer targeting specific overloaded resources, or 'all' for global change
            target = None
            if preferred_target:
                resolved = resolve_resource_name(preferred_target)
                if resolved and f"calendar:{resolved}" not in recently_tried:
                    target = resolved
                elif preferred_target.lower() in ('all', '*', 'global') and "calendar:all" not in recently_tried:
                    target = "all"

            # Fallback: target most overloaded resource not yet tried
            if not target:
                for r in overloaded_targets:
                    resolved = resolve_resource_name(r)
                    if resolved and f"calendar:{resolved}" not in recently_tried:
                        target = resolved
                        break

            # Final fallback: all calendars
            if not target:
                target = "all"

            change = "increase_hours"
            # Default to 2 extra hours; allow LLM to specify 1-4
            try:
                value = int(float(preferred_value or 2))
                value = max(1, min(4, value))  # Clamp to 1-4 hours
            except (TypeError, ValueError):
                value = 2

            return target, change, value, "calendar_adjustment"

        # Default: resources (safe positive capacity)
        # For resource additions, ONLY target overloaded resources (not underutilized!)
        overloaded_priority = strategy_hints.get("overloaded_priority_targets", [])

        target = preferred_target if preferred_target in resources else None

        # First try overloaded priority targets (resources with high utilization)
        if not target:
            for role in overloaded_priority:
                resolved = resolve_resource_name(role)
                if resolved:
                    target = resolved
                    break

        # Fallback to top_targets only if they're overloaded
        if not target:
            overloaded_set = set(strategy_hints.get("overloaded_targets", []))
            for role in top_targets:
                if role in resources and role in overloaded_set:
                    target = role
                    break

        # Final fallback: highest utilization resource from overloaded_utilizations
        if not target:
            overloaded_utils = strategy_hints.get("overloaded_utilizations", {})
            if overloaded_utils:
                # Sort by utilization descending and pick first that's in resources
                sorted_by_util = sorted(overloaded_utils.items(), key=lambda x: x[1], reverse=True)
                for res, util in sorted_by_util:
                    resolved = resolve_resource_name(res)
                    if resolved:
                        target = resolved
                        break

        # Ultimate fallback
        if not target:
            target = resources[0] if resources else "Resource_1"

        change = preferred_change if preferred_change in {"add", "increase_count"} else "add"

        # Determine increment based on utilization severity
        # Critical bottleneck (>90% util) needs bolder action (3-6 resources)
        # High bottleneck (>80% util) needs moderate action (2-4 resources)
        overloaded_utils = strategy_hints.get("overloaded_utilizations", {})

        # Try multiple key formats to find utilization
        target_norm = target.replace("_profile", "")
        target_util = (
            overloaded_utils.get(target, 0) or  # Try exact match first
            overloaded_utils.get(target_norm, 0) or  # Try without _profile
            overloaded_utils.get(f"{target_norm}_profile", 0)  # Try with _profile
        )

        try:
            value = int(float(preferred_value))
        except Exception:
            value = 2  # Default to 2 instead of 1

        # Scale increment based on utilization severity
        if target_util >= 0.95:
            # Critical: allow 4-6 resources
            value = max(4, min(6, value)) if value >= 3 else 4
        elif target_util >= 0.90:
            # Very high: allow 3-5 resources
            value = max(3, min(5, value)) if value >= 2 else 3
        elif target_util >= 0.80:
            # High: allow 2-4 resources
            value = max(2, min(4, value))
        else:
            # Normal: 1-2 resources
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
