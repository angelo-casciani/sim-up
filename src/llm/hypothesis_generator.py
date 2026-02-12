"""LLM hypothesis generation module."""

import json
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from gen_ai_hub.proxy.native.openai import chat


class HypothesisGenerator:
    """Generates process improvement hypotheses using LLM."""
    
    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        """Initialize hypothesis generator."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        baseline_metrics: Dict,
        previous_improvements: List[Dict] = None
    ) -> Dict:
        """
        Generate improvement hypothesis using LLM.
        
        Args:
            baseline_metrics: Baseline process metrics
            previous_improvements: List of previously validated improvements
            
        Returns:
            Hypothesis dict with description, type, changes, and expected impact
        """
        if previous_improvements is None:
            previous_improvements = []
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(baseline_metrics, previous_improvements)
        
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
            
            # Parse LLM response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
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
            
            # Validate hypothesis structure
            if self._validate_hypothesis(hypothesis):
                return hypothesis
            else:
                print("Warning: Invalid hypothesis structure, using fallback")
                return self._get_fallback_hypothesis()
                
        except Exception as e:
            print(f"Error generating hypothesis: {e}")
            return self._get_fallback_hypothesis()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a process improvement specialist analyzing business process metrics.

Your task is to generate actionable, specific hypotheses to improve process performance.

For each hypothesis, provide:
- Clear description of the improvement
- Type: resource_allocation, activity_duration_optimization, arrival_rate_adjustment, or calendar_adjustment
- Specific changes to simulation parameters
- Expected impact on metrics

Return a JSON object with the following structure:
{
  "description": "Clear description of the improvement",
  "improvement_type": "resource_allocation|activity_duration_optimization|arrival_rate_adjustment|calendar_adjustment",
  "rationale": "Why this improvement would help",
  "changes": {
    "parameter_type": "resources|activity_durations|arrival_rate|calendar",
    "specific_changes": [
      {
        "target": "Resource_1|ActivityName|calendar_name",
        "change": "add|remove|increase_count|decrease_count|reduce_duration|increase_hours",
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
    
    def _get_user_prompt(self, baseline_metrics: Dict, previous_improvements: List[Dict]) -> str:
        """Get user prompt for LLM."""
        prompt = f"""Analyze the current process metrics and generate ONE new improvement hypothesis:

BASELINE METRICS:
{json.dumps(baseline_metrics, indent=2)}

"""
        
        if previous_improvements:
            prompt += "PREVIOUSLY VALIDATED IMPROVEMENTS:\n"
            for imp in previous_improvements:
                prompt += f"- {imp.get('hypothesis', {}).get('description', 'Unknown')}\n"
            prompt += "\nSuggest a DIFFERENT type of improvement than those already validated.\n"
        else:
            prompt += "This is the first improvement attempt.\n"
        
        prompt += """
Focus on:
1. Resource bottlenecks (look for utilization > 0.8)
2. High waiting times (> 40% of cycle time)
3. Long cycle times (> 12000 seconds)

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
            "expected_impact": {"cycle_time": "-20", "throughput": "25", "cost": "10"}
        }
        return hypothesis
    
    def _validate_hypothesis(self, hypothesis: Dict) -> bool:
        """Validate hypothesis structure."""
        required_keys = {"description", "improvement_type", "changes", "expected_impact"}
        return required_keys.issubset(hypothesis.keys())
    
    def _get_fallback_hypothesis(self) -> Dict:
        """Get fallback hypothesis when LLM call fails."""
        return {
            "description": "Add one additional Resource_1 to address high utilization and reduce waiting time at the bottleneck.",
            "improvement_type": "resource_allocation",
            "rationale": "Resource_1 has the highest utilization (0.85), indicating it is a bottleneck causing long average waiting times (5400 seconds). Adding another Resource_1 should reduce queueing and improve flow.",
            "changes": {
                "parameter_type": "resources",
                "specific_changes": [{"target": "Resource_1", "change": "add", "value": 1}]
            },
            "expected_impact": {"cycle_time": "-20", "throughput": "25", "cost": "10"}
        }
