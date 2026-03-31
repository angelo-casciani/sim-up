"""Report generation module."""

import json
import os
import time
import traceback
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from gen_ai_hub.proxy.native.openai import chat

# Models that use Amazon Bedrock Converse API (Anthropic Claude models)
AMAZON_BEDROCK_MODELS = {'anthropic--claude-4.5-opus', 'anthropic--claude-4.5-sonnet', 'anthropic--claude-4.5-haiku',
                         'anthropic--claude-4-opus', 'anthropic--claude-4-sonnet'}

# Models that require max_completion_tokens instead of max_tokens (newer OpenAI reasoning models)
MAX_COMPLETION_TOKENS_MODELS = {'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'o3', 'o3-mini', 'o1', 'o1-mini', 'o1-preview'}

# Fallback model for report generation (used after retries exhausted)
# Using gpt-4.1 as fallback since Bedrock endpoints have timeout issues with large prompts
FALLBACK_MODEL = 'gpt-4.1'
SECONDARY_FALLBACK_MODEL = 'anthropic--claude-4.5-sonnet'

# Retry configuration
MAX_RETRIES = 2
RETRY_BASE_DELAY = 1  # seconds (minimal delay, just to allow transient issues to clear)

# Report generation requires more tokens than hypothesis generation
REPORT_MAX_TOKENS = 8192


class ReportGenerator:
    """Generates comprehensive process improvement reports."""

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        """Initialize report generator."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _is_amazon_bedrock_model(self) -> bool:
        """Check if the model uses Amazon Bedrock Converse API."""
        model_lower = self.model_name.lower()
        return any(m in model_lower for m in AMAZON_BEDROCK_MODELS)

    def _uses_max_completion_tokens(self) -> bool:
        """Check if the model requires max_completion_tokens instead of max_tokens (and no temperature)."""
        return self.model_name in MAX_COMPLETION_TOKENS_MODELS

    def _call_openai_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible chat API (works for GPT, Mistral models)."""
        # Build parameters - reasoning models need different params
        params = {
            "model_name": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        if self._uses_max_completion_tokens():
            # Reasoning models: use max_completion_tokens via extra_body, no temperature
            params["extra_body"] = {"max_completion_tokens": self.max_tokens}
        else:
            # Standard models: use max_tokens and temperature
            params["temperature"] = self.temperature
            params["max_tokens"] = self.max_tokens

        response = chat.completions.create(**params)
        return response.choices[0].message.content

    def _call_amazon_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """Call Amazon Bedrock Converse API for Anthropic models."""
        from gen_ai_hub.proxy.native.amazon.clients import Session
        from botocore.config import Config

        # Increase read timeout for large prompts (default is often 60s, we need more for reports)
        config = Config(read_timeout=300, connect_timeout=60)  # 5 min read timeout
        bedrock = Session().client(model_name=self.model_name, config=config)
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

    def _is_bedrock_model(self, model_name: str) -> bool:
        """Check if a specific model uses Amazon Bedrock Converse API."""
        model_lower = model_name.lower()
        return any(m in model_lower for m in AMAZON_BEDROCK_MODELS)

    def _call_llm_single(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        """Make a single LLM call to the specified model."""
        if self._is_bedrock_model(model_name):
            print(f"[ReportGenerator] Using Amazon Bedrock Converse API for {model_name}")
            return self._call_amazon_bedrock_with_model(system_prompt, user_prompt, model_name)
        else:
            print(f"[ReportGenerator] Using OpenAI Chat API for {model_name}")
            return self._call_openai_chat_with_model(system_prompt, user_prompt, model_name)

    def _call_openai_chat_with_model(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        """Call OpenAI-compatible chat API with a specific model."""
        params = {
            "model_name": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        if model_name in MAX_COMPLETION_TOKENS_MODELS:
            params["extra_body"] = {"max_completion_tokens": REPORT_MAX_TOKENS}
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = REPORT_MAX_TOKENS

        response = chat.completions.create(**params)
        return response.choices[0].message.content

    def _call_amazon_bedrock_with_model(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        """Call Amazon Bedrock Converse API with a specific model."""
        from gen_ai_hub.proxy.native.amazon.clients import Session
        from botocore.config import Config

        # Increase read timeout for large prompts (default is often 60s, we need more for reports)
        config = Config(read_timeout=300, connect_timeout=60)  # 5 min read timeout
        bedrock = Session().client(model_name=model_name, config=config)
        conversation = [
            {
                "role": "user",
                "content": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ]
        response = bedrock.converse(
            messages=conversation,
            inferenceConfig={"maxTokens": REPORT_MAX_TOKENS, "temperature": self.temperature},
        )
        return response["output"]["message"]["content"][0]["text"]

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Route LLM call with retry logic and fallback models."""
        # Try with primary model first (with retries)
        last_exception = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[ReportGenerator] Attempt {attempt}/{MAX_RETRIES} with {self.model_name}")
                return self._call_llm_single(system_prompt, user_prompt, self.model_name)
            except Exception as e:
                last_exception = e
                print(f"[ReportGenerator] Attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # Exponential backoff
                    print(f"[ReportGenerator] Waiting {delay}s before retry...")
                    time.sleep(delay)

        # All retries failed with primary model - try fallback model (gpt-4.1)
        if self.model_name != FALLBACK_MODEL:
            print(f"[ReportGenerator] Primary model {self.model_name} failed after {MAX_RETRIES} attempts")
            print(f"[ReportGenerator] Falling back to {FALLBACK_MODEL}")
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"[ReportGenerator] Fallback attempt {attempt}/{MAX_RETRIES} with {FALLBACK_MODEL}")
                    return self._call_llm_single(system_prompt, user_prompt, FALLBACK_MODEL)
                except Exception as e:
                    last_exception = e
                    print(f"[ReportGenerator] Fallback attempt {attempt} failed: {type(e).__name__}: {e}")
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        print(f"[ReportGenerator] Waiting {delay}s before retry...")
                        time.sleep(delay)

        # Try secondary fallback (Sonnet via Bedrock)
        if self.model_name != SECONDARY_FALLBACK_MODEL and FALLBACK_MODEL != SECONDARY_FALLBACK_MODEL:
            print(f"[ReportGenerator] Fallback {FALLBACK_MODEL} also failed")
            print(f"[ReportGenerator] Trying secondary fallback: {SECONDARY_FALLBACK_MODEL}")
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"[ReportGenerator] Secondary fallback attempt {attempt}/{MAX_RETRIES} with {SECONDARY_FALLBACK_MODEL}")
                    return self._call_llm_single(system_prompt, user_prompt, SECONDARY_FALLBACK_MODEL)
                except Exception as e:
                    last_exception = e
                    print(f"[ReportGenerator] Secondary fallback attempt {attempt} failed: {type(e).__name__}: {e}")
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                        print(f"[ReportGenerator] Waiting {delay}s before retry...")
                        time.sleep(delay)

        # All models failed
        raise last_exception
    
    def generate(
        self,
        baseline_metrics: Dict,
        validated_improvements: List[Dict],
        log_path: str,
        history_length: int = 0,
        execution_times: Dict = None,
        iterations_times: List[Dict] = None
    ) -> str:
        """
        Generate comprehensive process improvement report.
        
        Args:
            baseline_metrics: Baseline process metrics
            validated_improvements: List of validated improvements
            log_path: Path to event log being analyzed
            history_length: Number of iterations performed
            execution_times: Dict with execution times for each phase
            iterations_times: List with execution times for each iteration
            
        Returns:
            Generated report text with header and content
        """
        if execution_times is None:
            execution_times = {}
        if iterations_times is None:
            iterations_times = []
            
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(
            baseline_metrics,
            validated_improvements,
            log_path,
            history_length
        )
        
        try:
            print(f"[ReportGenerator] Calling LLM with model={self.model_name}, temperature={self.temperature}")

            # Route to appropriate provider based on model
            response_text = self._call_llm(system_prompt, user_prompt)
            print(f"[ReportGenerator] LLM call successful, response length: {len(response_text)} chars")

            # Build report header with metadata
            report_header = self._build_report_header(
                log_path,
                execution_times,
                iterations_times
            )

            grounded_ledger = self._build_grounded_validation_ledger(validated_improvements)

            # Return header + deterministic facts + LLM narrative
            return report_header + "\n\n" + grounded_ledger + "\n\n" + response_text

        except Exception as e:
            print(f"[ReportGenerator] ERROR: LLM call failed!")
            print(f"[ReportGenerator] Exception type: {type(e).__name__}")
            print(f"[ReportGenerator] Exception message: {e}")
            print(f"[ReportGenerator] Traceback:\n{traceback.format_exc()}")
            print(f"[ReportGenerator] Falling back to template report...")
            return self._generate_fallback_report(
                baseline_metrics,
                validated_improvements,
                log_path,
                execution_times,
                iterations_times
            )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for report generation."""
        return """You are a Business Process Management consultant preparing an executive report.

Create a clear, actionable report that highlights validated process improvements and their business impact.

DO NOT include generation date, event log name, model name, temperature, or max tokens - those are provided separately in the report header.

Structure the report with:

1. EXECUTIVE SUMMARY
   - Lead with the single most impactful finding
   - Quantify total potential savings (time, cost, throughput)
   - State confidence level based on DES/ABS agreement

2. BASELINE ANALYSIS
   - Current process performance with context (is it good/bad vs industry benchmarks if known?)
   - Identify the TOP 3 bottlenecks or inefficiencies from the data
   - Explain WHY these bottlenecks likely exist (root cause hypothesis based on the metrics)

3. VALIDATED IMPROVEMENTS (for each)
   - What was changed and why it works (causal mechanism)
   - Before/after metrics with percentage improvement
   - Confidence assessment: Did DES and ABS agree? Any caveats?
   - Prerequisites: What must be true for this to succeed in production?
   - Risks: What could go wrong? Dependencies on other changes?

4. IMPLEMENTATION ROADMAP
   - Prioritized sequence with SPECIFIC justification for ordering
   - Quick wins (< 1 week effort) vs strategic changes (> 1 month effort)
   - Resource requirements for each change (FTEs, systems, budget estimates)
   - Dependencies between improvements (which must happen first?)
   - Suggested pilot scope before full rollout

5. EXPECTED ROI & BUSINESS CASE
   - Quantified savings per improvement (hours/week, cases/day)
   - Translate to business terms (cost savings, revenue impact, SLA improvement)
   - Payback period estimate where possible
   - Comparison: effort to implement vs. expected benefit

6. RISKS & MONITORING
   - What metrics to track post-implementation
   - Warning signs that improvements aren't working as expected
   - Rollback criteria and contingency recommendations

Critical consistency rules:
- Use ONLY facts present in the provided input payload.
- Do NOT invent iterations, targets, people, or metric values.
- Keep DES and ABS evidence separate; never merge them into one synthetic metric.
- If DES and ABS differ, explicitly acknowledge the divergence and recommend caution.
- For each validated improvement, reference the executed change target from input data.
- Do not claim causality beyond what metrics support.
- If evidence is mixed, use cautious language like "preliminary evidence suggests".
- When estimating ROI or effort, clearly state these are estimates based on the simulation data.

Tone: Write for a VP of Operations who needs to decide whether to fund implementation.
Format with markdown for readability. Use tables for metric comparisons where appropriate."""
    
    def _get_user_prompt(
        self,
        baseline_metrics: Dict,
        validated_improvements: List[Dict],
        log_path: str,
        history_length: int
    ) -> str:
        """Get user prompt for report generation (without execution times)."""
        
        improvements_text = ""
        if validated_improvements:
            improvements_text = "VALIDATED IMPROVEMENTS:\n\n"
            for i, imp in enumerate(validated_improvements, 1):
                hypothesis = imp.get('hypothesis', {})
                impact = imp.get('impact') or imp.get('improvement', {})
                des_validation = impact.get('des_validation', {}) if isinstance(impact, dict) else {}
                abs_validation = impact.get('abs_validation', {}) if isinstance(impact, dict) else {}
                first_change = (hypothesis.get('changes', {}).get('specific_changes', [{}]) or [{}])[0]
                improvements_text += f"{i}. {hypothesis.get('description', 'N/A')}\n"
                improvements_text += f"   Type: {hypothesis.get('improvement_type', 'N/A')}\n"
                improvements_text += f"   Rationale: {hypothesis.get('rationale', 'N/A')}\n"
                improvements_text += f"   Executed target: {first_change.get('target', 'N/A')}\n"
                improvements_text += f"   Executed change: {first_change.get('change', 'N/A')} (value={first_change.get('value', 'N/A')})\n"
                improvements_text += f"   Impact: {impact.get('summary', 'N/A')}\n"
                improvements_text += f"   DES validation summary: {des_validation.get('summary', 'N/A')}\n"
                improvements_text += f"   ABS validation summary: {abs_validation.get('summary', 'N/A')}\n"
                improvements_text += f"   Detailed metrics:\n"
                for metric, data in impact.get('improvements', {}).items():
                    baseline_val = data['baseline']
                    new_val = data['new']
                    change_pct = data['change_pct']
                    improvements_text += (
                        f"     - {metric}: {baseline_val:.2f} -> {new_val:.2f} ({change_pct:+.1f}%)\n"
                    )
                improvements_text += "\n"
        else:
            improvements_text = "No validated improvements were found within the iteration limit."
        
        prompt = f"""Generate a comprehensive process improvement report for the following:

EVENT LOG: {os.path.basename(log_path)}

BASELINE METRICS:
{json.dumps(baseline_metrics, indent=2)}

{improvements_text}

NUMBER OF ITERATIONS: {history_length}

Create a professional report that executives can use to make decisions about process improvements.
Include specific numbers and percentages from the metrics provided.

Mandatory reporting rules:
- Keep DES-prefixed and ABS-prefixed metrics clearly separated.
- Do not present ABS-only gains as DES gains, or vice versa.
- Use the executed target/change exactly as provided.
- If simulator signals conflict, mention uncertainty and recommend controlled rollout.
- Never state improvements that are absent from the provided metrics."""
        
        return prompt

    def _build_grounded_validation_ledger(self, validated_improvements: List[Dict]) -> str:
        """Create deterministic section with executed changes and validation outcomes."""
        lines = []
        lines.append("## 0. VALIDATION LEDGER (GROUND TRUTH)")
        lines.append("")
        lines.append("This section is generated directly from pipeline data (not LLM interpretation).")

        if not validated_improvements:
            lines.append("")
            lines.append("No validated improvements were accepted.")
            return "\n".join(lines)

        for item in validated_improvements:
            iteration = item.get("iteration", "n/a")
            hypothesis = item.get("hypothesis", {}) or {}
            changes = (hypothesis.get("changes", {}) or {}).get("specific_changes", []) or []
            first_change = changes[0] if changes and isinstance(changes[0], dict) else {}
            improvement = item.get("improvement", {}) or {}
            des_validation = improvement.get("des_validation", {}) or {}
            abs_validation = improvement.get("abs_validation", {}) or {}

            lines.append("")
            lines.append(f"### Iteration {iteration}")
            lines.append(f"- Executed parameter type: {hypothesis.get('changes', {}).get('parameter_type', 'n/a')}")
            lines.append(f"- Executed target: {first_change.get('target', 'n/a')}")
            lines.append(f"- Executed change: {first_change.get('change', 'n/a')} (value={first_change.get('value', 'n/a')})")
            lines.append(f"- Decision rule: {improvement.get('decision_rule', 'n/a')}")
            lines.append(f"- Overall accepted: {bool(improvement.get('is_improvement'))}")
            lines.append(f"- DES validation: {des_validation.get('summary', 'n/a')}")
            lines.append(f"- ABS validation: {abs_validation.get('summary', 'n/a')}")

        return "\n".join(lines)
    
    def _build_report_header(
        self,
        log_path: str,
        execution_times: Dict,
        iterations_times: List[Dict]
    ) -> str:
        """Build report header with metadata and execution times."""
        from datetime import datetime
        
        header = "PROCESS IMPROVEMENT VALIDATION REPORT\n"
        header += "="*80 + "\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Event Log: {log_path}\n"
        header += f"Model: {self.model_name}\n"
        header += f"Temperature: {self.temperature}\n"
        header += f"Max Tokens: {self.max_tokens}\n"
        
        # Add execution times if available
        if execution_times or iterations_times:
            total_time = execution_times.get('total_pipeline', 0)
            baseline_time = execution_times.get('step_1_baseline_discovery', 0)
            report_time = execution_times.get('step_6_report_generation', 0)
            
            if total_time > 0:
                header += f"Total Execution Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n"
            if baseline_time > 0:
                header += f"Baseline Discovery Time: {baseline_time:.1f} seconds ({baseline_time/60:.1f} minutes)\n"
            
            if iterations_times:
                header += f"Iterations: {len(iterations_times)}\n"
                for it_data in iterations_times:
                    it_num = it_data['iteration']
                    it_total = it_data['total_time']
                    it_status = it_data['status']
                    header += f"  - Iteration {it_num} ({it_status}): {it_total:.1f}s"
                    if 'hypothesis_time' in it_data:
                        header += f" [Hypothesis: {it_data['hypothesis_time']:.1f}s, Simulation: {it_data['simulation_time']:.1f}s, Validation: {it_data['validation_time']:.1f}s]"
                    header += "\n"
            
            if report_time > 0:
                header += f"Report Generation Time: {report_time:.1f} seconds\n"
        
        header += "="*80
        
        return header
    
    def _generate_fallback_report(
        self,
        baseline_metrics: Dict,
        validated_improvements: List[Dict],
        log_path: str,
        execution_times: Dict = None,
        iterations_times: List[Dict] = None
    ) -> str:
        """Generate a fallback report using template."""
        if execution_times is None:
            execution_times = {}
        if iterations_times is None:
            iterations_times = []

        # Build header with metadata
        report = self._build_report_header(log_path, execution_times, iterations_times)

        report += f"""

⚠️ **WARNING: THIS IS A FALLBACK TEMPLATE REPORT** ⚠️
The LLM-based report generation failed. Check logs for details.

**Generated:** {os.path.basename(log_path)}

---

## 1. EXECUTIVE SUMMARY

This report presents a comprehensive analysis of the current process and the impact of validated process improvements. The improvement validation pipeline has identified and tested specific enhancements to process performance.

**Key Outcomes:**
"""
        
        if validated_improvements:
            for imp in validated_improvements:
                impact = imp.get('impact') or imp.get('improvement', {})
                report += f"- {impact.get('summary', 'N/A')}\n"
        else:
            report += "- Analysis completed; no significant improvements identified within iteration limit\n"
        
        report += f"""

---

## 2. BASELINE ANALYSIS

**Current Process Performance:**

| Metric                | Value         |
|-----------------------|:--------------|
| Average Cycle Time    | {baseline_metrics.get('avg_cycle_time', 'N/A')} sec   |
| Average Throughput    | {baseline_metrics.get('avg_throughput', 'N/A')} cases |
| Average Waiting Time  | {baseline_metrics.get('avg_waiting_time', 'N/A')} sec   |

**Key Observations:**
- Process baseline established through SIMOD discovery and simulation
- Resource utilization measured across all process resources
- Waiting times identified as a key performance indicator

---

## 3. VALIDATED IMPROVEMENTS

"""
        
        if validated_improvements:
            for i, imp in enumerate(validated_improvements, 1):
                hypothesis = imp.get('hypothesis', {})
                impact = imp.get('impact') or imp.get('improvement', {})
                report += f"### Improvement {i}: {hypothesis.get('description', 'N/A')}\n\n"
                report += f"**Type:** {hypothesis.get('improvement_type', 'N/A')}\n"
                report += f"**Rationale:** {hypothesis.get('rationale', 'N/A')}\n\n"
                report += "**Impact Summary:**\n"
                report += f"{impact.get('summary', 'N/A')}\n\n"
        else:
            report += "No validated improvements identified.\n\n"
        
        report += """---

## 4. IMPLEMENTATION ROADMAP

Recommend reviewing proposed improvements with stakeholders and initiating pilot testing.

---

## 5. EXPECTED ROI

Implementation of validated improvements is expected to result in optimized process performance.
"""
        
        return report
