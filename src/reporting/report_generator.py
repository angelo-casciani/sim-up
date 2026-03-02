"""Report generation module."""

import json
import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from gen_ai_hub.proxy.native.openai import chat


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
            response = chat.completions.create(
                model_name=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Build report header with metadata
            report_header = self._build_report_header(
                log_path,
                execution_times,
                iterations_times
            )

            grounded_ledger = self._build_grounded_validation_ledger(validated_improvements)
            
            # Return header + deterministic facts + LLM narrative
            return report_header + "\n\n" + grounded_ledger + "\n\n" + response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating report: {e}")
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
1. EXECUTIVE SUMMARY - Key findings and overall impact
2. BASELINE ANALYSIS - Current process performance
3. VALIDATED IMPROVEMENTS - Each improvement with specific metrics
4. IMPLEMENTATION ROADMAP - Recommended sequence and dependencies
5. EXPECTED ROI - Quantified business value

Critical consistency rules:
- Use ONLY facts present in the provided input payload.
- Do NOT invent iterations, targets, people, or metric values.
- Keep DES and ABS evidence separate; never merge them into one synthetic metric.
- If DES and ABS differ, explicitly acknowledge the divergence.
- For each validated improvement, reference the executed change target from input data.
- Do not claim causality beyond what metrics support.
- If evidence is mixed, use cautious language.

Use clear language suitable for business stakeholders, not just technical experts.
Format with markdown for readability."""
    
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
                        f"     - {metric}: {baseline_val:.2f} → {new_val:.2f} ({change_pct:+.1f}%)\n"
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
