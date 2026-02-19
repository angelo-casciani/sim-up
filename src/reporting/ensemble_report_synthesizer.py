"""Ensemble Report Synthesizer: Transforms divergence into research insight (Reality Check Report)."""

from datetime import datetime
from typing import Dict
from simulator.ensemble_arbiter import EnsembleArbiter, HypothesisType


class EnsembleReportSynthesizer:
    """Generates Reality Check Reports showcasing the Stereoscopic View (DES vs ABS)."""
    
    @staticmethod
    def synthesize_report(
        hypothesis: Dict,
        baseline_metrics: Dict,
        des_metrics: Dict,
        abs_metrics: Dict,
        arbitration_result: Dict,
        execution_times: Dict
    ) -> str:
        """
        Generate a unified Reality Check Report synthesizing both simulators.
        
        Args:
            hypothesis: Original hypothesis from LLM
            baseline_metrics: Original process metrics
            des_metrics: Prosimos (Structural/DES) results
            abs_metrics: AgentSimulator (Behavioral/ABS) results
            arbitration_result: Arbitration output
            execution_times: Timing info
        
        Returns:
            Formatted report string
        """
        report = []
        
        # === HEADER ===
        report.append("=" * 90)
        report.append("STEREOSCOPIC PROCESS IMPROVEMENT ANALYSIS")
        report.append("Reality Check Report: Structural vs. Behavioral Perspectives")
        report.append("=" * 90)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Hypothesis Type: {arbitration_result['hypothesis_type'].upper()}")
        report.append(f"Primary Trust Source: {arbitration_result['primary_source']}")
        report.append(f"Ensemble Confidence: {arbitration_result['confidence']*100:.1f}%")
        report.append("")
        
        # === HYPOTHESIS ===
        report.append("-" * 90)
        report.append("HYPOTHESIS")
        report.append("-" * 90)
        report.append(f"Change Type: {hypothesis.get('parameter_type', 'Unknown')}")
        report.append(f"Description: {hypothesis.get('description', 'N/A')}")
        report.append(f"Reasoning: {hypothesis.get('reasoning', 'N/A')}")
        report.append("")
        
        # === BASELINE ===
        report.append("-" * 90)
        report.append("BASELINE METRICS (Current Process)")
        report.append("-" * 90)
        report.append(f"  Cycle Time:      {baseline_metrics.get('avg_cycle_time', 0):>12,.0f} seconds")
        report.append(f"  Throughput:      {baseline_metrics.get('avg_throughput', 0):>12,.0f} cases/day")
        report.append(f"  Waiting Time:    {baseline_metrics.get('avg_waiting_time', 0):>12,.0f} seconds")
        report.append(f"  Total Cost:      {baseline_metrics.get('total_cost', 0):>12,.0f}")
        report.append(f"  Resources:       {len(baseline_metrics.get('resource_utilization', {})):>12} agents/pools")
        report.append("")
        
        # === STRUCTURAL VIEW (PROSIMOS) ===
        report.append("-" * 90)
        report.append("STRUCTURAL VIEW: Prosimos (DES - Discrete Event Simulation)")
        report.append("Perspective: How the process SHOULD work (theoretical flow efficiency)")
        report.append("-" * 90)
        report.append(f"  Cycle Time:      {des_metrics.get('avg_cycle_time', 0):>12,.0f} seconds  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_cycle_time'), des_metrics.get('avg_cycle_time')))
        report.append(f"  Throughput:      {des_metrics.get('avg_throughput', 0):>12,.0f} cases/day  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_throughput'), des_metrics.get('avg_throughput')))
        report.append(f"  Waiting Time:    {des_metrics.get('avg_waiting_time', 0):>12,.0f} seconds  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_waiting_time'), des_metrics.get('avg_waiting_time')))
        report.append(f"  Total Cost:      {des_metrics.get('total_cost', 0):>12,.0f}")
        report.append("")
        report.append("  Reliability:     HIGH for control-flow and structural changes")
        report.append("  Validity:        Assumes resources are available/substitutable")
        report.append("")
        
        # === BEHAVIORAL VIEW (AGENTSIMULATOR) ===
        report.append("-" * 90)
        report.append("BEHAVIORAL VIEW: AgentSimulator (ABS - Agent-Based Simulation)")
        report.append("Perspective: How the process ACTUALLY works (organizational/social friction)")
        report.append("-" * 90)
        report.append(f"  Cycle Time:      {abs_metrics.get('avg_cycle_time', 0):>12,.0f} seconds  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_cycle_time'), abs_metrics.get('avg_cycle_time')))
        report.append(f"  Throughput:      {abs_metrics.get('avg_throughput', 0):>12,.0f} cases/day  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_throughput'), abs_metrics.get('avg_throughput')))
        report.append(f"  Waiting Time:    {abs_metrics.get('avg_waiting_time', 0):>12,.0f} seconds  " +
                     EnsembleReportSynthesizer._delta_indicator(baseline_metrics.get('avg_waiting_time'), abs_metrics.get('avg_waiting_time')))
        report.append(f"  Total Cost:      {abs_metrics.get('total_cost', 0):>12,.0f}")
        report.append(f"  Agents Modeled:  {abs_metrics.get('num_agents', 0):>12} with individual calendars/skills")
        report.append("")
        report.append("  Reliability:     HIGH for resource allocation and handover dynamics")
        report.append("  Validity:        Captures specific agent behavior, skills, availability")
        report.append("")
        
        # === DIVERGENCE ANALYSIS (THE RESEARCH VALUE) ===
        report.append("-" * 90)
        report.append("BEHAVIORAL GAP ANALYSIS (Stereoscopic Insight)")
        report.append("The difference between structural potential and organizational reality")
        report.append("-" * 90)
        report.append(f"  Status: {arbitration_result['behavioral_gap']}")
        report.append("")
        
        divergence = arbitration_result.get("divergence", {})
        for metric, data in divergence.items():
            if isinstance(data, dict) and "des" in data:
                delta = data.get("delta_pct", 0)
                significance = "⚠️ SIGNIFICANT" if data.get("is_significant") else "   (minor)"
                report.append(f"  {metric}:")
                report.append(f"    DES predicts:  {data['des']:>.0f}")
                report.append(f"    ABS predicts:  {data['abs']:>.0f}")
                report.append(f"    Divergence:    {delta:+.1f}% {significance}")
                report.append("")
        
        # === ARBITRATION LOGIC ===
        report.append("-" * 90)
        report.append("ARBITRATION LOGIC")
        report.append(f"Arbitration Strategy: {arbitration_result['arbitration_strategy'].upper()}")
        report.append("-" * 90)
        
        hyp_type = arbitration_result.get("hypothesis_type", "control_flow").lower()
        if hyp_type == "control_flow":
            report.append("  Hypothesis Type: CONTROL-FLOW CHANGE (structural logic)")
            report.append("  Arbitration: TRUST PROSIMOS (handles BPMN logic natively)")
            report.append("               AgentSimulator may need rule engine updates")
        elif hyp_type == "resource_allocation":
            report.append("  Hypothesis Type: RESOURCE ALLOCATION (add/remove agents)")
            report.append("  Arbitration: TRUST AGENTSIMULATOR (models individual calendars/skills)")
            report.append("               Prosimos treats agents as anonymous resource pools")
        elif hyp_type == "social_handover":
            report.append("  Hypothesis Type: SOCIAL/HANDOVER CHANGE (routing, responsibilities)")
            report.append("  Arbitration: TRUST AGENTSIMULATOR (captures interaction dynamics)")
            report.append("               Prosimos cannot see interpersonal relationships")
        elif hyp_type == "arrival_surge":
            report.append("  Hypothesis Type: ARRIVAL/VOLUME SURGE")
            report.append("  Arbitration: REQUIRE CONSENSUS (both perspectives equally critical)")
            report.append("               Collapse in either model = serious risk")
        elif hyp_type == "activity_duration":
            report.append("  Hypothesis Type: ACTIVITY DURATION CHANGE (speedup)")
            report.append("  Arbitration: TRUST PROSIMOS (structural timing neutral)")
            report.append("               Verify with AgentSimulator for side effects")
        report.append("")
        
        # === WARNINGS (CONSERVATIVE PRINCIPLE) ===
        if arbitration_result.get("warnings"):
            report.append("-" * 90)
            report.append("⚠️  WARNINGS (Conservative Risk Assessment)")
            report.append("-" * 90)
            for warning in arbitration_result["warnings"]:
                report.append(f"  {warning}")
            report.append("")
        
        # === INSIGHT & RECOMMENDATION ===
        report.append("-" * 90)
        report.append("REALITY CHECK VERDICT")
        report.append("-" * 90)
        report.append(f"  {arbitration_result['insight']}")
        report.append("")
        
        # Determine improvement validatino
        des_cycle = des_metrics.get('avg_cycle_time', 0)
        abs_cycle = abs_metrics.get('avg_cycle_time', 0)
        baseline_cycle = baseline_metrics.get('avg_cycle_time', 0)
        
        des_improved = des_cycle < baseline_cycle * 0.95
        abs_improved = abs_cycle < baseline_cycle * 0.95
        
        if arbitration_result['arbitration_strategy'] == 'des':
            report.append(f"  RECOMMENDATION: {'✅ ACCEPT' if des_improved else '❌ REJECT'} (Structural model)")
            report.append(f"  Note: Verify with AgentSimulator that agents can execute this logic")
        elif arbitration_result['arbitration_strategy'] == 'abs':
            report.append(f"  RECOMMENDATION: {'✅ ACCEPT' if abs_improved else '❌ REJECT'} (Behavioral model)")
            report.append(f"  Note: Structure supports it, but organizational dynamics matter most")
        else:  # consensus
            both_agree = des_improved and abs_improved
            report.append(f"  RECOMMENDATION: {'✅ ACCEPT (HIGH CONFIDENCE)' if both_agree else '⚠️  UNCERTAIN - Models Disagree'}")
            if both_agree:
                report.append(f"  Both structural and behavioral analyses confirm improvement")
            else:
                report.append(f"  Structural model: {'Positive' if des_improved else 'Negative'}")
                report.append(f"  Behavioral model: {'Positive' if abs_improved else 'Negative'}")
        report.append("")
        
        # === EXECUTION TIMING ===
        report.append("-" * 90)
        report.append("EXECUTION TIMING")
        report.append("-" * 90)
        if execution_times:
            for phase, duration in execution_times.items():
                report.append(f"  {phase:.<40} {duration:>8.1f}s")
        report.append("")
        
        report.append("=" * 90)
        report.append("END OF REALITY CHECK REPORT")
        report.append("=" * 90)
        
        return "\n".join(report)
    
    @staticmethod
    def _delta_indicator(baseline: float, simulated: float) -> str:
        """Show improvement/degradation indicator."""
        if baseline == 0:
            return ""
        
        pct_change = ((simulated - baseline) / baseline) * 100
        
        if abs(pct_change) < 2:
            return "(no change)"
        elif pct_change < 0:
            return f"(↓ {abs(pct_change):.1f}% improvement)"
        else:
            return f"(↑ {pct_change:.1f}% degradation)"
