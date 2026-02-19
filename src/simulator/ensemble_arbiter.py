"""Ensemble Arbiter: Synthesizes Prosimos (DES) and AgentSimulator (ABS) results."""

from typing import Dict, Tuple
from enum import Enum


class HypothesisType(Enum):
    """Types of hypotheses with different trust profiles."""
    CONTROL_FLOW = "control_flow"  # Struct changes (Skip, reorder)
    RESOURCE_ALLOCATION = "resource_allocation"  # Add/remove agents
    SOCIAL_HANDOVER = "social_handover"  # Routing, responsibilities
    ARRIVAL_SURGE = "arrival_surge"  # Volume/timing changes
    ACTIVITY_DURATION = "activity_duration"  # Task speedup


class EnsembleArbiter:
    """Arbitrates between Prosimos (DES) and AgentSimulator (ABS) results."""
    
    # Confidence trust matrix: (DES reliability, ABS reliability)
    TRUST_MATRIX = {
        HypothesisType.CONTROL_FLOW: {"des": 0.9, "abs": 0.5, "arbitration": "des"},
        HypothesisType.RESOURCE_ALLOCATION: {"des": 0.6, "abs": 0.9, "arbitration": "abs"},
        HypothesisType.SOCIAL_HANDOVER: {"des": 0.3, "abs": 0.9, "arbitration": "abs"},
        HypothesisType.ARRIVAL_SURGE: {"des": 0.8, "abs": 0.8, "arbitration": "consensus"},
        HypothesisType.ACTIVITY_DURATION: {"des": 0.8, "abs": 0.7, "arbitration": "des"},
    }
    
    @staticmethod
    def classify_hypothesis(hypothesis: Dict) -> HypothesisType:
        """Classify hypothesis type from LLM output with priority order."""
        changes = hypothesis.get("changes", {}) if isinstance(hypothesis.get("changes", {}), dict) else {}
        param_type = (changes.get("parameter_type") or hypothesis.get("parameter_type", "")).lower()
        description = hypothesis.get("description", "").lower()
        reasoning = (hypothesis.get("reasoning") or hypothesis.get("rationale") or "").lower()
        
        # Combine all text for more robust matching
        full_text = f"{param_type} {description} {reasoning}".lower()
        
        # Priority 1: Explicit parameter_type match
        if "role_redesign" in param_type or "role" in param_type:
            return HypothesisType.SOCIAL_HANDOVER
        if "resource" in param_type or "agent" in param_type:
            return HypothesisType.RESOURCE_ALLOCATION
        if "routing" in param_type or "rule" in param_type or "gateway" in param_type:
            return HypothesisType.CONTROL_FLOW
        if "arrival" in param_type or "distribution" in param_type:
            return HypothesisType.ARRIVAL_SURGE
        if "duration" in param_type or "timing" in param_type:
            return HypothesisType.ACTIVITY_DURATION
        
        # Priority 2: Strong indicators in description
        # Control Flow: Skip, reorder, condition, gateway, branch
        if any(word in description for word in ["skip", "reorder", "gateway", "branch",
                                                  "condition", "sequential", "parallel"]):
            return HypothesisType.CONTROL_FLOW
        
        # Resource Allocation: Add/remove staff, capacity, bottleneck
        if any(word in description for word in ["add", "remove", "hire", "staff", "clerk",
                                                  "agent", "representative", "handler",
                                                  "allocat", "capacity", "bottleneck"]):
            return HypothesisType.RESOURCE_ALLOCATION
        
        # Social Handover: Route, handover, senior, expertise, relationship
        if any(word in description for word in ["route", "handover", "senior", "direct",
                                                  "expertise", "relationship", "complex"]):
            return HypothesisType.SOCIAL_HANDOVER
        
        # Arrival Surge: Volume increase, double, expand, incoming
        if any(word in description for word in ["volume", "surge", "double", "triple",
                                                  "increase", "expand", "incoming", "arrival"]):
            return HypothesisType.ARRIVAL_SURGE
        
        # Activity Duration: Speedup, automation, reduce time, process review
        if any(word in description for word in ["speedup", "automation", "faster",
                                                  "reduce", "shorten", "automate"]):
            return HypothesisType.ACTIVITY_DURATION
        
        # Priority 3: Description keywords (combined logic)
        if "process" in description and "verify" in description:
            return HypothesisType.CONTROL_FLOW
        if "route" in description:
            return HypothesisType.SOCIAL_HANDOVER
        if "increase" in description and ("volume" in description or "incoming" in description):
            return HypothesisType.ARRIVAL_SURGE
        
        return HypothesisType.CONTROL_FLOW  # Default fallback
    
    @staticmethod
    def arbitrate(
        hypothesis: Dict,
        des_metrics: Dict,  # Prosimos results
        abs_metrics: Dict,  # AgentSimulator results
        threshold: float = 2.0
    ) -> Dict:
        """
        Synthesize DES and ABS results with context-aware weighting.
        
        Returns:
            Dict with unified metrics, confidence, divergence analysis, and warnings
        """
        hyp_type = EnsembleArbiter.classify_hypothesis(hypothesis)
        trust = EnsembleArbiter.TRUST_MATRIX.get(hyp_type, EnsembleArbiter.TRUST_MATRIX[HypothesisType.CONTROL_FLOW])
        abs_quality = EnsembleArbiter._assess_abs_quality(abs_metrics, des_metrics)

        effective_des_trust = trust["des"]
        effective_abs_trust = trust["abs"]
        if abs_quality["is_uninformative"]:
            effective_abs_trust = min(effective_abs_trust, 0.2)
            effective_des_trust = max(effective_des_trust, 0.8)
        
        # Calculate divergence (Behavioral Gap)
        divergence = EnsembleArbiter._calculate_divergence(des_metrics, abs_metrics)
        risk_score, risk_level = EnsembleArbiter._calculate_risk_score(divergence)
        
        # Determine arbitration strategy
        arbitration_strategy = trust["arbitration"]
        if abs_quality["is_uninformative"] and arbitration_strategy == "abs":
            arbitration_strategy = "consensus"

        # Trust the trust matrix - don't override based on risk score
        # The trust matrix already considers which simulator to trust for each hypothesis type
        risk_adjusted = False
        
        # Synthesize results
        if arbitration_strategy == "des":
            unified_metrics = des_metrics.copy()
            primary_source = "Prosimos (DES - Structural Efficiency Model)"
            confidence = effective_des_trust
        elif arbitration_strategy == "abs":
            unified_metrics = abs_metrics.copy()
            primary_source = "AgentSimulator (ABS - Behavioral Realism Model)"
            confidence = effective_abs_trust
        else:  # consensus or default
            if abs_quality["is_uninformative"]:
                unified_metrics = EnsembleArbiter._weighted_consensus_metrics(
                    des_metrics,
                    abs_metrics,
                    des_weight=0.85,
                    abs_weight=0.15
                )
                primary_source = "Ensemble Consensus (DES-dominant, ABS low-signal)"
            else:
                unified_metrics = EnsembleArbiter._consensus_metrics(des_metrics, abs_metrics)
                primary_source = "Ensemble Consensus (DES + ABS)"
            confidence = (effective_des_trust + effective_abs_trust) / 2

        if risk_adjusted:
            primary_source = "Ensemble Consensus (Risk-Adjusted)"
            confidence = max(0.3, confidence * (1 - (risk_score * 0.5)))
        
        # Check for red flags (Conservative Principle)
        warnings = EnsembleArbiter._check_red_flags(des_metrics, abs_metrics, divergence, risk_score, risk_level)
        if abs_quality["is_uninformative"]:
            warnings.append(f"⚠️ ABS LOW SIGNAL: {abs_quality['reason']}")
        
        return {
            "unified_metrics": unified_metrics,
            "hypothesis_type": hyp_type.value,
            "arbitration_strategy": arbitration_strategy,
            "primary_source": primary_source,
            "confidence": confidence,
            "des_metrics": des_metrics,
            "abs_metrics": abs_metrics,
            "divergence": divergence,
            "behavioral_gap": divergence["gap_summary"],
            "warnings": warnings,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "abs_quality": abs_quality,
            "insight": EnsembleArbiter._generate_insight(divergence, hyp_type, warnings)
        }

    @staticmethod
    def _assess_abs_quality(abs_metrics: Dict, des_metrics: Dict) -> Dict:
        """Assess whether ABS metrics are informative enough for strong weighting."""
        if not abs_metrics:
            return {"is_uninformative": True, "reason": "ABS metrics missing"}

        simulation_source = str(abs_metrics.get("simulation_source", "")).lower()
        if "unavailable" in simulation_source or "mock" in simulation_source:
            return {"is_uninformative": True, "reason": "ABS is unavailable/mock"}

        resource_utilization = abs_metrics.get("resource_utilization", {})
        if isinstance(resource_utilization, dict) and len(resource_utilization) >= 5:
            values = [float(v) for v in resource_utilization.values() if isinstance(v, (int, float))]
            if values:
                spread = max(values) - min(values)
                if spread <= 0.02:
                    return {
                        "is_uninformative": True,
                        "reason": f"resource utilization is nearly flat (spread={spread:.3f})"
                    }

        abs_tp = float(abs_metrics.get("avg_throughput", 0) or 0)
        des_tp = float(des_metrics.get("avg_throughput", 0) or 0)
        if des_tp > 0 and abs_tp > 0 and abs_tp < (0.3 * des_tp):
            return {
                "is_uninformative": True,
                "reason": f"ABS throughput much lower than DES ({abs_tp:.1f} vs {des_tp:.1f})"
            }

        return {"is_uninformative": False, "reason": "ABS has usable signal"}
    
    @staticmethod
    def _calculate_divergence(des_metrics: Dict, abs_metrics: Dict) -> Dict:
        """Calculate the behavioral gap between DES and ABS."""
        divergence = {}
        
        for key in ["avg_cycle_time", "avg_throughput", "avg_waiting_time"]:
            if key in des_metrics and key in abs_metrics:
                des_val = des_metrics[key]
                abs_val = abs_metrics[key]
                
                if des_val != 0:
                    pct_diff = ((abs_val - des_val) / des_val) * 100
                else:
                    pct_diff = 0
                
                divergence[key] = {
                    "des": des_val,
                    "abs": abs_val,
                    "delta_pct": pct_diff,
                    "is_significant": abs(pct_diff) > 5  # >5% is significant divergence
                }
        
        # Summarize gap
        significant_gaps = [k for k, v in divergence.items() if v.get("is_significant")]
        gap_summary = f"Behavioral gap detected in: {', '.join(significant_gaps)}" if significant_gaps else "Both models agree"
        
        divergence["gap_summary"] = gap_summary
        
        return divergence

    @staticmethod
    def _calculate_risk_score(divergence: Dict) -> Tuple[float, str]:
        """Calculate a risk score based on DES/ABS divergence."""
        risk_score = 0.0
        significant_count = 0
        max_delta = 0.0

        for key in ["avg_cycle_time", "avg_throughput", "avg_waiting_time"]:
            div = divergence.get(key)
            if not isinstance(div, dict):
                continue
            delta = abs(div.get("delta_pct", 0))
            max_delta = max(max_delta, delta)
            if div.get("is_significant"):
                significant_count += 1
            if delta >= 50:
                risk_score += 0.4
            elif delta >= 20:
                risk_score += 0.2
            elif delta >= 10:
                risk_score += 0.1

        risk_score += min(0.1 * significant_count, 0.3)
        risk_score = min(1.0, risk_score)

        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return risk_score, risk_level
    
    @staticmethod
    def _consensus_metrics(des_metrics: Dict, abs_metrics: Dict) -> Dict:
        """Average metrics when both simulators should be weighted equally."""
        consensus = {}
        for key in ["avg_cycle_time", "avg_throughput", "avg_waiting_time", "total_cost"]:
            if key in des_metrics and key in abs_metrics:
                consensus[key] = (des_metrics[key] + abs_metrics[key]) / 2
        return consensus

    @staticmethod
    def _weighted_consensus_metrics(
        des_metrics: Dict,
        abs_metrics: Dict,
        des_weight: float,
        abs_weight: float
    ) -> Dict:
        """Weighted combination of metrics when one simulator has lower signal quality."""
        weighted = {}
        total_weight = max(1e-9, des_weight + abs_weight)
        for key in ["avg_cycle_time", "avg_throughput", "avg_waiting_time", "total_cost"]:
            if key in des_metrics and key in abs_metrics:
                weighted[key] = (
                    des_metrics[key] * des_weight + abs_metrics[key] * abs_weight
                ) / total_weight
            elif key in des_metrics:
                weighted[key] = des_metrics[key]
            elif key in abs_metrics:
                weighted[key] = abs_metrics[key]
        return weighted
    
    @staticmethod
    def _check_red_flags(
        des_metrics: Dict,
        abs_metrics: Dict,
        divergence: Dict,
        risk_score: float,
        risk_level: str
    ) -> list:
        """Check for warnings: bottlenecks, negative impacts, or high uncertainty."""
        warnings = []
        
        # Red flag 1: Either simulator predicts collapse (cycle time > 2x baseline)
        if des_metrics.get("avg_cycle_time", 0) > 100000:
            warnings.append("⚠️ STRUCTURAL RISK: Prosimos predicts severe bottleneck (cycle time spike)")
        
        if abs_metrics.get("avg_cycle_time", 0) > 100000:
            warnings.append("⚠️ BEHAVIORAL RISK: AgentSimulator predicts resource starvation (cycle time spike)")
        
        # Red flag 2: Large divergence (>10%) on critical metrics
        for key, div in divergence.items():
            if isinstance(div, dict) and div.get("is_significant"):
                if abs(div.get("delta_pct", 0)) > 10:
                    warnings.append(f"⚠️ HIGH VARIANCE: {key} differs by {div.get('delta_pct', 0):.1f}% between models")
        
        # Red flag 3: Throughput collapse
        if des_metrics.get("avg_throughput", 0) < 10 or abs_metrics.get("avg_throughput", 0) < 10:
            warnings.append("⚠️ THROUGHPUT COLLAPSE: Simulations show significant throughput reduction")

        if risk_score >= 0.6:
            warnings.append(f"⚠️ HIGH RISK: model divergence score {risk_score:.2f} ({risk_level})")
        elif risk_score >= 0.4:
            warnings.append(f"⚠️ MODERATE RISK: model divergence score {risk_score:.2f} ({risk_level})")
        
        return warnings
    
    @staticmethod
    def _generate_insight(divergence: Dict, hyp_type: HypothesisType, warnings: list) -> str:
        """Generate human-readable insight about the divergence."""
        if warnings:
            return f"Caution advised. {divergence['gap_summary']} Recommendation should be validated with domain experts."
        
        if divergence.get("gap_summary") == "Both models agree":
            return f"Strong consistency: Both structural and behavioral simulations confirm the improvement."
        
        # Divergence exists but not critical
        if hyp_type == HypothesisType.RESOURCE_ALLOCATION:
            return "Note: Agent-based model captures specific agent behavior. AgentSimulator provides more realistic resource impact."
        elif hyp_type == HypothesisType.SOCIAL_HANDOVER:
            return "Caution: Behavioral interactions (handover delays, skills) are critical. AgentSimulator's result is more reliable."
        elif hyp_type == HypothesisType.CONTROL_FLOW:
            return "Structural model (Prosimos) shows clear improvement. Check if agents can actually execute the new logic."
        
        return "Results show improvement. Verify feasibility with process stakeholders."
