"""Improvement validation module."""

from typing import Dict


class ImprovementValidator:
    """Validates if improvements are significant."""
    
    def validate(
        self,
        baseline_metrics: Dict,
        new_metrics: Dict,
        hypothesis: Dict,
        threshold: float = 2.0,
        min_improvements: int = 2,
        max_regression_pct: float = 5.0,
        max_regressions: int = 0,
        require_throughput_non_decrease: bool = True,
        des_metrics: Dict = None,
        abs_metrics: Dict = None,
        check_agreement: bool = True,
        agreement_threshold: float = 10.0,
        arbitration_result: Dict = None
    ) -> Dict:
        """
        Compare baseline and new metrics to validate improvement.
        
        Args:
            baseline_metrics: Baseline process metrics
            new_metrics: Metrics after applying hypothesis (unified)
            hypothesis: The improvement hypothesis
            threshold: Percentage change threshold for significance
            des_metrics: DES-specific metrics for agreement checking
            abs_metrics: ABS-specific metrics for agreement checking
            check_agreement: If True, verify DES/ABS directional agreement
            agreement_threshold: Threshold (%) for considering directions aligned
            arbitration_result: Result from EnsembleArbiter with trust strategy
            
        Returns:
            Dict with is_improvement bool and detailed comparison
        """
        # Use trust-based validation from arbitration
        # Only require strict agreement when arbitration_strategy is "consensus"
        require_strict_agreement = False
        if arbitration_result:
            arbitration_strategy = arbitration_result.get('arbitration_strategy', 'consensus')
            if arbitration_strategy == 'consensus':
                require_strict_agreement = True
            # For 'des' or 'abs' strategies, trust the primary simulator
        else:
            # Fallback: require agreement if check_agreement is True
            require_strict_agreement = check_agreement
        
        # Check DES/ABS agreement only if strict agreement is required
        if require_strict_agreement and des_metrics and abs_metrics:
            agreement_result = self._check_simulator_agreement(
                baseline_metrics,
                des_metrics,
                abs_metrics,
                agreement_threshold
            )
            if not agreement_result["agree"]:
                return {
                    "is_improvement": False,
                    "improvements": {},
                    "summary": f"Models contradict: {agreement_result['reason']}",
                    "agreement_check": agreement_result,
                    "arbitration_info": f"Consensus required but simulators disagree"
                }
        
        improvements = {}
        
        # Compare cycle time
        if "avg_cycle_time" in baseline_metrics and "avg_cycle_time" in new_metrics:
            ct_change = self._calculate_change(
                baseline_metrics["avg_cycle_time"],
                new_metrics["avg_cycle_time"]
            )
            improvements["cycle_time"] = {
                "baseline": baseline_metrics["avg_cycle_time"],
                "new": new_metrics["avg_cycle_time"],
                "change_pct": ct_change
            }
        
        # Compare throughput
        if "avg_throughput" in baseline_metrics and "avg_throughput" in new_metrics:
            tp_change = self._calculate_change(
                baseline_metrics["avg_throughput"],
                new_metrics["avg_throughput"]
            )
            improvements["throughput"] = {
                "baseline": baseline_metrics["avg_throughput"],
                "new": new_metrics["avg_throughput"],
                "change_pct": tp_change
            }
        
        # Compare waiting time
        if "avg_waiting_time" in baseline_metrics and "avg_waiting_time" in new_metrics:
            wt_change = self._calculate_change(
                baseline_metrics["avg_waiting_time"],
                new_metrics["avg_waiting_time"]
            )
            improvements["waiting_time"] = {
                "baseline": baseline_metrics["avg_waiting_time"],
                "new": new_metrics["avg_waiting_time"],
                "change_pct": wt_change
            }
        
        # Compare cost
        if "total_cost" in baseline_metrics and "total_cost" in new_metrics:
            cost_change = self._calculate_change(
                baseline_metrics["total_cost"],
                new_metrics["total_cost"]
            )
            improvements["cost"] = {
                "baseline": baseline_metrics["total_cost"],
                "new": new_metrics["total_cost"],
                "change_pct": cost_change
            }
        
        # Determine if improvement
        is_improvement, summary = self._assess_improvement(
            improvements,
            threshold,
            min_improvements,
            max_regression_pct,
            max_regressions,
            require_throughput_non_decrease,
            des_metrics,
            abs_metrics,
            baseline_metrics
        )
        
        result = {
            "is_improvement": is_improvement,
            "improvements": improvements,
            "summary": summary
        }
        
        # Include arbitration info
        if arbitration_result:
            result["arbitration_info"] = f"Trusted {arbitration_result['primary_source']} (strategy: {arbitration_result['arbitration_strategy']})"
        
        # Include agreement info if checked
        if require_strict_agreement and des_metrics and abs_metrics:
            agreement_result = self._check_simulator_agreement(
                baseline_metrics,
                des_metrics,
                abs_metrics,
                agreement_threshold
            )
            result["agreement_check"] = agreement_result
        
        return result
    
    def _calculate_change(self, baseline: float, new: float) -> float:
        """Calculate percentage change from baseline to new."""
        if baseline == 0:
            return 0
        return ((new - baseline) / baseline) * 100
    
    def _assess_improvement(
        self,
        improvements: Dict,
        threshold: float,
        min_improvements: int,
        max_regression_pct: float,
        max_regressions: int,
        require_throughput_non_decrease: bool,
        des_metrics: Dict = None,
        abs_metrics: Dict = None,
        baseline_metrics: Dict = None
    ) -> tuple:
        """Assess if improvements are significant."""
        is_improvement = False
        summary_parts = []
        guardrail_issues = []
        improvement_count = 0
        regression_count = 0
        hard_fail = False
        
        # Helper to build range summary when both simulators are available
        def format_range(metric_key, unified_change):
            if not (des_metrics and abs_metrics and baseline_metrics):
                return f"{abs(unified_change):.1f}%"
            
            baseline_val = baseline_metrics.get(metric_key, 0)
            if baseline_val == 0:
                return f"{abs(unified_change):.1f}%"
            
            des_val = des_metrics.get(metric_key, baseline_val)
            abs_val = abs_metrics.get(metric_key, baseline_val)
            
            des_change = ((des_val - baseline_val) / baseline_val) * 100
            abs_change = ((abs_val - baseline_val) / baseline_val) * 100
            
            # If close, just report unified
            if abs(des_change - abs_change) < 3:
                return f"{abs(unified_change):.1f}%"
            
            # Report range
            min_change = min(abs(des_change), abs(abs_change))
            max_change = max(abs(des_change), abs(abs_change))
            avg_change = (min_change + max_change) / 2
            return f"{min_change:.1f}-{max_change:.1f}% (avg {avg_change:.1f}%)"
        
        # Cycle time should decrease
        if "cycle_time" in improvements:
            ct_change = improvements["cycle_time"]["change_pct"]
            if ct_change < -threshold:
                improvement_count += 1
                range_str = format_range("avg_cycle_time", ct_change)
                summary_parts.append(f"Cycle time reduced by {range_str}")
            if ct_change > max_regression_pct:
                regression_count += 1
                guardrail_issues.append(f"cycle time increased by {ct_change:.1f}%")
        
        # Throughput should increase
        if "throughput" in improvements:
            tp_change = improvements["throughput"]["change_pct"]
            if tp_change > threshold:
                improvement_count += 1
                range_str = format_range("avg_throughput", tp_change)
                summary_parts.append(f"Throughput increased by {range_str}")
            if require_throughput_non_decrease and tp_change < 0:
                hard_fail = True
                guardrail_issues.append(f"throughput decreased by {abs(tp_change):.1f}%")
            if tp_change < -max_regression_pct:
                regression_count += 1
                if not (require_throughput_non_decrease and tp_change < 0):
                    guardrail_issues.append(f"throughput decreased by {abs(tp_change):.1f}%")
        
        # Waiting time should decrease
        if "waiting_time" in improvements:
            wt_change = improvements["waiting_time"]["change_pct"]
            if wt_change < -threshold:
                improvement_count += 1
                range_str = format_range("avg_waiting_time", wt_change)
                summary_parts.append(f"Waiting time reduced by {range_str}")
            if wt_change > max_regression_pct:
                regression_count += 1
                guardrail_issues.append(f"waiting time increased by {wt_change:.1f}%")
        
        # Cost should decrease (but not required)
        if "cost" in improvements:
            cost_change = improvements["cost"]["change_pct"]
            if cost_change < -threshold and not summary_parts:
                improvement_count += 1
                summary_parts.append(f"Cost reduced by {abs(cost_change):.1f}%")

        if hard_fail:
            is_improvement = False
        elif improvement_count >= min_improvements and regression_count <= max_regressions:
            is_improvement = True
        else:
            is_improvement = False

        if is_improvement:
            summary = "; ".join(summary_parts) if summary_parts else "No significant improvement"
        elif guardrail_issues:
            summary = f"Guardrail failed: {', '.join(guardrail_issues)}"
        else:
            summary = "No significant improvement"

        return is_improvement, summary

    def _check_simulator_agreement(
        self,
        baseline_metrics: Dict,
        des_metrics: Dict,
        abs_metrics: Dict,
        agreement_threshold: float = 10.0
    ) -> Dict:
        """
        Check if DES and ABS agree on improvement direction.
        
        Returns dict with:
        - agree: bool (True if no contradiction)
        - reason: str (explanation)
        - details: dict (per-metric agreement status)
        """
        key_metrics = ["avg_cycle_time", "avg_throughput", "avg_waiting_time"]
        metric_names = {"avg_cycle_time": "cycle_time", "avg_throughput": "throughput", "avg_waiting_time": "waiting_time"}
        
        details = {}
        contradictions = []
        agreements = []
        
        for metric_key in key_metrics:
            if metric_key not in baseline_metrics:
                continue
                
            baseline_val = baseline_metrics.get(metric_key, 0)
            if baseline_val == 0:
                continue
            
            des_val = des_metrics.get(metric_key)
            abs_val = abs_metrics.get(metric_key)
            
            if des_val is None or abs_val is None:
                details[metric_key] = {"status": "skipped", "reason": "missing data"}
                continue
            
            des_change = ((des_val - baseline_val) / baseline_val) * 100
            abs_change = ((abs_val - baseline_val) / baseline_val) * 100
            
            # Determine direction: positive change (increase), negative (decrease), or neutral
            des_direction = "increase" if des_change > agreement_threshold else ("decrease" if des_change < -agreement_threshold else "neutral")
            abs_direction = "increase" if abs_change > agreement_threshold else ("decrease" if abs_change < -agreement_threshold else "neutral")
            
            metric_name = metric_names.get(metric_key, metric_key)
            
            # Check agreement
            if des_direction == abs_direction:
                details[metric_key] = {
                    "status": "agree",
                    "des_change": des_change,
                    "abs_change": abs_change,
                    "direction": des_direction
                }
                agreements.append(metric_name)
            else:
                # Contradiction: different directions
                details[metric_key] = {
                    "status": "contradict",
                    "des_change": des_change,
                    "abs_change": abs_change,
                    "des_direction": des_direction,
                    "abs_direction": abs_direction
                }
                contradictions.append(f"{metric_name} (DES: {des_direction}, ABS: {abs_direction})")
        
        # Overall agreement: all 3 metrics must agree
        # If we have contradictions, reject
        if contradictions:
            return {
                "agree": False,
                "reason": f"Simulators contradict on {', '.join(contradictions)}",
                "details": details,
                "agreement_count": len(agreements),
                "contradiction_count": len(contradictions)
            }
        
        # All checked metrics agree
        return {
            "agree": True,
            "reason": f"Both simulators agree on all metrics ({len(agreements)} checked)",
            "details": details,
            "agreement_count": len(agreements),
            "contradiction_count": 0
        }
