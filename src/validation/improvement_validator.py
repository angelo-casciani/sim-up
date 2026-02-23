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
        max_cycle_time_increase_pct: float = 10.0,
        max_regressions: int = 0,
        require_throughput_non_decrease: bool = True
    ) -> Dict:
        """
        Compare baseline and new metrics to validate improvement.
        
        Args:
            baseline_metrics: Baseline process metrics
            new_metrics: Metrics after applying hypothesis (unified)
            hypothesis: The improvement hypothesis
            threshold: Percentage change threshold for significance
            
        Returns:
            Dict with is_improvement bool and detailed comparison
        """
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
            max_cycle_time_increase_pct,
            max_regressions,
            require_throughput_non_decrease,
            baseline_metrics
        )
        
        result = {
            "is_improvement": is_improvement,
            "improvements": improvements,
            "summary": summary
        }

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
        max_cycle_time_increase_pct: float,
        max_regressions: int,
        require_throughput_non_decrease: bool,
        baseline_metrics: Dict = None
    ) -> tuple:
        """Assess if improvements are significant."""
        is_improvement = False
        summary_parts = []
        guardrail_issues = []
        improvement_count = 0
        regression_count = 0
        hard_fail = False
        
        def format_change(change):
            return f"{abs(change):.1f}%"
        
        # Cycle time should decrease
        if "cycle_time" in improvements:
            ct_change = improvements["cycle_time"]["change_pct"]
            if ct_change < -threshold:
                improvement_count += 1
                range_str = format_change(ct_change)
                summary_parts.append(f"Cycle time reduced by {range_str}")
            if ct_change > max_cycle_time_increase_pct:
                hard_fail = True
                guardrail_issues.append(
                    f"cycle time increased by {ct_change:.1f}% (hard limit {max_cycle_time_increase_pct:.1f}%)"
                )
            if ct_change > max_regression_pct:
                regression_count += 1
                if ct_change <= max_cycle_time_increase_pct:
                    guardrail_issues.append(f"cycle time increased by {ct_change:.1f}%")
        
        # Throughput should increase
        if "throughput" in improvements:
            tp_change = improvements["throughput"]["change_pct"]
            if tp_change > threshold:
                improvement_count += 1
                range_str = format_change(tp_change)
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
                range_str = format_change(wt_change)
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
