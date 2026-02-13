"""Improvement validation module."""

from typing import Dict


class ImprovementValidator:
    """Validates if improvements are significant."""
    
    def validate(
        self,
        baseline_metrics: Dict,
        new_metrics: Dict,
        hypothesis: Dict,
        threshold: float = 2.0
    ) -> Dict:
        """
        Compare baseline and new metrics to validate improvement.
        
        Args:
            baseline_metrics: Baseline process metrics
            new_metrics: Metrics after applying hypothesis
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
        is_improvement, summary = self._assess_improvement(improvements, threshold)
        
        return {
            "is_improvement": is_improvement,
            "improvements": improvements,
            "summary": summary
        }
    
    def _calculate_change(self, baseline: float, new: float) -> float:
        """Calculate percentage change from baseline to new."""
        if baseline == 0:
            return 0
        return ((new - baseline) / baseline) * 100
    
    def _assess_improvement(self, improvements: Dict, threshold: float) -> tuple:
        """Assess if improvements are significant."""
        is_improvement = False
        summary_parts = []
        
        # Cycle time should decrease
        if "cycle_time" in improvements:
            ct_change = improvements["cycle_time"]["change_pct"]
            if ct_change < -threshold:
                is_improvement = True
                summary_parts.append(f"Cycle time reduced by {abs(ct_change):.1f}%")
        
        # Throughput should increase
        if "throughput" in improvements:
            tp_change = improvements["throughput"]["change_pct"]
            if tp_change > threshold:
                is_improvement = True
                summary_parts.append(f"Throughput increased by {tp_change:.1f}%")
        
        # Waiting time should decrease
        if "waiting_time" in improvements:
            wt_change = improvements["waiting_time"]["change_pct"]
            if wt_change < -threshold:
                is_improvement = True
                summary_parts.append(f"Waiting time reduced by {abs(wt_change):.1f}%")
        
        # Cost should decrease (but not required)
        if "cost" in improvements:
            cost_change = improvements["cost"]["change_pct"]
            if cost_change < -threshold and not summary_parts:
                is_improvement = True
                summary_parts.append(f"Cost reduced by {abs(cost_change):.1f}%")
        
        summary = "; ".join(summary_parts) if summary_parts else "No significant improvement"
        return is_improvement, summary
