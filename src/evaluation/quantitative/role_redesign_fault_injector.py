"""
Role Redesign Fault Injector for Quantitative Evaluation.

This module injects known faults into simulation models by creating workload imbalances.
The goal is to create M_fault from M_opt (the optimal/discovered model) and measure
how quickly different search strategies (LLM vs Heuristic) can recover optimal task distribution.

Fault Injection Strategy:
- Identify two roles with overlapping activities (can perform same tasks)
- Modify task_resource_distribution to concentrate work on one role (creating overload)
- Ground truth: LLM should propose shifting responsibility back to the underloaded role

Role Redesign Search Space (Combinatorial):
- N source roles × (N-1) target roles × M shift percentages
- For LoanApp with 6 roles and 5 shift levels: 6 × 5 × 5 = 150 combinations
"""

import copy
import re
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict


class RoleRedesignFaultInjector:
    """Inject role workload imbalance faults into simulation models for controlled experiments."""

    def __init__(self, fault_shift_fraction: float = 0.5):
        """
        Initialize role redesign fault injector.

        Args:
            fault_shift_fraction: Fraction of work to shift to overload a role (0.0-1.0)
                                  e.g., 0.5 means shift 50% of target's work away
        """
        self.fault_shift_fraction = fault_shift_fraction

    def _extract_role_name(self, resource_id: str) -> str:
        """
        Extract role name from resource ID.

        Examples:
            "Loan Officer-000001_profile" -> "Loan Officer"
            "Clerk-000003" -> "Clerk"
        """
        name = resource_id.replace("_profile", "").replace("_calendar", "")
        name = re.sub(r'-\d+$', '', name)
        return name.strip()

    def _get_resource_activities(self, model: Dict) -> Dict[str, Set[str]]:
        """
        Get the set of activities each resource can perform.

        Returns dict: resource_id -> set of activity names
        """
        resource_activities = defaultdict(set)
        task_distributions = model.get("prosimos_params", {}).get("task_resource_distribution", [])

        for task in task_distributions:
            task_id = task.get("task_id", "")
            resources = task.get("resources", [])
            for res in resources:
                res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                resource_activities[res_id].add(task_id)

        return dict(resource_activities)

    def _get_role_activities(self, model: Dict) -> Dict[str, Set[str]]:
        """
        Get the set of activities each role can perform (aggregated across resources).

        Returns dict: role_name -> set of activity names
        """
        resource_activities = self._get_resource_activities(model)
        role_activities = defaultdict(set)

        for res_id, activities in resource_activities.items():
            role = self._extract_role_name(res_id)
            role_activities[role].update(activities)

        return dict(role_activities)

    def find_overlapping_roles(self, model: Dict) -> List[Tuple[str, str, Set[str]]]:
        """
        Find pairs of roles that share overlapping activities.

        Returns list of (role1, role2, shared_activities) tuples.
        """
        role_activities = self._get_role_activities(model)
        overlaps = []

        roles = list(role_activities.keys())
        for i, role1 in enumerate(roles):
            for role2 in roles[i + 1:]:
                shared = role_activities[role1] & role_activities[role2]
                if shared:
                    overlaps.append((role1, role2, shared))

        return overlaps

    def find_best_roles_for_fault(
        self,
        baseline_metrics: Dict,
        model: Dict,
        min_overlap_activities: int = 1
    ) -> Tuple[Optional[str], Optional[str], Optional[Set[str]], Optional[Dict]]:
        """
        Find the best role pair for fault injection.

        Selection criteria:
        1. Roles must have overlapping activities
        2. PRIMARY: Large utilization difference (overload → underload)
        3. SECONDARY: More overlapping activities (more flexibility)

        Args:
            baseline_metrics: Baseline metrics containing resource_utilization
            model: The simulation model with prosimos_params
            min_overlap_activities: Minimum shared activities required

        Returns:
            Tuple of (overloaded_role, underloaded_role, shared_activities, role_info)
        """
        overlaps = self.find_overlapping_roles(model)
        utilization = baseline_metrics.get("resource_utilization", {})

        if not overlaps:
            print("  [!] No overlapping role pairs found in model")
            return None, None, None, None

        # Calculate average utilization per role
        role_utils = self._calculate_role_utilization(model, utilization)

        candidates = []
        for role1, role2, shared in overlaps:
            if len(shared) < min_overlap_activities:
                continue

            util1 = role_utils.get(role1, 0.5)
            util2 = role_utils.get(role2, 0.5)

            # Determine which role is higher utilized
            if util1 >= util2:
                high_role, low_role = role1, role2
                high_util, low_util = util1, util2
            else:
                high_role, low_role = role2, role1
                high_util, low_util = util2, util1

            util_diff = high_util - low_util

            # Score: prioritize utilization difference, then overlap size
            score = util_diff * 100 + len(shared)

            candidates.append({
                "high_role": high_role,
                "low_role": low_role,
                "high_util": high_util,
                "low_util": low_util,
                "util_diff": util_diff,
                "shared_activities": shared,
                "score": score
            })

        if not candidates:
            print(f"  [!] No role pairs with >= {min_overlap_activities} shared activities found")
            return None, None, None, None

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        print(f"  [OK] Selected role pair for fault injection:")
        print(f"       High util: {best['high_role']} ({best['high_util']:.2%})")
        print(f"       Low util:  {best['low_role']} ({best['low_util']:.2%})")
        print(f"       Shared activities: {len(best['shared_activities'])} ({list(best['shared_activities'])[:3]}...)")

        if len(candidates) > 1:
            print(f"       Other candidates: {[(c['high_role'], c['low_role']) for c in candidates[1:3]]}")

        return (
            best["high_role"],
            best["low_role"],
            best["shared_activities"],
            {
                "high_util": best["high_util"],
                "low_util": best["low_util"],
                "util_diff": best["util_diff"]
            }
        )

    def _calculate_role_utilization(self, model: Dict, utilization: Dict) -> Dict[str, float]:
        """Calculate average utilization per role."""
        role_utils = defaultdict(list)

        for util_name, util_val in utilization.items():
            # Extract role from utilization key
            util_norm = util_name.replace("_profile", "").replace("_calendar", "")
            role = self._extract_role_name(util_norm)
            role_utils[role].append(float(util_val))

        # Average across resources in each role
        return {
            role: sum(vals) / len(vals) if vals else 0.5
            for role, vals in role_utils.items()
        }

    def inject_fault(
        self,
        model: Dict,
        source_role: str,
        target_role: str,
        shift_fraction: Optional[float] = None,
        shared_activities: Optional[Set[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Inject a fault by shifting work FROM target_role TO source_role.

        This creates an imbalance where source_role becomes overloaded
        and target_role becomes underutilized.

        Args:
            model: The original (optimal) simulation model
            source_role: Role to OVERLOAD (shift work TO this role)
            target_role: Role to UNDERLOAD (shift work FROM this role)
            shift_fraction: Fraction of target's work to shift (default: self.fault_shift_fraction)
            shared_activities: Activities they share (will be detected if not provided)

        Returns:
            Tuple of (faulty_model, fault_info)
        """
        if shift_fraction is None:
            shift_fraction = self.fault_shift_fraction

        faulty_model = copy.deepcopy(model)
        task_distributions = faulty_model.get("prosimos_params", {}).get("task_resource_distribution", [])

        # Find shared activities if not provided
        if shared_activities is None:
            role_activities = self._get_role_activities(model)
            source_activities = role_activities.get(source_role, set())
            target_activities = role_activities.get(target_role, set())
            shared_activities = source_activities & target_activities

        if not shared_activities:
            raise ValueError(f"No shared activities between {source_role} and {target_role}")

        # Track modifications
        modified_tasks = []
        total_weight_shifted = 0.0

        for task in task_distributions:
            task_id = task.get("task_id", "")

            # Only modify shared activities
            if task_id not in shared_activities:
                continue

            resources = task.get("resources", [])
            if not resources:
                continue

            # Find source and target resources in this task
            source_resources = []
            target_resources = []
            other_resources = []

            for res in resources:
                res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                role = self._extract_role_name(res_id)

                if role.lower() == source_role.lower():
                    source_resources.append(res)
                elif role.lower() == target_role.lower():
                    target_resources.append(res)
                else:
                    other_resources.append(res)

            # Only modify if both roles are present in this task
            if not source_resources or not target_resources:
                continue

            # Calculate total target weight to shift
            target_total_weight = sum(
                float(r.get("distribution_weight", r.get("weight", 1.0)))
                for r in target_resources
            )

            weight_to_shift = target_total_weight * shift_fraction

            # Reduce target weights
            for res in target_resources:
                old_weight = float(res.get("distribution_weight", res.get("weight", 1.0)))
                new_weight = old_weight * (1 - shift_fraction)
                new_weight = max(0.01, new_weight)  # Keep minimal weight

                if "distribution_weight" in res:
                    res["distribution_weight"] = new_weight
                else:
                    res["weight"] = new_weight

            # Increase source weights proportionally
            source_total_weight = sum(
                float(r.get("distribution_weight", r.get("weight", 1.0)))
                for r in source_resources
            )

            for res in source_resources:
                old_weight = float(res.get("distribution_weight", res.get("weight", 1.0)))
                proportion = old_weight / source_total_weight if source_total_weight > 0 else 1.0 / len(source_resources)
                new_weight = old_weight + (weight_to_shift * proportion)

                if "distribution_weight" in res:
                    res["distribution_weight"] = new_weight
                else:
                    res["weight"] = new_weight

            modified_tasks.append(task_id)
            total_weight_shifted += weight_to_shift

        if not modified_tasks:
            raise ValueError(f"Could not modify any tasks for {source_role} <- {target_role}")

        fault_info = {
            "source_role": source_role,  # Now overloaded
            "target_role": target_role,  # Now underloaded
            "shift_fraction": shift_fraction,
            "modified_tasks": modified_tasks,
            "shared_activities": list(shared_activities),
            "total_weight_shifted": total_weight_shifted,
            "expected_fix": {
                "from_role": source_role,
                "to_role": target_role,
                "shift_fraction": shift_fraction
            }
        }

        print(f"    Shifted {shift_fraction:.0%} of work from {target_role} to {source_role}")
        print(f"    Modified {len(modified_tasks)} tasks: {modified_tasks[:5]}{'...' if len(modified_tasks) > 5 else ''}")

        return faulty_model, fault_info

    def apply_role_shift(
        self,
        model: Dict,
        from_role: str,
        to_role: str,
        shift_fraction: float
    ) -> Dict:
        """
        Apply a role shift modification (shift work FROM from_role TO to_role).

        This is the "fix" operation - shifting work away from overloaded role.

        Args:
            model: Model with workload imbalance
            from_role: Role to shift work FROM (overloaded)
            to_role: Role to shift work TO (underloaded)
            shift_fraction: Fraction of work to shift

        Returns:
            Modified model with rebalanced workload
        """
        modified_model = copy.deepcopy(model)
        task_distributions = modified_model.get("prosimos_params", {}).get("task_resource_distribution", [])

        # Find shared activities
        role_activities = self._get_role_activities(model)
        from_activities = role_activities.get(from_role, set())
        to_activities = role_activities.get(to_role, set())
        shared_activities = from_activities & to_activities

        for task in task_distributions:
            task_id = task.get("task_id", "")

            if task_id not in shared_activities:
                continue

            resources = task.get("resources", [])
            if not resources:
                continue

            from_resources = []
            to_resources = []

            for res in resources:
                res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                role = self._extract_role_name(res_id)

                if role.lower() == from_role.lower():
                    from_resources.append(res)
                elif role.lower() == to_role.lower():
                    to_resources.append(res)

            if not from_resources or not to_resources:
                continue

            # Calculate weight to shift
            from_total_weight = sum(
                float(r.get("distribution_weight", r.get("weight", 1.0)))
                for r in from_resources
            )

            weight_to_shift = from_total_weight * shift_fraction

            # Reduce from_role weights
            for res in from_resources:
                old_weight = float(res.get("distribution_weight", res.get("weight", 1.0)))
                new_weight = old_weight * (1 - shift_fraction)
                new_weight = max(0.01, new_weight)

                if "distribution_weight" in res:
                    res["distribution_weight"] = new_weight
                else:
                    res["weight"] = new_weight

            # Increase to_role weights
            to_total_weight = sum(
                float(r.get("distribution_weight", r.get("weight", 1.0)))
                for r in to_resources
            )

            for res in to_resources:
                old_weight = float(res.get("distribution_weight", res.get("weight", 1.0)))
                proportion = old_weight / to_total_weight if to_total_weight > 0 else 1.0 / len(to_resources)
                new_weight = old_weight + (weight_to_shift * proportion)

                if "distribution_weight" in res:
                    res["distribution_weight"] = new_weight
                else:
                    res["weight"] = new_weight

        return modified_model

    def get_all_roles(self, model: Dict) -> List[Dict]:
        """Get all roles with their activity information."""
        role_activities = self._get_role_activities(model)

        results = []
        for role, activities in role_activities.items():
            results.append({
                "role_name": role,
                "num_activities": len(activities),
                "activities": list(activities)
            })

        return sorted(results, key=lambda x: x["num_activities"], reverse=True)
