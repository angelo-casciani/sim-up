"""
Fault Injector for Quantitative Evaluation.

This module injects known faults into simulation models by reducing resource counts.
The goal is to create M_fault from M_opt (the optimal/discovered model) and measure
how quickly different search strategies (LLM vs Grid) can recover the optimal configuration.
"""

import copy
import re
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


class FaultInjector:
    """Inject resource faults into simulation models for controlled experiments."""

    def __init__(self, fault_resource_count: int = 1):
        """
        Initialize fault injector.

        Args:
            fault_resource_count: The count to set for the faulty resource (default: 1)
        """
        self.fault_resource_count = fault_resource_count

    def _extract_role_name(self, resource_id: str) -> str:
        """
        Extract role name from resource ID.

        Examples:
            "Loan Officer-000001_profile" -> "Loan Officer"
            "Clerk-000003" -> "Clerk"
            "AML Investigator-000002_profile" -> "AML Investigator"
        """
        # Remove _profile suffix
        name = resource_id.replace("_profile", "")
        # Remove numeric suffix like -000001, -000002
        name = re.sub(r'-\d+$', '', name)
        return name.strip()

    def _group_resources_by_role(self, model: Dict) -> Dict[str, List[Dict]]:
        """
        Group resource profiles by their role name.

        Returns dict: role_name -> list of profile dicts
        """
        resource_profiles = model.get("prosimos_params", {}).get("resource_profiles", [])
        roles = defaultdict(list)

        for profile in resource_profiles:
            profile_id = profile.get("id", profile.get("name", ""))
            role_name = self._extract_role_name(profile_id)
            roles[role_name].append(profile)

        return dict(roles)

    def find_best_resource_for_fault(
        self,
        baseline_metrics: Dict,
        model: Dict,
        min_count: int = 2
    ) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        """
        Find the best ROLE for fault injection.

        CRITICAL: For a meaningful experiment, we need a role with multiple resources
        so that reducing it creates a real bottleneck.

        Selection criteria:
        1. Role must have >= min_count resources (profiles)
        2. Prefer higher aggregate utilization
        3. Prefer higher resource count (more dramatic fault effect)

        Args:
            baseline_metrics: Baseline metrics containing resource_utilization
            model: The simulation model with prosimos_params
            min_count: Minimum resource count required (default: 2)

        Returns:
            Tuple of (role_name, original_count, avg_utilization)
        """
        roles = self._group_resources_by_role(model)
        utilization = baseline_metrics.get("resource_utilization", {})

        if not roles:
            return None, None, None

        # Build candidates: roles with enough resources
        candidates = []
        for role_name, profiles in roles.items():
            role_count = len(profiles)

            if role_count < min_count:
                continue

            # Calculate average utilization for this role
            role_utils = []
            for profile in profiles:
                profile_id = profile.get("id", profile.get("name", ""))
                # Find utilization matching this profile
                for util_name, util_val in utilization.items():
                    util_norm = util_name.replace("_profile", "")
                    profile_norm = profile_id.replace("_profile", "")
                    if util_norm == profile_norm or profile_norm in util_name:
                        role_utils.append(float(util_val))
                        break

            avg_util = sum(role_utils) / len(role_utils) if role_utils else 0.5

            candidates.append({
                "role_name": role_name,
                "count": role_count,
                "utilization": avg_util,
                "profiles": profiles,
                # Score: prefer high utilization AND high count
                "score": avg_util * 0.5 + (role_count / 10) * 0.5
            })

        if not candidates:
            print(f"  [!] No roles with count >= {min_count}")
            # List what we have
            for role_name, profiles in roles.items():
                print(f"      {role_name}: {len(profiles)} resources")
            return None, None, None

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        print(f"  [OK] Selected ROLE for fault injection:")
        print(f"       {best['role_name']} ({best['count']} resources, avg util={best['utilization']:.2%})")
        if len(candidates) > 1:
            print(f"       Other candidates: {[(c['role_name'], c['count']) for c in candidates[1:3]]}")

        return best["role_name"], best["count"], best["utilization"]

    def find_most_utilized_resource(
        self,
        baseline_metrics: Dict,
        model: Dict
    ) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        """Deprecated: Use find_best_resource_for_fault() instead."""
        return self.find_best_resource_for_fault(baseline_metrics, model, min_count=2)

    def inject_fault(
        self,
        model: Dict,
        target_resource: str,
        fault_count: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Inject a fault by reducing resources for a role.

        For logs with individual resources (Clerk-000001, Clerk-000002, etc.),
        we REMOVE profiles to simulate reducing the role's capacity.

        Args:
            model: The original (optimal) simulation model
            target_resource: Role name (e.g., "Clerk", "Loan Officer")
            fault_count: Number of resources to KEEP (default: self.fault_resource_count)

        Returns:
            Tuple of (faulty_model, fault_info)
        """
        if fault_count is None:
            fault_count = self.fault_resource_count

        faulty_model = copy.deepcopy(model)
        resource_profiles = faulty_model.get("prosimos_params", {}).get("resource_profiles", [])

        # Find all profiles belonging to this role
        role_profiles = []
        other_profiles = []

        for profile in resource_profiles:
            profile_id = profile.get("id", profile.get("name", ""))
            role_name = self._extract_role_name(profile_id)

            if role_name == target_resource:
                role_profiles.append(profile)
            else:
                other_profiles.append(profile)

        original_count = len(role_profiles)

        if original_count == 0:
            raise ValueError(f"Role '{target_resource}' not found in model")

        # Keep only fault_count profiles, remove the rest
        kept_profiles = role_profiles[:fault_count]
        removed_profiles = role_profiles[fault_count:]

        # Update model with reduced profiles
        faulty_model["prosimos_params"]["resource_profiles"] = other_profiles + kept_profiles

        # Also need to update task_resource_distribution to remove references to removed resources
        self._update_task_distributions(faulty_model, removed_profiles)

        fault_info = {
            "target_resource": target_resource,
            "original_count": original_count,
            "fault_count": fault_count,
            "kept_profiles": [p.get("id", p.get("name", "")) for p in kept_profiles],
            "removed_profiles": [p.get("id", p.get("name", "")) for p in removed_profiles],
            "expected_fix_count": original_count
        }

        print(f"    Removed {len(removed_profiles)} of {original_count} '{target_resource}' resources")

        return faulty_model, fault_info

    def _update_task_distributions(self, model: Dict, removed_profiles: List[Dict]):
        """Remove references to removed profiles from task distributions."""
        removed_ids = {
            p.get("id", p.get("name", "")).replace("_profile", "")
            for p in removed_profiles
        }

        task_distributions = model.get("prosimos_params", {}).get("task_resource_distribution", [])

        for task in task_distributions:
            resources = task.get("resources", [])
            if resources:
                filtered = []
                for res in resources:
                    res_id = res.get("resource_id", res.get("id", "")).replace("_profile", "")
                    if res_id not in removed_ids:
                        filtered.append(res)
                task["resources"] = filtered

    def get_all_resources(self, model: Dict) -> List[Dict]:
        """Get all resources grouped by role."""
        roles = self._group_resources_by_role(model)

        result = []
        for role_name, profiles in roles.items():
            result.append({
                "role_name": role_name,
                "count": len(profiles),
                "profile_ids": [p.get("id", p.get("name", "")) for p in profiles]
            })

        return result
