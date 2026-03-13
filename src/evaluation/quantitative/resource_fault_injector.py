"""
Fault Injector for Quantitative Evaluation.

This module injects known faults into simulation models by reducing resource counts.
The goal is to create M_fault from M_opt (the optimal/discovered model) and measure
how quickly different search strategies (LLM vs Grid) can recover the optimal configuration.

Resource Grouping Strategy:
1. Pattern-based: Extract role from naming pattern (e.g., "Clerk-000001" → "Clerk")
2. Activity-based: Group resources that perform identical sets of activities
3. Fallback: Treat all resources as a single pool (__ALL_RESOURCES__)
"""

import copy
import re
from typing import Dict, Tuple, List, Optional, Set, FrozenSet
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

    def _group_resources_by_activity(self, model: Dict) -> Dict[str, List[str]]:
        """
        Group resources by their activity sets (resources performing identical activities).

        This creates "pseudo-roles" for logs where resources don't follow naming patterns
        but multiple resources handle the same tasks.

        Returns dict: pseudo_role_name -> list of resource IDs
        """
        resource_activities = self._get_resource_activities(model)

        # Group by activity set (use frozenset as dict key)
        activity_groups: Dict[FrozenSet[str], List[str]] = defaultdict(list)
        for res_id, activities in resource_activities.items():
            activity_key = frozenset(activities)
            activity_groups[activity_key].append(res_id)

        # Create pseudo-role names based on activities
        pseudo_roles = {}
        for i, (activity_set, resources) in enumerate(activity_groups.items()):
            if len(resources) >= 2:  # Only groups with 2+ resources are useful
                # Create a readable name from activities
                sorted_activities = sorted(activity_set)
                if len(sorted_activities) <= 3:
                    role_name = f"Group_{'+'.join(sorted_activities[:3])}"
                else:
                    role_name = f"Group_{len(sorted_activities)}activities_{i}"
                pseudo_roles[role_name] = resources

        return pseudo_roles

    def _group_resources_by_activity_with_profiles(self, model: Dict) -> Dict[str, List[Dict]]:
        """
        Group resource profiles by their activity sets.

        Returns dict: pseudo_role_name -> list of profile dicts
        """
        pseudo_roles = self._group_resources_by_activity(model)
        resource_profiles = model.get("prosimos_params", {}).get("resource_profiles", [])

        # Map resource IDs to profiles
        id_to_profile = {}
        for profile in resource_profiles:
            profile_id = profile.get("id", profile.get("name", "")).replace("_profile", "")
            id_to_profile[profile_id] = profile

        # Build result with profile objects
        result = {}
        for role_name, resource_ids in pseudo_roles.items():
            profiles = []
            for res_id in resource_ids:
                if res_id in id_to_profile:
                    profiles.append(id_to_profile[res_id])
                else:
                    # Try with _profile suffix
                    profile_id = f"{res_id}_profile"
                    for profile in resource_profiles:
                        if profile.get("id", profile.get("name", "")) == profile_id:
                            profiles.append(profile)
                            break
            if profiles:
                result[role_name] = profiles

        return result

    def find_best_resource_for_fault(
        self,
        baseline_metrics: Dict,
        model: Dict,
        min_count: int = 3
    ) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        """
        Find the best ROLE for fault injection.

        CRITICAL: For a meaningful experiment, we need a role with multiple resources
        so that reducing it creates a real bottleneck. Prefer roles with HIGH resource
        count to maximize the impact of the fault.

        Selection criteria:
        1. Role must have >= min_count resources (profiles)
        2. PRIMARY: Prefer higher resource count (more dramatic fault effect)
        3. SECONDARY: Prefer higher aggregate utilization (tie-breaker)

        FALLBACK: If all roles have only 1-2 resources (individual resources), treat
        all resources as a single pool and return a special "__ALL_RESOURCES__" role.

        Args:
            baseline_metrics: Baseline metrics containing resource_utilization
            model: The simulation model with prosimos_params
            min_count: Minimum resource count required (default: 3)

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
                # Score: PRIMARY is count, SECONDARY is utilization (tie-breaker)
                "score": role_count * 100 + avg_util
            })

        if not candidates:
            # FALLBACK 1: Try activity-based grouping (pseudo-roles)
            print(f"  [!] No pattern-based roles found, trying activity-based grouping...")
            activity_roles = self._group_resources_by_activity_with_profiles(model)

            if activity_roles:
                # Build candidates from activity-based groups
                for role_name, profiles in activity_roles.items():
                    role_count = len(profiles)
                    if role_count < min_count:
                        continue

                    # Calculate average utilization for this group
                    role_utils = []
                    for profile in profiles:
                        profile_id = profile.get("id", profile.get("name", ""))
                        for util_name, util_val in utilization.items():
                            util_norm = util_name.replace("_profile", "")
                            profile_norm = profile_id.replace("_profile", "")
                            if util_norm == profile_norm or profile_norm in util_name:
                                role_utils.append(float(util_val))
                                break

                    avg_util = sum(role_utils) / len(role_utils) if role_utils else 0.5

                    # For activity-based groups, balance count with utilization more evenly
                    # High utilization (>0.7) indicates resources on critical path
                    # Score formula: count matters, but high utilization gets bonus
                    util_bonus = 50 if avg_util > 0.7 else (25 if avg_util > 0.5 else 0)
                    score = role_count * 10 + avg_util * 100 + util_bonus

                    candidates.append({
                        "role_name": role_name,
                        "count": role_count,
                        "utilization": avg_util,
                        "profiles": profiles,
                        "is_activity_based": True,  # Mark as activity-based
                        # Score: balance count and utilization for activity-based groups
                        "score": score
                    })

                if candidates:
                    print(f"  [OK] Found {len(candidates)} activity-based pseudo-roles")
                    # Show top 3 candidates for debugging
                    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                    for i, c in enumerate(sorted_candidates[:3]):
                        print(f"       #{i+1}: {c['role_name']} (count={c['count']}, util={c['utilization']:.1%}, score={c['score']:.1f})")

            # FALLBACK 2: All resources as a single pool
            if not candidates:
                total_resources = sum(len(profiles) for profiles in roles.values())
                if total_resources >= min_count:
                    print(f"  [!] No activity-based groups found, treating all {total_resources} resources as a single pool")

                    # Calculate average utilization across all resources
                    all_utils = list(utilization.values()) if utilization else [0.5]
                    avg_util = sum(float(u) for u in all_utils) / len(all_utils)

                    return "__ALL_RESOURCES__", total_resources, avg_util

                print(f"  [!] Not enough resources for fault injection (have {total_resources}, need {min_count})")
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
        return self.find_best_resource_for_fault(baseline_metrics, model, min_count=3)

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

        Special cases:
        - "__ALL_RESOURCES__": treat all resources as a single pool
        - "Group_*": activity-based pseudo-role (look up by resource ID)

        Args:
            model: The original (optimal) simulation model
            target_resource: Role name (e.g., "Clerk", "Loan Officer"), "__ALL_RESOURCES__", or "Group_*"
            fault_count: Number of resources to KEEP (default: self.fault_resource_count)

        Returns:
            Tuple of (faulty_model, fault_info)
        """
        if fault_count is None:
            fault_count = self.fault_resource_count

        faulty_model = copy.deepcopy(model)
        resource_profiles = faulty_model.get("prosimos_params", {}).get("resource_profiles", [])

        # Special case: treat all resources as a single pool
        if target_resource == "__ALL_RESOURCES__":
            original_count = len(resource_profiles)

            if original_count == 0:
                raise ValueError("No resources found in model")

            # Sort by some criterion (e.g., ID) for deterministic selection
            sorted_profiles = sorted(resource_profiles, key=lambda p: p.get("id", p.get("name", "")))

            # Keep only fault_count profiles, remove the rest
            kept_profiles = sorted_profiles[:fault_count]
            removed_profiles = sorted_profiles[fault_count:]

            # Update model with reduced profiles
            faulty_model["prosimos_params"]["resource_profiles"] = kept_profiles

            # Update task distributions
            self._update_task_distributions(faulty_model, removed_profiles)

            fault_info = {
                "target_resource": target_resource,
                "original_count": original_count,
                "fault_count": fault_count,
                "kept_profiles": [p.get("id", p.get("name", "")) for p in kept_profiles],
                "removed_profiles": [p.get("id", p.get("name", "")) for p in removed_profiles],
                "expected_fix_count": original_count
            }

            print(f"    Removed {len(removed_profiles)} of {original_count} resources (all resources as pool)")

            return faulty_model, fault_info

        # Activity-based pseudo-role (Group_*)
        if target_resource.startswith("Group_"):
            activity_roles = self._group_resources_by_activity_with_profiles(model)

            if target_resource not in activity_roles:
                raise ValueError(f"Activity-based group '{target_resource}' not found in model")

            role_profiles = activity_roles[target_resource]
            original_count = len(role_profiles)

            # Get resource IDs in this group
            group_resource_ids = {
                p.get("id", p.get("name", "")).replace("_profile", "")
                for p in role_profiles
            }

            # Split profiles into group members and others
            other_profiles = []
            actual_role_profiles = []
            for profile in resource_profiles:
                profile_id = profile.get("id", profile.get("name", "")).replace("_profile", "")
                if profile_id in group_resource_ids:
                    actual_role_profiles.append(profile)
                else:
                    other_profiles.append(profile)

            # Keep only fault_count profiles, remove the rest
            kept_profiles = actual_role_profiles[:fault_count]
            removed_profiles = actual_role_profiles[fault_count:]

            # Update model with reduced profiles
            faulty_model["prosimos_params"]["resource_profiles"] = other_profiles + kept_profiles

            # Update task distributions
            self._update_task_distributions(faulty_model, removed_profiles)

            fault_info = {
                "target_resource": target_resource,
                "original_count": original_count,
                "fault_count": fault_count,
                "kept_profiles": [p.get("id", p.get("name", "")) for p in kept_profiles],
                "removed_profiles": [p.get("id", p.get("name", "")) for p in removed_profiles],
                "expected_fix_count": original_count,
                "is_activity_based": True
            }

            print(f"    Removed {len(removed_profiles)} of {original_count} '{target_resource}' resources (activity-based group)")

            return faulty_model, fault_info

        # Normal case: pattern-based role
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
