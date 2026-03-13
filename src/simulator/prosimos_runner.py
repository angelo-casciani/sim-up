import os
import json
import subprocess
import copy
from typing import Dict, Optional
from simulator.metrics_extractor import MetricsExtractor
from utils.column_utils import (
    normalize_role_label,
    parse_role_redesign_target,
    resolve_shift_fraction,
    find_resources_matching_role,
    extract_resource_role,
    calculate_target_distribution,
)
import shutil


class ProsimosRunner:
    def __init__(self, use_sudo: bool = True):
        self.docker_image = "nokal/simod:latest"
        self.use_sudo = use_sudo
        self.work_dir = None
    
    def simulate(
        self,
        model: Dict,
        modifications: Dict,
        work_dir: str,
        total_cases: Optional[int] = None,
        time_horizon_seconds: Optional[float] = None,
        baseline_metrics: Optional[Dict] = None,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Run Prosimos simulation with modifications.

        Args:
            model: Baseline simulation model
            modifications: Parameter modifications from hypothesis
            work_dir: Working directory
            baseline_metrics: Optional baseline metrics with real resources for enrichment
            seed: Optional random seed for reproducibility (ensures identical inputs
                  yield identical outputs for fair benchmarking)

        Returns:
            Dictionary of simulated metrics
        """
        self.work_dir = work_dir

        # Apply modifications to Prosimos parameters
        modified_params = self._apply_modifications(model, modifications)

        # Save modified parameters
        params_file = os.path.join(work_dir, "modified_params.json")
        with open(params_file, 'w') as f:
            json.dump(modified_params, f, indent=2)

        # Run simulation - NO FALLBACK, fail clearly if something goes wrong
        result = self._run_simulation(
            model,
            params_file,
            work_dir,
            total_cases=total_cases,
            time_horizon_seconds=time_horizon_seconds,
            seed=seed
        )

        if not result:
            raise RuntimeError("DES simulation returned empty result - check Prosimos/Docker setup")

        if not result.get('resource_utilization'):
            raise RuntimeError("DES simulation returned no resource utilization data")

        return result
    
    def _apply_modifications(self, model: Dict, modifications: Dict) -> Dict:
        """Apply hypothesis modifications to Prosimos parameters."""
        modified_params = copy.deepcopy(model.get('prosimos_params', {}))
        if not modified_params:
            return modified_params
        param_type = modifications.get('parameter_type')
        specific_changes = modifications.get('specific_changes', [])
        for change in specific_changes:
            target = change.get('target')
            change_type = change.get('change')
            value = change.get('value')
            # Only allow types supported by AgentSimulator and LLM
            if param_type == 'resources':
                self._modify_resources(modified_params, target, change_type, value)
            elif param_type == 'role_redesign':
                self._modify_role_redesign(modified_params, target, change_type, value, change)
            elif param_type == 'activity_durations':
                self._modify_durations(modified_params, target, value)
            elif param_type == 'calendar':
                self._modify_calendar_hours(modified_params, target, change_type, value)
            # Remove support for arrival_rate, gateway_probabilities, and any other types
        return modified_params
    
    def _modify_resources(self, params: Dict, target: str, change_type: str, value: int):
        """Modify resource profiles in Prosimos parameters with role-aware distribution.

        Handles resource modifications by:
        1. Identifying all profiles belonging to the target role (e.g., "Loan Officer")
        2. Distributing the requested change across all matching profiles
        3. Creating new profiles if needed to reach the target count

        This ensures that "increase Loan Officer to 4" correctly handles cases where
        the log has "Loan Officer-000001", "Loan Officer-000002", etc.

        Note: Prosimos resource_profiles structure has:
        - profile['resource_list'][0]['amount'] for the actual resource count
        - profile['amount'] may also exist at profile level (legacy format)
        """
        if 'resource_profiles' not in params:
            return

        resource_profiles = params['resource_profiles']

        # Find all profiles matching the target role
        matching_profiles, matched_role = find_resources_matching_role(
            resource_profiles, target
        )

        if not matching_profiles:
            print(f"  [!] No resources found matching role '{target}'")
            return

        def get_profile_amount(profile: Dict) -> int:
            """Get amount from profile, checking both nested and flat structures."""
            # Check nested structure first (resource_list[0]['amount'])
            resource_list = profile.get('resource_list', [])
            if resource_list and isinstance(resource_list, list) and len(resource_list) > 0:
                return resource_list[0].get('amount', 1)
            # Fallback to flat structure
            return profile.get('amount', 1)

        def set_profile_amount(profile: Dict, new_amount: int):
            """Set amount in profile, handling both nested and flat structures."""
            # Set in nested structure if it exists
            resource_list = profile.get('resource_list', [])
            if resource_list and isinstance(resource_list, list) and len(resource_list) > 0:
                resource_list[0]['amount'] = new_amount
            # Also set at profile level for compatibility
            profile['amount'] = new_amount

        # Calculate current total for this role
        current_total = sum(get_profile_amount(p) for p in matching_profiles)
        n_profiles = len(matching_profiles)

        # Determine target total based on change type
        if 'add' in change_type.lower() or 'increase' in change_type.lower():
            target_total = current_total + int(value)
        elif 'remove' in change_type.lower() or 'decrease' in change_type.lower():
            target_total = max(n_profiles, current_total - int(value))
        elif 'set' in change_type.lower():
            target_total = int(value)
        else:
            target_total = current_total + int(value)  # Default to add

        print(f"  [OK] Role '{matched_role}': {n_profiles} profiles, "
              f"current total={current_total}, target total={target_total}")

        # Distribute evenly across existing profiles
        base_amount = target_total // n_profiles
        remainder = target_total % n_profiles

        for i, profile in enumerate(matching_profiles):
            old_amount = get_profile_amount(profile)
            new_amount = base_amount + (1 if i < remainder else 0)
            set_profile_amount(profile, new_amount)
            res_id = profile.get('id', profile.get('name', ''))
            if old_amount != new_amount:
                print(f"    - {res_id}: {old_amount} -> {new_amount}")

        # If target_total > n_profiles * max_per_profile, we need to create new profiles
        # For now, assume max_per_profile is unbounded (Prosimos handles it)
        # But if we need more profiles (e.g., for better parallelism), create them
        if target_total > n_profiles and n_profiles == 1:
            # Single profile case: create additional profiles to distribute load
            template = matching_profiles[0]
            template_id = template.get('id', template.get('name', ''))
            role_base = extract_resource_role(template_id)

            # Calculate how many new profiles to create
            # Use 1 resource per profile for better load distribution
            profiles_needed = target_total
            profiles_to_create = profiles_needed - n_profiles

            if profiles_to_create > 0:
                # Find the highest existing suffix number for this role
                max_suffix = 0
                for p in resource_profiles:
                    p_id = p.get('id', p.get('name', ''))
                    if extract_resource_role(p_id) == role_base:
                        import re
                        match = re.search(r'-(\d+)$', p_id)
                        if match:
                            max_suffix = max(max_suffix, int(match.group(1)))

                # Update the original profile to have 1 resource
                set_profile_amount(matching_profiles[0], 1)
                print(f"    - {template_id}: adjusted to 1 (redistributing to new profiles)")

                # Get the original resource_id for task_resource_distribution lookup
                original_resource_id = None
                if 'resource_list' in template and template['resource_list']:
                    original_resource_id = template['resource_list'][0].get('id', template['resource_list'][0].get('name'))
                if not original_resource_id:
                    original_resource_id = template_id.replace('_profile', '')

                # Create new profiles with 1 resource each
                created_resource_ids = []
                for i in range(profiles_to_create):
                    new_profile = copy.deepcopy(template)
                    new_suffix = max_suffix + i + 1
                    new_id = f"{role_base}_extra_{new_suffix}"
                    new_resource_id = new_id.replace('_profile', '')

                    # Update profile ID
                    if 'id' in new_profile:
                        new_profile['id'] = new_id
                    if 'name' in new_profile:
                        new_profile['name'] = new_id

                    # Update resource_list if it exists
                    if 'resource_list' in new_profile and new_profile['resource_list']:
                        new_profile['resource_list'][0]['id'] = new_resource_id
                        new_profile['resource_list'][0]['name'] = new_resource_id
                        new_profile['resource_list'][0]['amount'] = 1
                        # Keep the original calendar reference - do NOT rename it
                        # Prosimos needs the calendar to exist in resource_calendars

                    new_profile['amount'] = 1
                    resource_profiles.append(new_profile)
                    created_resource_ids.append(new_resource_id)
                    print(f"    + Created new profile: {new_id} (amount=1)")

                # Add new resources to task_resource_distribution
                # Clone entries from original resource to new resources
                task_distributions = params.get('task_resource_distribution', [])
                tasks_updated = 0
                for task in task_distributions:
                    resources = task.get('resources', [])
                    # Find entries for the original resource
                    original_entries = [r for r in resources
                                       if r.get('resource_id') == original_resource_id or
                                          r.get('resource') == original_resource_id]

                    if original_entries:
                        # Clone the original entry for each new resource
                        for new_res_id in created_resource_ids:
                            new_entry = copy.deepcopy(original_entries[0])
                            if 'resource_id' in new_entry:
                                new_entry['resource_id'] = new_res_id
                            if 'resource' in new_entry:
                                new_entry['resource'] = new_res_id
                            resources.append(new_entry)
                        tasks_updated += 1

                if tasks_updated > 0:
                    print(f"    + Added {len(created_resource_ids)} new resources to {tasks_updated} task distributions")

    def _modify_role_redesign(self, params: Dict, target: str, change_type: str, value: int, change_payload: Dict):
        """Create/reshape roles to model specialization without only adding generic capacity."""
        print(f"  [DEBUG] _modify_role_redesign called: target='{target}', change_type='{change_type}'")

        if 'resource_profiles' not in params or not target:
            print(f"  [!] _modify_role_redesign: missing resource_profiles or target")
            return

        source_role, destination_role = parse_role_redesign_target(target)
        print(f"  [DEBUG] Parsed roles: source='{source_role}', destination='{destination_role}'")

        resource_profiles = params.get('resource_profiles', [])
        matched_profile = None
        for resource in resource_profiles:
            res_id = str(resource.get('id', resource.get('name', '')))
            if source_role and (source_role in res_id or res_id == source_role):
                matched_profile = resource
                break

        if not matched_profile:
            print(f"  [!] _modify_role_redesign: could not find profile for source '{source_role}'")
            return

        if 'shift_responsibility' in str(change_type) and destination_role:
            shift_fraction = resolve_shift_fraction(
                change_payload.get('value')
            )
            print(f"  [DEBUG] Attempting shift_responsibility with fraction={shift_fraction}")
            shifted = self._shift_role_responsibilities(
                params,
                source_role=source_role,
                destination_role=destination_role,
                shift_fraction=shift_fraction
            )
            if shifted:
                print(f"  [OK] shift_responsibility succeeded")
                return
            else:
                print(f"  [!] shift_responsibility returned False, NOT falling back to specialist creation")
                return  # Don't fall back to specialist if shift was explicitly requested but failed

        base_id = str(matched_profile.get('id', matched_profile.get('name', target)))
        specialist_id = f"{base_id}_specialist"

        # Avoid duplicate specialist creation
        existing_ids = {
            str(r.get('id', r.get('name', '')))
            for r in resource_profiles
        }
        if specialist_id in existing_ids:
            return

        specialist_profile = copy.deepcopy(matched_profile)
        if 'id' in specialist_profile:
            specialist_profile['id'] = specialist_id
        elif 'name' in specialist_profile:
            specialist_profile['name'] = specialist_id
        else:
            specialist_profile['id'] = specialist_id

        # Create specialist with minimal added capacity (value interpreted as new count)
        specialist_amount = max(1, int(value) if value is not None else 1)
        specialist_profile['amount'] = specialist_amount
        resource_profiles.append(specialist_profile)

        task_hint = change_payload.get('task_target') or change_payload.get('task')
        self._assign_specialist_to_tasks(params, base_id, specialist_id, task_hint, change_type)

    def _shift_role_responsibilities(
        self,
        params: Dict,
        source_role: str,
        destination_role: str,
        shift_fraction: float = 0.5
    ) -> bool:
        """Redistribute source-role responsibilities to destination with bounded fraction.

        Rebalances assignments globally across task-resource mappings by moving a
        subset of source references to destination references.
        """
        if not source_role or not destination_role:
            print(f"  [!] shift_responsibility: missing source ({source_role}) or destination ({destination_role})")
            return False

        # First, find the actual full resource_id for source and destination from resource_profiles
        resource_profiles = params.get('resource_profiles', [])
        source_full_id = None
        destination_full_id = None

        print(f"  [DEBUG] Looking for source='{source_role}', destination='{destination_role}'")
        profile_ids = [p.get('id', p.get('name', '')) for p in resource_profiles]
        print(f"  [DEBUG] Available {len(profile_ids)} profiles: {profile_ids[:5]}{'...' if len(profile_ids) > 5 else ''}")

        for profile in resource_profiles:
            profile_id = str(profile.get('id', profile.get('name', '')))
            profile_id_norm = profile_id.replace('_profile', '')

            # Match source
            if source_full_id is None:
                if (source_role in profile_id or source_role in profile_id_norm or
                    profile_id == source_role or profile_id_norm == source_role):
                    source_full_id = profile_id
                    print(f"  [DEBUG] Matched source: '{source_role}' -> '{source_full_id}'")

            # Match destination
            if destination_full_id is None:
                if (destination_role in profile_id or destination_role in profile_id_norm or
                    profile_id == destination_role or profile_id_norm == destination_role):
                    destination_full_id = profile_id
                    print(f"  [DEBUG] Matched destination: '{destination_role}' -> '{destination_full_id}'")

        if not source_full_id:
            print(f"  [!] shift_responsibility: could not find source resource '{source_role}' in profiles")
            return False
        if not destination_full_id:
            print(f"  [!] shift_responsibility: could not find destination resource '{destination_role}' in profiles")
            return False

        print(f"  [OK] shift_responsibility: {source_full_id} -> {destination_full_id}")

        # Determine the format used in task_resource_distribution (with or without _profile)
        # We need to match the existing format to avoid breaking Prosimos
        source_refs = []
        destination_count = 0
        task_resource_format = None  # Will be detected from actual entries
        role_keys = ('resource_id', 'resource', 'resource_name', 'id', 'name')

        def collect_references(entries: list, task_id: str = ""):
            nonlocal destination_count, task_resource_format
            for entry in entries:
                for key in role_keys:
                    if key not in entry:
                        continue
                    role_ref = str(entry.get(key, ''))

                    # Detect the format used in task_resource_distribution
                    if task_resource_format is None and role_ref:
                        task_resource_format = 'with_profile' if '_profile' in role_ref else 'without_profile'

                    # Match against both formats
                    source_match = (role_ref == source_full_id or
                                   role_ref == source_role or
                                   source_role in role_ref)
                    dest_match = (role_ref == destination_full_id or
                                 role_ref == destination_role or
                                 destination_role in role_ref)

                    if source_match:
                        source_refs.append((entry, key, task_id, role_ref))
                        break
                    if dest_match:
                        destination_count += 1
                        break

        task_distributions = params.get('task_resource_distribution', [])
        print(f"  [DEBUG] Found {len(task_distributions)} tasks in task_resource_distribution")

        # Debug: show structure of first task
        if task_distributions:
            first_task = task_distributions[0]
            print(f"  [DEBUG] First task keys: {list(first_task.keys())}")
            resources = first_task.get('resources', [])
            if resources:
                print(f"  [DEBUG] First task has {len(resources)} resources, first resource keys: {list(resources[0].keys())}")
                print(f"  [DEBUG] First resource entry: {resources[0]}")

        for task in task_distributions:
            task_id = task.get('task_id', task.get('activity', 'unknown'))
            resources = task.get('resources', []) or []
            distribution = task.get('distribution', []) or []

            collect_references(resources, task_id)
            collect_references(distribution, task_id)

        source_count = len(source_refs)
        print(f"  [DEBUG] Found {source_count} source task assignments, {destination_count} destination assignments")

        if source_count == 0:
            print(f"  [!] shift_responsibility: no task assignments found for source '{source_full_id}'")
            # Debug: show what resource_ids ARE in the tasks
            all_refs = set()
            for task in task_distributions:
                for res in task.get('resources', []) or []:
                    for key in role_keys:
                        if key in res:
                            all_refs.add(res[key])
                            break
            print(f"  [DEBUG] All resource refs in tasks ({len(all_refs)} total): {list(all_refs)[:10]}{'...' if len(all_refs) > 10 else ''}")
            return False

        # Calculate how many tasks to move from source to destination
        # shift_fraction means "move X% of source's tasks to destination"
        # e.g., shift_fraction=0.3 means move 30% of source's tasks
        move_count = max(1, int(round(source_count * shift_fraction)))

        # Cap at max_movable to ensure source keeps at least 1 assignment
        max_movable = max(0, source_count - 1)
        move_count = min(move_count, max_movable)

        print(f"  [DEBUG] shift_fraction={shift_fraction}, source_count={source_count}, move_count={move_count}, max_movable={max_movable}")
        print(f"  [DEBUG] task_resource_format detected: {task_resource_format}")

        if move_count == 0:
            print(f"  [!] shift_responsibility: nothing to move (source only has {source_count} assignments)")
            return False

        # Determine the correct destination ID format to use
        # Must match the format used in task_resource_distribution
        if task_resource_format == 'without_profile':
            # Tasks use names without _profile suffix
            destination_replacement = destination_role  # e.g., "Tesca Lobes"
        else:
            # Tasks use names with _profile suffix
            destination_replacement = destination_full_id  # e.g., "Tesca Lobes_profile"

        print(f"  [DEBUG] Using destination replacement: '{destination_replacement}'")

        # Replace source references with destination
        for entry, key, task_id, old_value in source_refs[:move_count]:
            entry[key] = destination_replacement
            print(f"    - Shifted task '{task_id}': {old_value} -> {destination_replacement}")

        print(f"  [OK] Shifted {move_count} task assignments: {source_role} -> {destination_replacement}")

        return True

    def _assign_specialist_to_tasks(self, params: Dict, base_role_id: str, specialist_role_id: str, task_hint: str, change_type: str):
        """Attach specialist role to selected tasks and apply slight duration gain for specialization."""
        task_distributions = params.get('task_resource_distribution', [])
        for task in task_distributions:
            task_id = str(task.get('task_id', task.get('activity', '')))
            if task_hint and task_hint not in task_id:
                continue

            distributions = task.get('distribution')
            if not isinstance(distributions, list):
                continue

            candidate_entries = []
            for dist in distributions:
                role_ref = str(
                    dist.get('resource_id', dist.get('resource', dist.get('resource_name', '')))
                )
                if base_role_id in role_ref or role_ref == base_role_id:
                    candidate_entries.append(dist)

            if not candidate_entries:
                continue

            for base_entry in candidate_entries:
                specialist_entry = copy.deepcopy(base_entry)
                if 'resource_id' in specialist_entry:
                    specialist_entry['resource_id'] = specialist_role_id
                elif 'resource' in specialist_entry:
                    specialist_entry['resource'] = specialist_role_id
                elif 'resource_name' in specialist_entry:
                    specialist_entry['resource_name'] = specialist_role_id
                else:
                    specialist_entry['resource_id'] = specialist_role_id

                # Specialization effect: slightly faster execution
                speed_gain = 0.90 if 'create_specialist' in str(change_type) else 0.95
                if 'mean' in specialist_entry and isinstance(specialist_entry['mean'], (int, float)):
                    specialist_entry['mean'] = specialist_entry['mean'] * speed_gain

                distributions.append(specialist_entry)
    
    def _modify_durations(self, params: Dict, target: str, reduction_pct: float):
        """Modify activity durations in Prosimos parameters (duration distributions).
        
        For complex distributions (gamma, lognorm), scale ALL parameters proportionally
        to maintain mathematical validity.
        """
        if 'task_resource_distribution' not in params:
            return
        
        reduction_factor = 1 - (abs(reduction_pct) / 100)
        modified_count = 0
        
        for task in params['task_resource_distribution']:
            task_id = task.get('task_id', task.get('activity', ''))
            if target in task_id or target == task_id:
                # Prosimos uses 'resources' key (not 'distribution')
                # Each resource has duration distribution_params (complex distributions like gamma, expon)
                resources = task.get('resources', task.get('distribution', []))
                
                for resource_dist in resources:
                    dist_name = resource_dist.get('distribution_name', 'unknown').lower()
                    
                    # Try modern SIMOD format (distribution_params)
                    if 'distribution_params' in resource_dist:
                        params_list = resource_dist['distribution_params']
                        
                        # For exponential: only first param is the mean (scale)
                        if 'expon' in dist_name:
                            if params_list and 'value' in params_list[0]:
                                params_list[0]['value'] *= reduction_factor
                                modified_count += 1
                        
                        # For gamma/lognorm: ALL params are interdependent, scale ALL
                        # This maintains the distribution shape while changing the scale
                        elif any(x in dist_name for x in ['gamma', 'lognorm', 'norm', 'weib']):
                            for param in params_list:
                                if 'value' in param:
                                    # Scale all params - preserves distribution shape
                                    param['value'] *= reduction_factor
                                    modified_count += 1
                        
                        # For other distributions, try to scale intelligently
                        else:
                            for param in params_list:
                                if 'value' in param:
                                    param['value'] *= reduction_factor
                                    modified_count += 1
                    
                    # Fallback for legacy format (direct mean)
                    elif 'mean' in resource_dist:
                        resource_dist['mean'] *= reduction_factor
                        modified_count += 1
        
        if modified_count > 0:
            pct = abs(reduction_pct)
            print(f"  [OK] Modified {modified_count} duration parameters ({pct:.0f}% reduction) for {target}")

    def _modify_calendar_hours(self, params: Dict, target: str, change_type: str, value: float):
        """Modify resource calendars to extend working hours.

        For increase_hours: extends each time period by the specified hours,
        effectively simulating longer working days. This increases resource
        availability and can reduce waiting times.

        The extension is applied by:
        1. For targeted calendars (matching target name): extend end times
        2. If target is 'all' or empty: extend all resource calendars

        Args:
            params: Prosimos parameters dictionary
            target: Calendar/resource name to target, or 'all' for all calendars
            change_type: Must contain 'increase_hours'
            value: Number of hours to extend (e.g., 2 means add 2 hours)
        """
        if 'increase_hours' not in str(change_type):
            return

        try:
            extra_hours = float(value or 0)
        except (TypeError, ValueError):
            print(f"  [!] Invalid calendar hours value: {value}")
            return

        if extra_hours <= 0:
            print(f"  [!] Calendar hours must be positive, got: {extra_hours}")
            return

        resource_calendars = params.get('resource_calendars', [])
        if not resource_calendars:
            print("  [!] No resource_calendars found in Prosimos parameters")
            return

        # Determine which calendars to modify
        target_all = not target or target.lower() in ('all', '*', 'global')
        target_norm = target.lower().replace('_profile', '').replace('_calendar', '') if target else ''

        modified_calendars = 0
        modified_periods = 0

        for calendar in resource_calendars:
            cal_id = str(calendar.get('id', calendar.get('name', '')))
            cal_norm = cal_id.lower().replace('_profile', '').replace('_calendar', '')

            # Check if this calendar should be modified
            if not target_all and target_norm not in cal_norm and cal_norm not in target_norm:
                continue

            time_periods = calendar.get('time_periods', [])
            if not time_periods:
                continue

            for period in time_periods:
                end_time_str = period.get('endTime', '')
                if not end_time_str:
                    continue

                try:
                    # Parse end time (format: HH:MM:SS or HH:MM:SS.ffffff)
                    end_parts = end_time_str.split(':')
                    end_hour = int(end_parts[0])
                    end_min = int(end_parts[1])
                    end_sec_parts = end_parts[2].split('.') if len(end_parts) > 2 else ['0']
                    end_sec = int(end_sec_parts[0])

                    # Add extra hours
                    new_end_hour = end_hour + int(extra_hours)
                    new_end_min = end_min + int((extra_hours % 1) * 60)

                    # Handle minute overflow
                    if new_end_min >= 60:
                        new_end_hour += 1
                        new_end_min -= 60

                    # Cap at 23:59:59 (don't overflow to next day)
                    if new_end_hour >= 24:
                        new_end_hour = 23
                        new_end_min = 59
                        end_sec = 59

                    # Format new end time
                    new_end_time = f"{new_end_hour:02d}:{new_end_min:02d}:{end_sec:02d}"
                    if '.' in end_time_str:
                        # Preserve microseconds if present
                        new_end_time += '.' + end_time_str.split('.')[-1]

                    if new_end_time != end_time_str:
                        period['endTime'] = new_end_time
                        modified_periods += 1

                except (ValueError, IndexError) as e:
                    print(f"  [!] Could not parse time '{end_time_str}': {e}")
                    continue

            if modified_periods > 0:
                modified_calendars += 1

        if modified_calendars > 0:
            target_desc = 'all calendars' if target_all else f"'{target}'"
            print(f"  [OK] Extended {modified_periods} time periods across {modified_calendars} calendars "
                  f"(+{extra_hours:.1f}h) for {target_desc}")
        else:
            print(f"  [!] No calendars matched target '{target}'")

    def _run_simulation(
        self,
        model: Dict,
        params_file: str,
        work_dir: str,
        total_cases: Optional[int] = None,
        time_horizon_seconds: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict:
        """Run Prosimos simulation via Docker.

        Args:
            model: Baseline simulation model
            params_file: Path to modified parameters JSON
            work_dir: Working directory
            total_cases: Number of cases to simulate
            time_horizon_seconds: Optional time horizon filter
            seed: Random seed for reproducibility

        Returns:
            Dictionary of simulated metrics

        Raises:
            RuntimeError: If BPMN file not found, Docker fails, or metrics extraction fails
        """

        # Get BPMN file path - NO FALLBACK
        bpmn_work_file = os.path.join(work_dir, "model.bpmn")

        if 'bpmn_path' in model:
            bpmn_path = model['bpmn_path']
            if os.path.exists(bpmn_path):
                print(f"[OK] Found BPMN file: {bpmn_path}")
                shutil.copy2(bpmn_path, bpmn_work_file)
            else:
                raise RuntimeError(f"BPMN file not found: {bpmn_path}")
        elif 'bpmn_model' in model:
            with open(bpmn_work_file, 'w') as f:
                f.write(model['bpmn_model'])
        else:
            raise RuntimeError("No BPMN model available for simulation (need 'bpmn_path' or 'bpmn_model')")

        # Prepare simulation command
        output_dir = os.path.join(work_dir, "simulated_output")
        os.makedirs(output_dir, exist_ok=True)

        # Convert to absolute path for Docker volume mount (required on Windows)
        abs_work_dir = os.path.abspath(work_dir)

        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        sim_cases = int(total_cases) if total_cases is not None else 1000
        cmd.extend([
            "docker", "run", "--rm",
            "-v", f"{abs_work_dir}:/data",
            "-w", "/usr/src/Simod",
            self.docker_image,
            "/usr/src/Simod/.venv/bin/prosimos",
            "start-simulation",
            "--bpmn_path", "/data/model.bpmn",
            "--json_path", "/data/modified_params.json",
            "--total_cases", str(sim_cases),
            "--stat_out_path", "/data/simulated_output/statistics.csv",
            "--log_out_path", "/data/simulated_output/simulation_log.csv"
        ])

        if time_horizon_seconds:
            print(f"Running Prosimos simulation (max {sim_cases} cases, will filter to {time_horizon_seconds/86400:.1f} days)...")
        else:
            print(f"Running Prosimos simulation ({sim_cases} cases)...")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Parameters file: {params_file}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = f"Prosimos simulation failed (exit code {result.returncode})"
            if result.stderr:
                error_msg += f"\n  stderr: {result.stderr[:500]}"
            if result.stdout:
                error_msg += f"\n  stdout: {result.stdout[:500]}"
            raise RuntimeError(error_msg)

        print(f"[OK] Prosimos simulation completed")

        # Extract metrics from output - NO FALLBACK
        simulated_log = os.path.join(output_dir, "simulation_log.csv")
        if not os.path.exists(simulated_log):
            available = os.listdir(output_dir) if os.path.exists(output_dir) else []
            raise RuntimeError(f"Simulation log not found at {simulated_log}. Available files: {available}")

        extractor = MetricsExtractor()
        metrics = extractor.extract_from_csv(
            simulated_log,
            time_horizon_seconds=time_horizon_seconds
        )

        if not metrics:
            raise RuntimeError(f"Failed to extract metrics from {simulated_log}")

        print(f"  [OK] Extracted metrics: {list(metrics.keys())}")
        return metrics
