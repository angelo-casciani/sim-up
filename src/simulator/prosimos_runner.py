import os
import json
import subprocess
import copy
from typing import Dict, Optional
from simulator.metrics_extractor import MetricsExtractor
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
        baseline_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Run Prosimos simulation with modifications.
        
        Args:
            model: Baseline simulation model
            modifications: Parameter modifications from hypothesis
            work_dir: Working directory
            baseline_metrics: Optional baseline metrics with real resources for enrichment
            
        Returns:
            Dictionary of simulated metrics
        """
        self.work_dir = work_dir
        
        try:
            # Apply modifications to Prosimos parameters
            modified_params = self._apply_modifications(model, modifications)

            # Save modified parameters
            params_file = os.path.join(work_dir, "modified_params.json")
            with open(params_file, 'w') as f:
                json.dump(modified_params, f, indent=2)
            
            # Run simulation - Try Docker first, fallback to estimation preserving real resources
            result = self._run_simulation(
                model,
                params_file,
                work_dir,
                total_cases=total_cases,
                time_horizon_seconds=time_horizon_seconds
            )
            
            # If docker failed, ensure we're using real resources in fallback
            if not result or len(result.get('resource_utilization', {})) == 0:
                print("Simulation unavailable, using estimation mode with baseline resources")
                # Fall back to estimation based on real baseline resources
                if baseline_metrics and 'resource_utilization' in baseline_metrics:
                    real_resources = {
                        str(k): v for k, v in baseline_metrics['resource_utilization'].items()
                    }
                    result['resource_utilization'] = real_resources
                    print(f"  Using {len(real_resources)} real baseline resources in estimation")
                elif 'prosimos_params' in model and 'resource_profiles' in model['prosimos_params']:
                    real_profiles = model['prosimos_params']['resource_profiles']
                    real_resources = {
                        str(p.get('id', p.get('name', f'Resource_{i}'))): 0.85 
                        for i, p in enumerate(real_profiles)
                    }
                    result['resource_utilization'] = real_resources
                    print(f"  Using {len(real_resources)} SIMOD-discovered resources in estimation")
                else:
                    # Minimal fallback with default values
                    result['resource_utilization'] = {'resource_pool': 0.5}
            
            return result
            
        except subprocess.TimeoutExpired:
            print("Simulation timed out")
            return {}
        except Exception as e:
            print(f"Error during simulation: {e}")
            return {}
    
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
        """Modify resource profiles in Prosimos parameters."""
        if 'resource_profiles' not in params:
            return
        
        for resource in params['resource_profiles']:
            res_id = resource.get('id', resource.get('name'))
            if target in res_id or res_id == target:
                current_amount = resource.get('amount', 1)
                if 'add' in change_type or 'increase' in change_type:
                    resource['amount'] = current_amount + int(value)
                elif 'remove' in change_type or 'decrease' in change_type:
                    resource['amount'] = max(1, current_amount - int(value))

    def _modify_role_redesign(self, params: Dict, target: str, change_type: str, value: int, change_payload: Dict):
        """Create/reshape roles to model specialization without only adding generic capacity."""
        if 'resource_profiles' not in params or not target:
            return

        source_role, destination_role = self._parse_role_redesign_target(target)
        resource_profiles = params.get('resource_profiles', [])
        matched_profile = None
        for resource in resource_profiles:
            res_id = str(resource.get('id', resource.get('name', '')))
            if source_role and (source_role in res_id or res_id == source_role):
                matched_profile = resource
                break

        if not matched_profile:
            return

        if 'shift_responsibility' in str(change_type) and destination_role:
            shift_fraction = self._resolve_shift_fraction(
                change_payload.get('value')
            )
            shifted = self._shift_role_responsibilities(
                params,
                source_role=source_role,
                destination_role=destination_role,
                shift_fraction=shift_fraction
            )
            if shifted:
                return

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

    @staticmethod
    def _normalize_role_label(label: str) -> str:
        role = str(label or '').strip()
        if role.lower().endswith('_profile'):
            role = role[:-len('_profile')].strip()
        return role

    def _parse_role_redesign_target(self, target: str):
        raw = str(target or '')
        for sep in ['→', '->']:
            if sep in raw:
                left, right = raw.split(sep, 1)
                return self._normalize_role_label(left), self._normalize_role_label(right)
        normalized = self._normalize_role_label(raw)
        return normalized, normalized

    @staticmethod
    def _resolve_shift_fraction(value) -> float:
        """Resolve shift fraction from payload; handles fractions and integer severity values."""
        try:
            numeric = float(value)
        except Exception:
            return 0.5
        if 0 < numeric < 1:
            return max(0.1, min(0.9, numeric))
        if numeric >= 1:
            n = int(numeric)
            return min(0.9, 0.1 + n * 0.2)  # 1→0.3, 2→0.5, 3→0.7, >=4→0.9
        return 0.5

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
            return False

        source_refs = []
        destination_count = 0
        role_keys = ('resource_id', 'resource', 'resource_name', 'id', 'name')

        def collect_references(entries: list):
            nonlocal destination_count
            for entry in entries:
                for key in role_keys:
                    if key not in entry:
                        continue
                    role_ref = str(entry.get(key, ''))
                    if source_role in role_ref or role_ref == source_role:
                        source_refs.append((entry, key))
                        break
                    if destination_role in role_ref or role_ref == destination_role:
                        destination_count += 1
                        break

        task_distributions = params.get('task_resource_distribution', [])
        for task in task_distributions:
            collect_references(task.get('resources', []) or [])
            collect_references(task.get('distribution', []) or [])

        source_count = len(source_refs)
        if source_count == 0:
            return False

        total_related = source_count + destination_count
        if total_related <= 1:
            return False

        target_destination = int(round(total_related * shift_fraction))
        target_destination = max(1, min(total_related - 1, target_destination))

        max_movable = max(0, source_count - 1)
        move_count = max(0, min(max_movable, target_destination - destination_count))
        if move_count == 0:
            return False

        for entry, key in source_refs[:move_count]:
            entry[key] = destination_role

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
            print(f"  ✓ Modified {modified_count} duration parameters ({pct:.0f}% reduction) for {target}")
    
    def _run_simulation(
        self,
        model: Dict,
        params_file: str,
        work_dir: str,
        total_cases: Optional[int] = None,
        time_horizon_seconds: Optional[float] = None
    ) -> Dict:
        """Run Prosimos simulation via Docker."""
        
        # Get BPMN file path
        bpmn_path = None
        bpmn_work_file = os.path.join(work_dir, "model.bpmn")
        
        if 'bpmn_path' in model:
            # Use path from SIMOD output
            bpmn_path = model['bpmn_path']
            if os.path.exists(bpmn_path):
                print(f"✓ Found BPMN file: {bpmn_path}")
                shutil.copy2(bpmn_path, bpmn_work_file)
            else:
                print(f"✗ BPMN file not found: {bpmn_path}")
                print(f"  WARNING: Using estimation fallback with real resources from params")
                return self._estimate_metrics_from_model(model, params_file)
        elif 'bpmn_model' in model:
            # Legacy: use model content
            with open(bpmn_work_file, 'w') as f:
                f.write(model['bpmn_model'])
        else:
            print("Warning: No BPMN model for simulation")
            print(f"  Using estimation fallback with real resources from params")
            return self._estimate_metrics_from_model(model, params_file)
        
        # Prepare simulation command
        output_dir = os.path.join(work_dir, "simulated_output")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        sim_cases = int(total_cases) if total_cases is not None else 1000
        cmd.extend([
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/data",
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
        print(f"  Parameters file: {params_file}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"⚠️  Prosimos simulation failed (exit code {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr[:300]}")
            print(f"  Using estimation fallback with real resources from params")
            return self._estimate_metrics_from_model(model, params_file)
        
        print(f"✓ Prosimos simulation completed")
        
        # Extract metrics from output
        simulated_log = os.path.join(output_dir, "simulation_log.csv")
        if os.path.exists(simulated_log):
            # Use simulator's metrics extractor for detailed analysis
            try:
                extractor = MetricsExtractor()
                metrics = extractor.extract_from_csv(
                    simulated_log,
                    time_horizon_seconds=time_horizon_seconds
                )
                print(f"  ✓ Extracted metrics: {list(metrics.keys())}")
                return metrics
            except Exception as e:
                print(f"Warning: Failed to extract metrics: {e}")
                print(f"  Using estimation fallback with real resources from params")
                return self._estimate_metrics_from_model(model, params_file)
        else:
            print(f"Warning: No simulation log found at {simulated_log}")
            print(f"  Available files in output_dir: {os.listdir(output_dir) if os.path.exists(output_dir) else 'none'}")
            print(f"  Using estimation fallback with real resources from params")
            return self._estimate_metrics_from_model(model, params_file)
    
    def _estimate_metrics_from_model(self, model: Dict, params_file: str) -> Dict:
        """Estimate metrics from modified parameters using REAL resource profiles from Prosimos params."""
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Extract REAL resource profiles that SIMOD discovered
            real_resources = {}
            if 'resource_profiles' in params and isinstance(params['resource_profiles'], list):
                print(f"Using {len(params['resource_profiles'])} SIMOD-discovered resource profiles:")
                for i, profile in enumerate(params['resource_profiles']):
                    res_id = profile.get('id') or profile.get('name') or f'Resource_{i}'
                    amount = profile.get('amount', 1)
                    # Use realistic utilization based on profile amount
                    util = min(0.95, max(0.2, 0.85 - (amount - 1) * 0.15))
                    real_resources[str(res_id)] = util
                    if i < 3:  # Print first 3
                        print(f"  - {res_id}: amount={amount}, utilization={util:.2f}")
                if len(params['resource_profiles']) > 3:
                    print(f"  ... and {len(params['resource_profiles']) - 3} more")
            else:
                print("Warning: No resource_profiles found in Prosimos parameters")
            
            # If we couldn't extract resources, return empty dict to indicate failure
            if not real_resources:
                print("Error: No resources extracted from parameters")
                return {}
            
            # Estimate metrics based on resource capacity
            total_capacity = sum(
                p.get('amount', 1) for p in params.get('resource_profiles', [])
            ) if 'resource_profiles' in params else 1
            
            # Baseline assumption: more resources = better throughput/lower cycle time
            baseline_resources = 2
            if total_capacity > baseline_resources:
                improvement_factor = baseline_resources / total_capacity
            else:
                improvement_factor = 1.0
            
            baseline_cycle = 36432.72  
            baseline_throughput = 110.0
            baseline_waiting = 10929.82
            
            est_metrics = {
                'avg_cycle_time': baseline_cycle * improvement_factor,
                'avg_throughput': baseline_throughput,
                'avg_waiting_time': baseline_waiting * improvement_factor,
                'resource_utilization': real_resources,
                'estimation_mode': True,  # Mark as estimation not actual simulation
                'num_resources': len(real_resources),
                'total_resource_capacity': total_capacity
            }
            
            print(f"✓ Estimated metrics for {len(real_resources)} resources (capacity={total_capacity})")
            return est_metrics
            
        except Exception as e:
            print(f"Error estimating metrics: {e}")
            return {}
