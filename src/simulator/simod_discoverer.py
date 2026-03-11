import os
import json
import subprocess
import shutil
from typing import Tuple, Dict
import yaml
import pm4py
import pandas as pd
from collections import Counter

from utils.column_utils import find_column


class SIMODDiscoverer:
    def __init__(self, use_sudo: bool = True):
        """
        Initialize SIMOD discoverer.
        
        Args:
            use_sudo: Whether to use sudo for Docker commands
        """
        self.docker_image = "nokal/simod:latest"
        self.use_sudo = use_sudo
    
    def discover(self, log_path: str, work_dir: str) -> Tuple[Dict, Dict]:
        """
        Discover baseline simulation model.
        Args:
            log_path: Path to XES or CSV event log
            work_dir: Working directory for SIMOD outputs
        Returns:
            Tuple of (model_dict, metrics_dict)
        Raises:
            RuntimeError if any step fails
        """
        # Extract real baseline metrics from the log FIRST (regardless of SIMOD)
        baseline_metrics = self._extract_baseline_from_log(log_path)
        # Convert XES to CSV if needed
        csv_path = self._convert_xes_to_csv(log_path)
        # Ensure working directory exists
        os.makedirs(work_dir, exist_ok=True)
        # Copy CSV to working directory
        csv_name = os.path.basename(csv_path)
        work_csv_path = os.path.join(work_dir, csv_name)
        shutil.copy2(csv_path, work_csv_path)
        # Create SIMOD output directory
        output_dir = os.path.join(work_dir, "baseline")
        os.makedirs(output_dir, exist_ok=True)
        # Create and run SIMOD configuration (pass actual csv_path for column detection)
        config_path = self._create_simod_config(
            actual_csv_path=csv_path,
            docker_csv_path=f"/data/{csv_name}",
            output_dir="/data/baseline",
            work_dir=work_dir
        )
        # Run SIMOD discovery via Docker
        self._run_simod_discovery(work_dir, output_dir)
        # Parse outputs
        model_data = self._parse_simod_output(output_dir)
        if model_data:
            # Use SIMOD model with real baseline metrics from log
            return model_data, baseline_metrics
        else:
            raise RuntimeError("Could not parse SIMOD output.")

    def _convert_xes_to_csv(self, log_path: str) -> str:
        """Convert XES event log to CSV if needed. Returns existing CSV or converted path."""
        try:
            if log_path.endswith('.csv'):
                print(f"[OK] Log is already in CSV format: {log_path}")
                return log_path
            
            print(f"Converting XES to CSV for SIMOD compatibility...")
            log = pm4py.read_xes(log_path)
            df = pm4py.convert_to_dataframe(log)
            column_mapping = {
                'case:concept:name': 'case_id',
                'concept:name': 'activity',
                'time:timestamp': 'end_time',
                'org:resource': 'resource'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            if 'start_time' not in df.columns and 'end_time' in df.columns:
                df['start_time'] = df['end_time']
            
            csv_path = log_path.replace('.xes', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"[OK] Converted to CSV: {csv_path}")
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)[:5]}...")
            
            return csv_path
            
        except Exception as e:
            print(f"Error converting XES to CSV: {e}")
            return log_path
    
    def _create_simod_config(self, actual_csv_path: str, docker_csv_path: str, output_dir: str, work_dir: str) -> str:
        """Create SIMOD configuration YAML file with auto-detected column names.
        
        Args:
            actual_csv_path: Actual path to CSV on disk (for reading columns)
            docker_csv_path: Path as it appears in Docker container (for SIMOD config)
            output_dir: Output directory path in Docker
            work_dir: Working directory on host
        """

        # Read columns from the actual CSV file on disk
        try:
            df = pd.read_csv(actual_csv_path, nrows=1)
            columns = set(df.columns.tolist())
        except Exception as e:
            print(f"Warning: Could not read CSV {actual_csv_path}, using defaults: {e}")
            columns = set()

        # SIMOD expects specific column names in log_ids
        # We detect actual columns and map them
        case_col = find_column(columns, ['case_id', 'case'], 'case_id')

        # Activity: try concept:name first (XES standard), then activity_name, then Activity
        activity_col = find_column(columns, ['concept:name', 'activity_name', 'activity'], 'activity_name')

        # Resource: try org:resource first, then resource
        resource_col = find_column(columns, ['org:resource', 'resourceid', 'resource'], 'resource')

        # Start time - optional, many logs don't have it
        start_col = find_column(columns, ['start_timestamp', 'start_time', 'start'], None)

        # End time - critical, always needed
        end_col = find_column(columns, ['end_timestamp', 'end_time', 'end'], 'end_timestamp')

        print(f"[OK] Detected SIMOD columns from {os.path.basename(actual_csv_path)}:")
        print(f"  case_id={case_col}, activity={activity_col}, resource={resource_col}")
        if start_col:
            print(f"  start_time={start_col}")
        print(f"  end_time={end_col}")
        
        # Build log_ids mapping for SIMOD
        log_ids_mapping = {
            'case': case_col,
            'activity': activity_col,
            'resource': resource_col,
            'end_time': end_col
        }
        if start_col:
            log_ids_mapping['start_time'] = start_col
        
        config = {
            'version': 5,
            'common': {
                'train_log_path': docker_csv_path,
                'log_ids': log_ids_mapping,
                'discover_data_attributes': False
            },
            'preprocessing': {
                'enable_time_concurrency_threshold': 0.75
            },
            'control_flow': {
                'optimization_metric': 'two_gram_distance',
                'num_iterations': 5,
                'num_evaluations_per_iteration': 1,
                'gateway_probabilities': 'discovery',
                'mining_algorithm': 'sm1',
                'epsilon': [0.05, 0.4],
                'eta': [0.2, 0.7],
            },
            'resource_model': {
                'optimization_metric': 'circadian_emd',
                'num_iterations': 5,
                'num_evaluations_per_iteration': 1,
                'resource_profiles': {
                    'discovery_type': 'differentiated',
                    'granularity': 60,
                    'confidence': [0.5, 0.85],
                    'support': [0.05, 0.5],
                    'participation': 0.4
                }
            }
        }
        
        config_path = os.path.join(work_dir, 'simod_config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[OK] Created SIMOD config: {config_path}")
        return config_path
    def _run_simod_discovery(self, work_dir: str, output_dir: str):
        """Run SIMOD discovery via Docker."""
        cmd = []
        if self.use_sudo:
            cmd.append("sudo")
        cmd.extend([
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/data",
            "-w", "/usr/src/Simod",
            self.docker_image,
            "bash", "/usr/src/Simod/run.sh",
            "/data/simod_config.yml",
            "/data/baseline"
        ])
        
        print(f"Running SIMOD: {' '.join(cmd)}")
        print("Note: SIMOD discovery may take 10-30 minutes...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        # Always print SIMOD output for debugging
        if result.stdout:
            print(f"SIMOD stdout:\n{result.stdout[:1000]}")  # First 1000 chars
        if result.stderr:
            print(f"SIMOD stderr:\n{result.stderr[:1000]}")  # First 1000 chars
        
        if result.returncode != 0:
            print(f"SIMOD error (exit code {result.returncode}): {result.stderr}")
            raise RuntimeError(f"SIMOD discovery failed: {result.stderr}")
        
        print("[OK] SIMOD discovery completed")
    
    def _parse_simod_output(self, output_dir: str) -> Dict:
        """Parse SIMOD output files."""
        try:
            # Debug: List all files in output directory
            print(f"\nSearching for SIMOD output in: {output_dir}")
            if os.path.exists(output_dir):
                all_files = []
                for root, dirs, files in os.walk(output_dir):
                    for f in files:
                        rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                        size = os.path.getsize(os.path.join(root, f))
                        all_files.append((rel_path, size))
                
                print(f"Files found in output directory ({len(all_files)} total):")
                for fname, size in sorted(all_files):
                    print(f"  - {fname} ({size} bytes)")
            else:
                print(f"ERROR: Output directory does not exist!")
            
            # SIMOD outputs files in a best_result/ subdirectory
            best_result_dir = os.path.join(output_dir, "best_result")
            if not os.path.exists(best_result_dir):
                print(f"[X] ERROR: No best_result/ subdirectory found!")
                return None
            
            print(f"[OK] Found best_result/ subdirectory")
            
            # Look for BPMN file
            bpmn_files = [f for f in os.listdir(best_result_dir) if f.endswith('.bpmn')]
            if not bpmn_files:
                print(f"[X] ERROR: No BPMN file found in best_result/")
                return None
            
            bpmn_path = os.path.join(best_result_dir, bpmn_files[0])
            bpmn_basename = os.path.splitext(bpmn_files[0])[0]  # e.g., "sepsis"
            print(f"[OK] Found BPMN model: {bpmn_files[0]}")
            
            # Look for Prosimos JSON parameters - prefer one with same base name as BPMN
            json_files = [f for f in os.listdir(best_result_dir) if f.endswith('.json')]
            if not json_files:
                print(f"[X] ERROR: No JSON parameter file found in best_result/")
                return None
            
            # Prefer JSON with same base name as BPMN (e.g., sepsis.json for sepsis.bpmn)
            prosimos_file = None
            for jf in json_files:
                if jf.startswith(bpmn_basename):
                    prosimos_file = jf
                    break
            
            # Fall back to first JSON file if no matching name found
            if not prosimos_file:
                # Filter out common metadata files
                metadata_files = {'canonical_model.json', 'configuration.yaml', 'runtimes.json'}
                prosimos_file = next((f for f in json_files if f not in metadata_files), json_files[0])
            
            prosimos_path = os.path.join(best_result_dir, prosimos_file)
            print(f"[OK] Found Prosimos parameters: {prosimos_file}")
            
            with open(prosimos_path, 'r') as f:
                prosimos_data = json.load(f)
            
            # Validate that prosimos_data has expected structure
            if not isinstance(prosimos_data, dict):
                print(f"[X] WARNING: Prosimos parameters not a dict, got {type(prosimos_data)}")
                return None
            
            return {
                'bpmn_path': bpmn_path,
                'prosimos_path': prosimos_path,
                'prosimos_params': prosimos_data,
                'output_dir': output_dir
            }
            
        except Exception as e:
            print(f"Error parsing SIMOD output: {e}")
            return None

    def _extract_baseline_from_log(self, log_path: str) -> Dict:
        """Extract real baseline metrics from the event log using pm4py with smart column detection."""
        try:
            print("Extracting baseline metrics from event log...")

            # Read log (handle both XES and CSV)
            if log_path.endswith('.csv'):
                df = pd.read_csv(log_path)

                # Smart column detection (same as SIMOD config)
                columns = set(df.columns.tolist())

                case_col = find_column(columns, ['case_id', 'case'], None)
                activity_col = find_column(columns, ['concept:name', 'activity_name', 'activity'], None)
                start_col = find_column(columns, ['start_timestamp', 'start_time', 'start'], None)
                end_col = find_column(columns, ['end_timestamp', 'end_time', 'end', 'time:timestamp'], None)

                # If we found these columns, extract metrics from CSV
                if case_col and activity_col and end_col:
                    print(f"  Found columns: case={case_col}, activity={activity_col}, start={start_col}, end={end_col}")
                    metrics = self._extract_metrics_from_csv(df, case_col, start_col, end_col)
                else:
                    print(f"  Warning: Could not find required columns. Found: case={case_col}, activity={activity_col}, end={end_col}")
            else:
                log = pm4py.read_xes(log_path)
                metrics = self._extract_metrics_from_xes_log(log)

            print(f"[OK] Extracted baseline metrics from log:")
            avg_cycle = metrics.get('avg_cycle_time', 0)
            avg_through = metrics.get('avg_throughput', 0)
            avg_wait = metrics.get('avg_waiting_time', 0)
            print(f"  - Cycle time: {avg_cycle:.0f} seconds ({avg_cycle/3600:.1f} hours)")
            print(f"  - Throughput: {avg_through:.1f} cases/day")
            print(f"  - Waiting time: {avg_wait:.0f} seconds ({avg_wait/3600:.1f} hours)")
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not extract metrics from log: {e}")
            return
    
    def _extract_metrics_from_xes_log(self, log) -> Dict:
        try:
            metrics = {}
            if isinstance(log, pd.DataFrame):
                # Cycle time: time difference per case
                if 'case:concept:name' in log.columns and 'time:timestamp' in log.columns:
                    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], errors='coerce')
                    try:
                        cycle_times = log.groupby('case:concept:name')['time:timestamp'].apply(
                            lambda x: (x.max() - x.min()).total_seconds()
                        )
                        # Filter out negative cycle times
                        valid_cycle_times = cycle_times[cycle_times >= 0]
                        metrics['avg_cycle_time'] = float(valid_cycle_times.mean()) if len(valid_cycle_times) > 0 else 14400
                    except:
                        metrics['avg_cycle_time'] = 14400

                # Throughput: cases per day (consistent with CSV extraction)
                if 'case:concept:name' in log.columns and 'time:timestamp' in log.columns:
                    num_cases = log['case:concept:name'].nunique()
                    time_span_seconds = (log['time:timestamp'].max() - log['time:timestamp'].min()).total_seconds()
                    if time_span_seconds > 0:
                        metrics['avg_throughput'] = float(num_cases / (time_span_seconds / 86400.0))
                    else:
                        metrics['avg_throughput'] = float(num_cases)
                elif 'case:concept:name' in log.columns:
                    metrics['avg_throughput'] = float(log['case:concept:name'].nunique())

                # Waiting time: estimate as 30% of cycle time
                if 'avg_cycle_time' in metrics:
                    metrics['avg_waiting_time'] = metrics['avg_cycle_time'] * 0.3

                # Resource utilization: from org:resource or Resource column
                resource_col = next((c for c in ('org:resource', 'Resource') if c in log.columns), None)
                if resource_col:
                    resource_counts = log[resource_col].value_counts()
                    total = len(log)
                    metrics['resource_utilization'] = {
                        str(res): round(min(count / total * 10, 0.95), 2)
                        for res, count in resource_counts.items() if pd.notna(res)
                    }
                else:
                    metrics['resource_utilization'] = {}

            else:
                # Handle event log object format (fallback)
                try:
                    cycle_times = pm4py.statistics.traces.generic.log.case_statistics.get_all_case_durations(log)
                    if cycle_times:
                        metrics['avg_cycle_time'] = float(sum(cycle_times) / len(cycle_times))
                except:
                    metrics['avg_cycle_time'] = 14400

                # For event log objects, just use total cases (can't easily get time span)
                metrics['avg_throughput'] = float(len(log))

                if 'avg_cycle_time' in metrics:
                    metrics['avg_waiting_time'] = metrics['avg_cycle_time'] * 0.3

                # Try to extract resources from event objects
                resources_list = []
                try:
                    for trace in log:
                        for event in trace:
                            if isinstance(event, dict):
                                if 'org:resource' in event:
                                    resources_list.append(event['org:resource'])
                except:
                    pass
                
                if resources_list:
                    resource_counts = Counter(resources_list)
                    total_events = len(resources_list)
                    resources = {}
                    for res, count in resource_counts.items():
                        utilization = min(count / total_events * 10, 0.95)
                        resources[res] = round(utilization, 2)
                    metrics['resource_utilization'] = resources
                else:
                    metrics['resource_utilization'] = {}

            return metrics
            
        except Exception as e:
            print(f"Error extracting from XES: {e}")
            return {}
    
    def _extract_metrics_from_csv(self, df, case_col: str, start_col: str, end_col: str) -> Dict:
        """Extract metrics from CSV using same methodology as simulation."""
        try:
            metrics = {}

            # Parse timestamps
            if end_col and end_col in df.columns:
                df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
            if start_col and start_col in df.columns:
                df[start_col] = pd.to_datetime(df[start_col], errors='coerce')

            # Calculate cycle time: max(end) - min(start) per case
            if start_col and start_col in df.columns and end_col and end_col in df.columns:
                case_start = df.groupby(case_col)[start_col].min()
                case_end = df.groupby(case_col)[end_col].max()
                cycle_times = (case_end - case_start).dt.total_seconds()
                # Filter out invalid (negative) cycle times caused by data errors
                valid_cycle_times = cycle_times[cycle_times >= 0]
                if len(valid_cycle_times) < len(cycle_times):
                    n_invalid = len(cycle_times) - len(valid_cycle_times)
                    print(f"  [!] Filtered {n_invalid} cases with invalid timestamps (end < start)")
                if len(valid_cycle_times) > 0:
                    metrics['avg_cycle_time'] = float(valid_cycle_times.mean())
                else:
                    metrics['avg_cycle_time'] = 0.0
            elif end_col and end_col in df.columns:
                # Fallback: use end_timestamp only
                cycle_times = df.groupby(case_col)[end_col].apply(
                    lambda x: (x.max() - x.min()).total_seconds())
                valid_cycle_times = cycle_times[cycle_times >= 0]
                metrics['avg_cycle_time'] = float(valid_cycle_times.mean()) if len(valid_cycle_times) > 0 else 0.0

            # Calculate throughput as cases/day (matching simulation)
            num_cases = df[case_col].nunique()
            time_anchor = end_col if end_col and end_col in df.columns else start_col
            if time_anchor and time_anchor in df.columns and df[time_anchor].notna().any():
                time_span_seconds = (df[time_anchor].max() - df[time_anchor].min()).total_seconds()
                if time_span_seconds > 0:
                    metrics['avg_throughput'] = float(num_cases / (time_span_seconds / 86400.0))
                else:
                    metrics['avg_throughput'] = float(num_cases)
            else:
                metrics['avg_throughput'] = float(num_cases)

            # Waiting time estimate
            if 'avg_cycle_time' in metrics:
                metrics['avg_waiting_time'] = metrics['avg_cycle_time'] * 0.3

            # Resource utilization (unchanged)
            resource_cols = [col for col in df.columns if 'resource' in col.lower()]
            if resource_cols:
                resource_col = resource_cols[0]
                resources_list = df[resource_col].dropna().tolist()
                if resources_list:
                    resource_counts = Counter(resources_list)
                    total_events = len(resources_list)
                    resources = {}
                    for res, count in resource_counts.items():
                        utilization = min(count / total_events * 10, 0.95)
                        resources[str(res)] = round(utilization, 2)
                    metrics['resource_utilization'] = resources
                    print(f"    [OK] Found {len(resources)} real resources from '{resource_col}' column")
                else:
                    metrics['resource_utilization'] = {}
            else:
                print(f"    Warning: No resource column found. Available columns: {list(df.columns)[:10]}")
                metrics['resource_utilization'] = {}

            return metrics
        except Exception as e:
            print(f"    Error extracting from CSV: {e}")
            return {}