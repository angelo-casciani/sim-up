"""AgentSimulator (ABS) wrapper for parallel execution with Prosimos (DES)."""

import os
import json
import sys
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime
import shutil
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSimulatorRunner:
    """Runs AgentSimulator for agent-based simulation perspective (Organizational/Social View)."""
    
    def __init__(self, agent_simulator_path: str = None):
        """
        Initialize AgentSimulator runner.
        
        Args:
            agent_simulator_path: Path to AgentSimulator repo (if None, tries to discover)
        """
        self.agent_simulator_path = agent_simulator_path or self._discover_agent_simulator()
        self.available = False
        self.agent_simulator_module = None
        
        if self.agent_simulator_path and os.path.exists(self.agent_simulator_path):
            # Add to sys.path to import source module
            if self.agent_simulator_path not in sys.path:
                sys.path.insert(0, self.agent_simulator_path)
            
            try:
                from source.agent_simulator import AgentSimulator
                self.agent_simulator_module = AgentSimulator
                self.available = True
                logger.info(f"AgentSimulator initialized from {self.agent_simulator_path}")
            except ImportError as e:
                logger.warning(f"Could not import AgentSimulator: {e}. ABS simulations will be skipped.")
                self.available = False
        else:
            logger.warning(f"AgentSimulator not found at {self.agent_simulator_path}. ABS simulations will be skipped.")
    
    def _discover_agent_simulator(self) -> str:
        """Try to find AgentSimulator installation."""
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))
        repo_root = os.path.dirname(src_dir)
        env_path = os.getenv("AGENT_SIMULATOR_PATH")

        # Check preferred local and common locations
        candidates = [
            env_path,
            os.path.join(src_dir, "AgentSimulator"),
            os.path.join(repo_root, "AgentSimulator"),
            "/home/angelo/projects/AgentSimulator",
            "/opt/AgentSimulator",
            os.path.expanduser("~/AgentSimulator"),
            "../AgentSimulator",
            "../../AgentSimulator"
        ]
        
        for path in candidates:
            if not path:
                continue
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                source_path = os.path.join(full_path, "source")
                if os.path.exists(source_path):
                    return full_path
        
        return None
    def simulate(
        self,
        log_path: str,
        modifications: Dict = None,
        output_dir: str = None,
        orchestration_type: str = "auto",
        delay_discovery: bool = False,
        num_simulations: int = 10,
        enabled: bool = True,
        time_horizon_seconds: float = None
    ) -> tuple:
        """
        Run AgentSimulator with given log and modifications.
        
        Args:
            log_path: Path to input event log (XES/CSV)
            modifications: Dictionary of hypothesis modifications (not directly used by AgentSimulator)
            output_dir: Directory to save simulated logs
            orchestration_type: "centralized" or "autonomous" (default: auto)
            delay_discovery: Whether to enable delay discovery
            num_simulations: Number of parallel simulations to run
            enabled: If False, skip ABS and return mock metrics
        
        Returns:
            Tuple of (metrics_dict, output_dir_path) for compatibility with caller
        """
        if not enabled:
            logger.error("AgentSimulator disabled. Cannot run ABS simulation.")
            raise RuntimeError("AgentSimulator disabled; ABS simulation cannot run.")
        
        if not self.available:
            logger.error("AgentSimulator not available. Cannot run ABS simulation.")
            raise RuntimeError("AgentSimulator not available; ABS simulation cannot run.")
        
        # Use timestamp-based output dir to avoid conflicts between runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:15]
        log_basename = os.path.splitext(os.path.basename(log_path))[0]
        output_dir = os.path.join(os.path.dirname(log_path), f"agent_simulated_{log_basename}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        temp_modified_log = None
        
        try:
            import warnings
            warnings.filterwarnings("ignore")
            
            # Detect column names from the log
            df = pd.read_csv(log_path) if log_path.endswith('.csv') else pd.DataFrame()
            
            if df.empty and log_path.endswith('.csv'):
                logger.error(f"Could not read log file: {log_path}")
                raise RuntimeError(f"ABS input log is empty or unreadable: {log_path}")
            
            columns = set(df.columns.tolist()) if not df.empty else set()
            
            def find_column(patterns, fallback=None):
                """Find column matching pattern."""
                for col in columns:
                    if col.lower() in [p.lower() for p in patterns]:
                        return col
                for col in columns:
                    col_lower = col.lower()
                    for pattern in patterns:
                        if pattern.lower() in col_lower:
                            return col
                return fallback
            
            # Map column names
            case_col = find_column(['case_id', 'case'], 'case_id')
            activity_col = find_column(['activity_name', 'concept:name', 'activity'], 'activity_name')
            resource_col = find_column(['resource', 'org:resource', 'resourceid'], 'resource')
            end_ts_col = find_column(['end_timestamp', 'end_time', 'end'], 'end_timestamp')
            start_ts_col = find_column(['start_timestamp', 'start_time', 'start'], 'start_timestamp')
            
            # Validate columns actually exist in data (not just using fallbacks)
            detected_cols = {case_col, activity_col, resource_col, end_ts_col, start_ts_col}
            actual_cols = set(df.columns)
            missing_cols = detected_cols - actual_cols
            if missing_cols:
                logger.warning(f"⚠️  Using fallback column names for: {missing_cols}")
                logger.warning(f"   Available columns: {actual_cols}")
            else:
                logger.info(f"✓ All required columns detected: case={case_col}, activity={activity_col}, resource={resource_col}, end_ts={end_ts_col}, start_ts={start_ts_col}")

            # Apply modifications to a temporary log for ABS so hypotheses affect both simulators
            abs_log_path = log_path
            if modifications and log_path.endswith('.csv'):
                temp_modified_log = self._create_modified_abs_log(
                    log_path,
                    modifications,
                    case_col,
                    activity_col,
                    resource_col,
                    start_ts_col,
                    end_ts_col
                )
                if temp_modified_log:
                    abs_log_path = temp_modified_log
                    logger.info(f"Applied ABS log modifications using temp file: {abs_log_path}")
            else:
                # Even without modifications, convert timestamps to numeric for AgentSimulator compatibility
                if log_path.endswith('.csv'):
                    try:
                        df_for_abs = pd.read_csv(log_path)
                        df_for_abs = self._normalize_timestamp_columns(df_for_abs, start_ts_col, end_ts_col)
                        
                        temp_abs_log = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', prefix='abs_numeric_')
                        temp_abs_log.close()
                        df_for_abs.to_csv(temp_abs_log.name, index=False)
                        abs_log_path = temp_abs_log.name
                        temp_modified_log = temp_abs_log.name  # Track to cleanup later
                        logger.info(f"Normalized timestamps for AgentSimulator: {abs_log_path}")
                    except Exception as e:
                        logger.warning(f"Could not normalize timestamps: {e}. Using original log.")
                        temp_modified_log = None

            abs_log_basename = os.path.splitext(os.path.basename(abs_log_path))[0]
            
            logger.info(f"Using columns: case={case_col}, activity={activity_col}, resource={resource_col}, end_ts={end_ts_col}")
            
            # Prepare parameters for AgentSimulator
            column_names = {
                case_col: 'case_id',
                activity_col: 'activity_name',
                resource_col: 'resource',
                end_ts_col: 'end_timestamp',
                start_ts_col: 'start_timestamp'
            }
            
            params = {
                'discover_extr_delays': delay_discovery,
                'discover_parallel_work': False,
                'central_orchestration': orchestration_type == "centralized",
                'determine_automatically': orchestration_type == "auto",
                'PATH_LOG': abs_log_path,
                'PATH_LOG_test': None,
                'train_and_test': False,
                'column_names': column_names,
                'num_simulations': num_simulations,
                'output_dir': output_dir
            }

            # Prepare AgentSimulator default output tree to avoid mkdir failures in its internals
            try:
                default_main_results = os.path.join(os.getcwd(), "simulated_data", abs_log_basename, "main_results")
                os.makedirs(default_main_results, exist_ok=True)
            except Exception as prep_err:
                logger.debug(f"Could not pre-create default AgentSimulator output tree: {prep_err}")
            
            logger.info(f"Running AgentSimulator with params: {params}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"AgentSimulator module path: {self.agent_simulator_path}")
            
            # Log columns being used
            logger.info(f"Column mapping: case={case_col}, activity={activity_col}, resource={resource_col}, end_ts={end_ts_col}, start_ts={start_ts_col}")
            
            simulator = self.agent_simulator_module(params)
            logger.info(f"AgentSimulator initialized, about to call execute_pipeline()...")
            simulator.execute_pipeline()
            logger.info(f"AgentSimulator.execute_pipeline() completed successfully")
            
            # AgentSimulator outputs to simulated_data/LOG_NAME/main_results/ by default
            # Try to find output directory (search in multiple places in case AgentSimulator changes cwd)
            actual_output_dir = None
            
            possible_paths = [
                output_dir,
                os.path.join(os.getcwd(), "simulated_data", abs_log_basename, "main_results"),
                os.path.join(os.path.dirname(self.agent_simulator_path), "simulated_data", abs_log_basename, "main_results"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    csv_files = list(Path(path).glob("*.csv"))
                    if len(csv_files) > 0:
                        actual_output_dir = path
                        logger.info(f"Found AgentSimulator outputs in: {actual_output_dir}")
                        break
            
            if not actual_output_dir:
                logger.error(f"❌ AgentSimulator execution failed: No output files found")
                logger.error(f"   Searched paths: {possible_paths}")
                logger.error(f"   This means execute_pipeline() did not produce output CSVs")
                logger.error(f"   Check AgentSimulator logs above for errors")
                raise RuntimeError("AgentSimulator produced no output CSVs.")
            
            # Extract metrics from simulated logs
            metrics = self._extract_metrics_from_simulated_logs(
                actual_output_dir,
                log_path,
                time_horizon_seconds=time_horizon_seconds
            )
            logger.info(f"Extracted ABS metrics: {metrics}")
            
            result = (metrics, actual_output_dir)
            
            # NOTE: Do NOT cleanup output_dir here! The pipeline needs to compute time horizon from it
            # Cleanup will be done by the pipeline AFTER _compute_time_horizon_from_abs_outputs()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ AgentSimulator.simulate() threw exception: {type(e).__name__}: {e}")
            logger.error("   Full traceback:", exc_info=True)
            raise
        finally:
            if temp_modified_log and os.path.exists(temp_modified_log):
                try:
                    os.remove(temp_modified_log)
                except Exception:
                    pass

    def _create_modified_abs_log(
        self,
        log_path: str,
        modifications: Dict,
        case_col: str,
        activity_col: str,
        resource_col: str,
        start_ts_col: str,
        end_ts_col: str
    ) -> str:
        """Create a temporary CSV log with modifications applied for ABS simulation."""
        try:
            df = pd.read_csv(log_path)
            if df.empty:
                return None

            param_type = modifications.get('parameter_type')
            specific_changes = modifications.get('specific_changes', [])

            if not param_type or not specific_changes:
                return None

            if param_type == 'activity_durations':
                df = self._apply_activity_duration_modifications(
                    df,
                    specific_changes,
                    activity_col,
                    start_ts_col,
                    end_ts_col
                )
            elif param_type == 'resources':
                df = self._apply_resource_modifications(df, specific_changes, resource_col)
            elif param_type == 'role_redesign':
                df = self._apply_role_redesign_modifications(df, specific_changes, activity_col, resource_col)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', prefix='abs_modified_')
            temp_file.close()
            df = self._normalize_timestamp_columns(df, start_ts_col, end_ts_col)
            df.to_csv(temp_file.name, index=False)
            return temp_file.name
        except Exception as e:
            logger.warning(f"Could not apply ABS modifications to log: {e}")
            return None

    def apply_modifications_to_log(self, log_path: str, modifications: Dict) -> str:
        """Apply ABS modifications to a CSV log and return the new log path.

        Raises RuntimeError if the log cannot be read or modifications fail.
        """
        if not modifications:
            return log_path

        if not log_path.endswith(".csv"):
            raise RuntimeError("ABS log modifications require a CSV log.")

        df = pd.read_csv(log_path)
        if df.empty:
            raise RuntimeError(f"ABS log is empty or unreadable: {log_path}")

        columns = set(df.columns.tolist())

        def find_column(patterns, fallback=None):
            for col in columns:
                if col.lower() in [p.lower() for p in patterns]:
                    return col
            for col in columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern.lower() in col_lower:
                        return col
            return fallback

        case_col = find_column(['case_id', 'case'], 'case_id')
        activity_col = find_column(['activity_name', 'concept:name', 'activity'], 'activity_name')
        resource_col = find_column(['resource', 'org:resource', 'resourceid'], 'resource')
        end_ts_col = find_column(['end_timestamp', 'end_time', 'end'], 'end_timestamp')
        start_ts_col = find_column(['start_timestamp', 'start_time', 'start'], 'start_timestamp')

        modified_log = self._create_modified_abs_log(
            log_path,
            modifications,
            case_col,
            activity_col,
            resource_col,
            start_ts_col,
            end_ts_col
        )

        if not modified_log:
            raise RuntimeError("ABS log modification failed; no output log was created.")

        return modified_log

    def _normalize_timestamp_columns(
        self,
        df: pd.DataFrame,
        start_ts_col: str,
        end_ts_col: str
    ) -> pd.DataFrame:
        """Normalize timestamp columns to consistent non-empty strings."""
        df = df.copy()

        def normalize(col: str, fallback_col: str) -> None:
            if col not in df.columns:
                return
            try:
                dt_col = pd.to_datetime(df[col], errors='coerce', utc=True)
                if fallback_col in df.columns:
                    fallback_dt = pd.to_datetime(df[fallback_col], errors='coerce', utc=True)
                    dt_col = dt_col.fillna(fallback_dt)
                dt_col = dt_col.fillna(pd.Timestamp("1970-01-01", tz="UTC"))
                df[col] = dt_col.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            except Exception as e:
                logger.warning(f"Could not normalize {col} timestamp format: {e}")

        normalize(start_ts_col, end_ts_col)
        normalize(end_ts_col, start_ts_col)
        return df

    def _apply_activity_duration_modifications(
        self,
        df: pd.DataFrame,
        changes: list,
        activity_col: str,
        start_ts_col: str,
        end_ts_col: str
    ) -> pd.DataFrame:
        """Apply activity duration reductions by shifting end timestamps."""
        if activity_col not in df.columns or start_ts_col not in df.columns or end_ts_col not in df.columns:
            return df

        df = df.copy()
        df[start_ts_col] = pd.to_datetime(df[start_ts_col], errors='coerce')
        df[end_ts_col] = pd.to_datetime(df[end_ts_col], errors='coerce')

        for change in changes:
            target = str(change.get('target', ''))
            pct = float(change.get('value', 0) or 0)
            if pct <= 0:
                continue

            factor = max(0.1, 1.0 - (pct / 100.0))
            mask = df[activity_col].astype(str).str.contains(target, na=False) | (df[activity_col].astype(str) == target)
            if not mask.any():
                continue

            duration = (df.loc[mask, end_ts_col] - df.loc[mask, start_ts_col]).dt.total_seconds()
            duration = duration.fillna(0).clip(lower=0)
            new_duration_seconds = duration * factor
            df.loc[mask, end_ts_col] = df.loc[mask, start_ts_col] + pd.to_timedelta(new_duration_seconds, unit='s')

        return df

    def _apply_resource_modifications(self, df: pd.DataFrame, changes: list, resource_col: str) -> pd.DataFrame:
        """Apply resource changes by splitting workload onto synthetic resources."""
        if resource_col not in df.columns:
            return df

        df = df.copy()
        for change in changes:
            target = self._normalize_target_label(change.get('target', ''))
            change_type = str(change.get('change', ''))
            value = int(float(change.get('value', 0) or 0))
            if value <= 0:
                continue
            if not any(k in change_type for k in ['add', 'increase']):
                continue

            mask = df[resource_col].astype(str).str.contains(target, na=False) | (df[resource_col].astype(str) == target)
            idx = df.index[mask]
            if len(idx) == 0:
                continue

            # Deterministically split a portion to synthetic additional resources
            for i in range(value):
                synthetic_id = f"{target}_abs_extra_{i+1}"
                chosen_idx = idx[i::max(1, value + 1)]
                if len(chosen_idx) > 0:
                    df.loc[chosen_idx, resource_col] = synthetic_id

        return df

    def _apply_role_redesign_modifications(
        self,
        df: pd.DataFrame,
        changes: list,
        activity_col: str,
        resource_col: str
    ) -> pd.DataFrame:
        """Apply role redesign by creating specialist resources and shifting selected workloads."""
        if resource_col not in df.columns:
            return df

        df = df.copy()
        resources = df[resource_col].astype(str)
        utilization = resources.value_counts(normalize=True)
        underutilized = [r for r, u in utilization.items() if u < 0.05]

        for change in changes:
            target = self._normalize_target_label(change.get('target', ''))
            change_type = str(change.get('change', ''))
            raw_value = float(change.get('value', 0) or 0)

            if 'create_specialist' in change_type:
                mask = resources.str.contains(target, na=False) | (resources == target)
                idx = df.index[mask]
                if len(idx) == 0:
                    continue
                specialist_id = f"{target}_specialist_abs"
                chosen_idx = idx[::2]
                if len(chosen_idx) > 0:
                    df.loc[chosen_idx, resource_col] = specialist_id

            elif 'shift_responsibility' in change_type:
                shift_fraction = raw_value if 0 < raw_value <= 1 else 0.2
                source_mask = resources.str.contains(target, na=False) | (resources == target)
                source_idx = df.index[source_mask]
                if len(source_idx) == 0:
                    continue

                destination = underutilized[0] if underutilized else f"{target}_shifted_abs"
                shift_count = max(1, int(len(source_idx) * shift_fraction))
                chosen_idx = source_idx[:shift_count]
                df.loc[chosen_idx, resource_col] = destination

        # Optional activity scoping if available in change payload and column exists
        if activity_col in df.columns:
            return df
        return df

    @staticmethod
    def _normalize_target_label(target) -> str:
        """Normalize LLM-proposed target labels to actual resource ids in logs."""
        target_str = str(target or "").strip()
        if not target_str:
            return ""

        for separator in ["→", "->"]:
            if separator in target_str:
                target_str = target_str.split(separator)[0].strip()
                break

        if target_str.lower().endswith("_profile"):
            target_str = target_str[: -len("_profile")].strip()

        return target_str
    
    def _extract_metrics_from_simulated_logs(
        self,
        output_dir: str,
        original_log: str,
        time_horizon_seconds: float = None
    ) -> Dict:
        """
        Extract consolidated metrics from AgentSimulator outputs.
        Calculates: cycle time, throughput, waiting time, resource utilization.
        """
        try:
            # Find only actual simulation logs (avoid auxiliary CSV artifacts)
            simulated_logs = sorted(Path(output_dir).glob("simulated_log*.csv"))
            if not simulated_logs:
                simulated_logs = sorted(Path(output_dir).glob("simulation_log*.csv"))
            if not simulated_logs:
                simulated_logs = sorted(Path(output_dir).glob("*.csv"))
            
            if not simulated_logs:
                logger.error(f"❌ No CSV files found in AgentSimulator output directory: {output_dir}")
                logger.error(f"   Contents of output dir: {list(Path(output_dir).iterdir()) if Path(output_dir).exists() else 'DIR DOES NOT EXIST'}")
                raise RuntimeError("No CSV outputs found in AgentSimulator output directory.")
            
            logger.info(f"Found {len(simulated_logs)} AgentSimulator logs in {output_dir}")
            
            from simulator.metrics_extractor import MetricsExtractor
            extractor = MetricsExtractor()

            # Consolidate metrics across all logs
            all_cycle_times = []
            all_throughputs = []
            all_waiting_times = []
            resource_counts = {}
            
            for log_file in simulated_logs[:10]:  # Process all available (max 10)
                try:
                    metrics = extractor.extract_from_csv(str(log_file), time_horizon_seconds=time_horizon_seconds)
                    if not metrics:
                        continue
                    all_cycle_times.append(float(metrics.get('avg_cycle_time', 0.0)))
                    all_throughputs.append(float(metrics.get('avg_throughput', 0.0)))
                    all_waiting_times.append(float(metrics.get('avg_waiting_time', 0.0)))

                    for res, util in metrics.get('resource_utilization', {}).items():
                        resource_counts.setdefault(str(res), []).append(float(util))
                    
                except Exception as e:
                    logger.warning(f"Error extracting metrics from {log_file}: {e}")
                    continue
            
            # Average across all simulated logs
            if all_cycle_times:
                avg_cycle_time = sum(all_cycle_times) / len(all_cycle_times)
                avg_throughput = sum(all_throughputs) / len(all_throughputs)
                avg_waiting_time = sum(all_waiting_times) / len(all_waiting_times)
            else:
                logger.error(f"❌ Could not extract metrics from any of {len(simulated_logs)} simulated logs")
                raise RuntimeError("Failed to extract metrics from AgentSimulator outputs.")
            
            # Calculate resource utilization
            resource_utilization = {}
            for res, util_values in resource_counts.items():
                if util_values:
                    resource_utilization[res] = round(sum(util_values) / len(util_values), 4)
            
            metrics = {
                "avg_cycle_time": round(avg_cycle_time, 2),
                "avg_throughput": round(avg_throughput, 2),
                "avg_waiting_time": round(avg_waiting_time, 2),
                "simulated_case_count": int(round(avg_throughput)),
                "total_cost": 2160,
                "resource_utilization": resource_utilization,
                "num_agents": len(resource_utilization),
                "simulation_source": "AgentSimulator (ABS)"
            }
            
            logger.info(f"ABS metrics: cycle={avg_cycle_time:.0f}s, throughput={avg_throughput:.0f}, waiting={avg_waiting_time:.0f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error consolidating AgentSimulator metrics: {type(e).__name__}: {e}")
            logger.error("   Full traceback:", exc_info=True)
            raise
