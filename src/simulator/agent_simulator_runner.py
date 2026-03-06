import math
import os
import sys
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime
import tempfile
import warnings

from simulator.metrics_extractor import MetricsExtractor
from utils.column_utils import (
    find_column,
    normalize_role_label,
    parse_role_redesign_target,
    resolve_shift_fraction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentSimulatorRunner:
    """Runs AgentSimulator for agent-based simulation perspective (Organizational/Social View)."""

    def __init__(self):
        self.agent_simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AgentSimulator"))
        self.available = False
        self.agent_simulator_module = None
        if os.path.exists(self.agent_simulator_path):
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
        
        # Use timestamp-based output dir in temp folder to avoid polluting logs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:15]
        log_basename = os.path.splitext(os.path.basename(log_path))[0]
        import tempfile
        output_dir = os.path.join(tempfile.gettempdir(), f"agent_simulated_{log_basename}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        temp_modified_log = None

        try:
            warnings.filterwarnings("ignore")

            # Detect column names from the log
            df = pd.read_csv(log_path) if log_path.endswith('.csv') else pd.DataFrame()

            if df.empty and log_path.endswith('.csv'):
                logger.error(f"Could not read log file: {log_path}")
                raise RuntimeError(f"ABS input log is empty or unreadable: {log_path}")

            columns = df.columns.tolist() if not df.empty else []

            # Map column names using shared utility
            case_col = find_column(columns, ['case_id', 'case'], 'case_id')
            activity_col = find_column(columns, ['activity_name', 'concept:name', 'activity'], 'activity_name')
            resource_col = find_column(columns, ['resource', 'org:resource', 'resourceid'], 'resource')
            end_ts_col = find_column(columns, ['end_timestamp', 'end_time', 'end'], 'end_timestamp')
            start_ts_col = find_column(columns, ['start_timestamp', 'start_time', 'start'], 'start_timestamp')

            # Validate columns actually exist in data (not just using fallbacks)
            detected_cols = {case_col, activity_col, resource_col, end_ts_col, start_ts_col}
            actual_cols = set(df.columns)
            missing_cols = detected_cols - actual_cols
            if missing_cols:
                logger.warning(f"[!]  Using fallback column names for: {missing_cols}")
                logger.warning(f"   Available columns: {actual_cols}")
            else:
                logger.info(f"[OK] All required columns detected: case={case_col}, activity={activity_col}, resource={resource_col}, end_ts={end_ts_col}, start_ts={start_ts_col}")

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
            
            # Compute time-horizon-based case count so ABS generates a volume
            # comparable to DES for the same observation window.
            n_cases_override = None
            if time_horizon_seconds and time_horizon_seconds > 0 and not df.empty and case_col in df.columns:
                try:
                    ts_col_for_rate = start_ts_col if start_ts_col in df.columns else end_ts_col
                    if ts_col_for_rate in df.columns:
                        case_starts = pd.to_datetime(
                            df.groupby(case_col)[ts_col_for_rate].min(),
                            errors='coerce', utc=True
                        ).dropna()
                        if len(case_starts) >= 2:
                            log_span_seconds = (case_starts.max() - case_starts.min()).total_seconds()
                            if log_span_seconds > 0:
                                arrival_rate = len(case_starts) / log_span_seconds
                                n_cases_override = max(10, int(math.ceil(arrival_rate * time_horizon_seconds)))
                                logger.info(
                                    f"Time-horizon case count override: {n_cases_override} cases "
                                    f"(rate={arrival_rate:.5f} cases/s, horizon={time_horizon_seconds:.0f}s, "
                                    f"log_cases={len(case_starts)}, log_span={log_span_seconds:.0f}s)"
                                )
                except Exception as e:
                    logger.warning(f"Could not compute time-horizon case count: {e}")

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
                'output_dir': output_dir,
                'num_cases_to_simulate': n_cases_override,
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
                logger.error(f"[ERROR] AgentSimulator execution failed: No output files found")
                logger.error(f"   Searched paths: {possible_paths}")
                logger.error(f"   This means execute_pipeline() did not produce output CSVs")
                logger.error(f"   Check AgentSimulator logs above for errors")
                raise RuntimeError("AgentSimulator produced no output CSVs.")
            
            # Extract metrics from simulated logs.
            # Do NOT pass time_horizon_seconds here — ABS already generated the correct number
            # of cases based on the time horizon (via n_cases_override); post-hoc filtering
            # would double-filter and discard too many cases.
            metrics = self._extract_metrics_from_simulated_logs(
                actual_output_dir,
                log_path,
                time_horizon_seconds=None
            )
            logger.info(f"Extracted ABS metrics: {metrics}")
            
            result = (metrics, actual_output_dir)
            
            # NOTE: Do NOT cleanup output_dir here! The pipeline needs to compute time horizon from it
            # Cleanup will be done by the pipeline AFTER _compute_time_horizon_from_abs_outputs()
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] AgentSimulator.simulate() threw exception: {type(e).__name__}: {e}")
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
                df = self._apply_resource_modifications(df, specific_changes, case_col, resource_col)
            elif param_type == 'role_redesign':
                df = self._apply_role_redesign_modifications(df, specific_changes, case_col, activity_col, resource_col)
            elif param_type == 'calendar':
                df = self._apply_calendar_modifications(df, specific_changes, start_ts_col, end_ts_col)

            df = self._sync_resource_columns(df, resource_col)

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

        columns = df.columns.tolist()

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
            param_type = modifications.get('parameter_type')
            logger.warning(
                f"ABS log modification returned no output for param_type={param_type!r}; "
                "using original log unchanged."
            )
            return log_path

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

    def _apply_resource_modifications(
        self,
        df: pd.DataFrame,
        changes: list,
        case_col: str,
        resource_col: str
    ) -> pd.DataFrame:
        """Apply resource changes by splitting workload onto synthetic resources at case level.

        Cases where the target resource appears are distributed evenly across the original
        resource and N synthetic copies.  All events in a given case stay with one resource,
        preserving within-case routing coherence for AgentSimulator.
        """
        if resource_col not in df.columns:
            return df

        df = df.copy()
        for change in changes:
            target = normalize_role_label(change.get('target', ''))
            change_type = str(change.get('change', ''))
            value = int(float(change.get('value', 0) or 0))
            if value <= 0:
                continue
            if not any(k in change_type for k in ['add', 'increase']):
                continue

            if case_col not in df.columns:
                # Fallback: original row-level split when case column is unavailable
                mask = df[resource_col].astype(str).str.contains(target, na=False) | (df[resource_col].astype(str) == target)
                idx = df.index[mask]
                if len(idx) == 0:
                    continue
                for i in range(value):
                    synthetic_id = f"{target}_abs_extra_{i+1}"
                    chosen_idx = idx[i::max(1, value + 1)]
                    if len(chosen_idx) > 0:
                        df.loc[chosen_idx, resource_col] = synthetic_id
                continue

            # Case-level split: find cases that involve this resource and distribute
            # them evenly across (1 original + value synthetic) pools.
            mask = df[resource_col].astype(str).str.contains(target, na=False) | (df[resource_col].astype(str) == target)
            cases_with_target = df.loc[mask, case_col].unique()
            if len(cases_with_target) == 0:
                continue

            total_pools = value + 1  # original + N synthetics
            for i in range(value):
                synthetic_id = f"{target}_abs_extra_{i+1}"
                # Assign every (i+1)-th case (0-indexed) to this synthetic resource
                assigned_cases = cases_with_target[i + 1::total_pools]
                if len(assigned_cases) == 0:
                    continue
                case_mask = df[case_col].isin(assigned_cases)
                resource_mask = df[resource_col].astype(str).str.contains(target, na=False) | (df[resource_col].astype(str) == target)
                df.loc[case_mask & resource_mask, resource_col] = synthetic_id

        return df

    def _apply_role_redesign_modifications(
        self,
        df: pd.DataFrame,
        changes: list,
        case_col: str,
        activity_col: str,
        resource_col: str
    ) -> pd.DataFrame:
        """Apply role redesign by shifting workload between resources at case level.

        For shift_responsibility: all events in a selected subset of cases where the
        source resource appears are reassigned to the destination resource.  Operating at
        case granularity preserves within-case routing sequences so AgentSimulator can
        learn coherent agent behaviours from the modified log.

        For create_specialist: selected cases are handled by a synthetic specialist agent.
        """
        if resource_col not in df.columns:
            return df

        df = df.copy()

        for change in changes:
            source_target, destination_target = parse_role_redesign_target(change.get('target', ''))
            target = source_target
            change_type = str(change.get('change', ''))
            # Read value from the correct key; use it to derive shift fraction
            shift_fraction = resolve_shift_fraction(change.get('value'))

            resource_mask = (
                df[resource_col].astype(str).str.contains(target, na=False) |
                (df[resource_col].astype(str) == target)
            )

            if 'create_specialist' in change_type:
                if case_col in df.columns:
                    cases_with_source = df.loc[resource_mask, case_col].unique()
                    shift_count = max(1, int(round(len(cases_with_source) * shift_fraction)))
                    chosen_cases = cases_with_source[::max(1, len(cases_with_source) // shift_count)][:shift_count]
                    specialist_id = f"{target}_specialist_abs"
                    case_mask = df[case_col].isin(chosen_cases)
                    df.loc[case_mask & resource_mask, resource_col] = specialist_id
                else:
                    # Fallback: row-level every-other-event
                    idx = df.index[resource_mask]
                    if len(idx) > 0:
                        specialist_id = f"{target}_specialist_abs"
                        df.loc[idx[::2], resource_col] = specialist_id

            elif 'shift_responsibility' in change_type:
                destination = destination_target if destination_target else f"{target}_shifted_abs"

                if case_col in df.columns:
                    cases_with_source = df.loc[resource_mask, case_col].unique()
                    if len(cases_with_source) == 0:
                        continue
                    shift_count = max(1, int(round(len(cases_with_source) * shift_fraction)))
                    shift_count = min(shift_count, len(cases_with_source))
                    # Take the first shift_count cases deterministically
                    chosen_cases = cases_with_source[:shift_count]
                    case_mask = df[case_col].isin(chosen_cases)
                    df.loc[case_mask & resource_mask, resource_col] = destination
                else:
                    # Fallback: row-level shift
                    source_idx = df.index[resource_mask]
                    source_count = len(source_idx)
                    if source_count == 0:
                        continue
                    row_shift_count = max(1, int(round(source_count * shift_fraction)))
                    row_shift_count = min(row_shift_count, source_count)
                    df.loc[source_idx[:row_shift_count], resource_col] = destination

        return df

    def _apply_calendar_modifications(
        self,
        df: pd.DataFrame,
        changes: list,
        start_ts_col: str,
        end_ts_col: str
    ) -> pd.DataFrame:
        """Apply calendar changes (increase_hours) by compressing activity durations across all events.

        Simulates extended working hours by scaling down durations proportionally:
        more available hours per day means work can be distributed more densely,
        equivalent to a uniform throughput improvement.
        Base working day is assumed to be 8 hours; increasing by N hours yields
        a compression factor of 8 / (8 + N).
        """
        if start_ts_col not in df.columns or end_ts_col not in df.columns:
            return df

        df = df.copy()
        df[start_ts_col] = pd.to_datetime(df[start_ts_col], errors='coerce')
        df[end_ts_col] = pd.to_datetime(df[end_ts_col], errors='coerce')

        for change in changes:
            change_type = str(change.get('change', ''))
            if 'increase_hours' not in change_type:
                continue
            try:
                extra_hours = float(change.get('value', 0) or 0)
            except (TypeError, ValueError):
                continue
            if extra_hours <= 0:
                continue

            base_hours = 8.0
            factor = base_hours / (base_hours + extra_hours)
            factor = max(0.5, min(1.0, factor))  # clamp to sensible range

            duration = (df[end_ts_col] - df[start_ts_col]).dt.total_seconds()
            duration = duration.fillna(0).clip(lower=0)
            new_duration_seconds = duration * factor
            df[end_ts_col] = df[start_ts_col] + pd.to_timedelta(new_duration_seconds, unit='s')

        return df

    @staticmethod
    def _sync_resource_columns(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
        """Keep common resource columns synchronized after ABS modifications."""
        if source_col not in df.columns:
            return df

        aliases = ['resource', 'org:resource']
        for alias in aliases:
            if alias in df.columns and alias != source_col:
                df[alias] = df[source_col]
        return df

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
                logger.error(f"[ERROR] No CSV files found in AgentSimulator output directory: {output_dir}")
                logger.error(f"   Contents of output dir: {list(Path(output_dir).iterdir()) if Path(output_dir).exists() else 'DIR DOES NOT EXIST'}")
                raise RuntimeError("No CSV outputs found in AgentSimulator output directory.")
            
            logger.info(f"Found {len(simulated_logs)} AgentSimulator logs in {output_dir}")
        
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
                logger.error(f"[ERROR] Could not extract metrics from any of {len(simulated_logs)} simulated logs")
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
                "resource_utilization": resource_utilization,
                "num_agents": len(resource_utilization),
                "simulation_source": "AgentSimulator (ABS)"
            }
            
            logger.info(f"ABS metrics: cycle={avg_cycle_time:.0f}s, throughput={avg_throughput:.0f}, waiting={avg_waiting_time:.0f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"[ERROR] Error consolidating AgentSimulator metrics: {type(e).__name__}: {e}")
            logger.error("   Full traceback:", exc_info=True)
            raise
