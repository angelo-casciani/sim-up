"""SIMOD parameter calibration using pm4py ground truth metrics."""

import json
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple
import pm4py


class SIMODCalibrator:
    """Calibrate SIMOD discovered parameters using pm4py ground truth."""
    
    def __init__(self, log_path: str):
        """
        Initialize calibrator with event log.
        
        Args:
            log_path: Path to event log CSV
        """
        self.log_path = log_path
        self.execution_times = None
        self.activity_stats = None
        self.case_cycle_times = None
        
    def extract_baseline_metrics(self) -> Dict:
        """
        Extract baseline metrics from event log using pm4py.
        
        Returns:
            Dictionary with overall and activity-level metrics
        """
        # Load event log
        df = pd.read_csv(self.log_path)
        df['case_id'] = df['case_id'].astype(str)
        
        # Prepare for pm4py
        df_pm4py = df[['case_id', 'activity_name', 'start_timestamp', 'end_timestamp']].copy()
        df_pm4py.columns = ['case:concept:name', 'concept:name', 'time:timestamp', 'time:timestamp_end']
        df_pm4py['time:timestamp'] = pd.to_datetime(df_pm4py['time:timestamp'], format='mixed', utc=True)
        df_pm4py['time:timestamp_end'] = pd.to_datetime(df_pm4py['time:timestamp_end'], format='mixed', utc=True)
        
        event_log = pm4py.convert_to_event_log(df_pm4py)
        
        # Extract execution times by activity
        execution_times = defaultdict(list)
        for trace in event_log:
            for event in trace:
                if 'time:timestamp' in event and 'time:timestamp_end' in event:
                    duration = (event['time:timestamp_end'] - event['time:timestamp']).total_seconds()
                    activity = event['concept:name']
                    execution_times[activity].append(duration)
        
        # Calculate statistics
        activity_stats = {}
        all_durations = []
        for activity, durations in execution_times.items():
            stats = {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'mean': sum(durations) / len(durations),
                'median': sorted(durations)[len(durations)//2],
                'q90': sorted(durations)[int(len(durations)*0.9)]
            }
            activity_stats[activity] = stats
            all_durations.extend(durations)
        
        # Case cycle times
        case_durations = []
        for trace in event_log:
            if len(trace) > 0:
                start = min(e['time:timestamp'] for e in trace if 'time:timestamp' in e)
                end = max(e['time:timestamp_end'] for e in trace if 'time:timestamp_end' in e)
                duration = (end - start).total_seconds()
                case_durations.append(duration)
        
        overall_mean = sum(all_durations) / len(all_durations) if all_durations else 0
        
        self.execution_times = execution_times
        self.activity_stats = activity_stats
        self.case_cycle_times = case_durations
        
        return {
            'num_traces': len(event_log),
            'num_events': sum(len(trace) for trace in event_log),
            'num_activities': len(activity_stats),
            'overall_mean_duration': overall_mean,
            'overall_median_duration': sorted(all_durations)[len(all_durations)//2] if all_durations else 0,
            'case_mean_cycle_time': sum(case_durations) / len(case_durations) if case_durations else 0,
            'activity_stats': activity_stats
        }
    
    def calculate_calibration_factor(self, simod_model: Dict) -> Tuple[float, Dict]:
        """
        Calculate calibration factor from SIMOD discovered vs log actual.
        
        Args:
            simod_model: SIMOD discovered model JSON
            
        Returns:
            Tuple of (overall_factor, activity_factors)
        """
        if not self.activity_stats:
            raise ValueError("Must call extract_baseline_metrics first")
        
        # Get SIMOD durations - handle both flat and wrapped model structures
        simod_tasks = simod_model.get('task_resource_distribution', [])
        if not simod_tasks and 'prosimos_params' in simod_model:
            # Model is wrapped in prosimos_params dict (from SIMOD discoverer)
            simod_tasks = simod_model['prosimos_params'].get('task_resource_distribution', [])
        
        simod_durations = {}
        for task in simod_tasks:
            task_id = task.get('task_id', 'unknown')
            durations = []
            for res in task.get('resources', []):
                params = res.get('distribution_params', [])
                if params and params[0].get('value', 0) > 0:
                    durations.append(params[0]['value'])
            if durations:
                simod_durations[task_id] = sum(durations) / len(durations)
        
        # Calculate overall factor
        all_simod = list(simod_durations.values())
        overall_pm4py = sum(stats['mean'] for stats in self.activity_stats.values()) / len(self.activity_stats)
        overall_simod = sum(all_simod) / len(all_simod) if all_simod else overall_pm4py
        overall_factor = overall_pm4py / overall_simod if overall_simod > 0 else 1.0
        
        return overall_factor, {
            'overall_pm4py_mean': overall_pm4py,
            'overall_simod_mean': overall_simod,
            'factor': overall_factor
        }
    
    def calibrate_simod_model(self, simod_model: Dict) -> Tuple[Dict, float]:
        """
        Calibrate SIMOD model durations to match pm4py ground truth.
        
        Args:
            simod_model: SIMOD discovered model JSON
            
        Returns:
            Calibrated SIMOD model
        """
        if not self.activity_stats:
            raise ValueError("Must call extract_baseline_metrics first")
        
        factor, info = self.calculate_calibration_factor(simod_model)
        
        # Apply calibration - handle both flat and wrapped model structures
        calibrated = json.loads(json.dumps(simod_model))  # Deep copy
        
        # Get reference to the tasks dict (could be at top level or in prosimos_params)
        trd = None
        if 'task_resource_distribution' in calibrated:
            trd = calibrated['task_resource_distribution']
        elif 'prosimos_params' in calibrated and 'task_resource_distribution' in calibrated['prosimos_params']:
            trd = calibrated['prosimos_params']['task_resource_distribution']
        
        if trd:
            # Apply scaling factor to all distribution parameters
            for task in trd:
                for res in task.get('resources', []):
                    params = res.get('distribution_params', [])
                    if params:
                        # Scale all parameter values by factor
                        for param in params:
                            if param.get('value', 0) > 0:
                                param['value'] = param['value'] * factor
        
        return calibrated, factor


def extract_and_calibrate_baseline(log_path: str, simod_model_path: str) -> Tuple[Dict, Dict, float]:
    """
    Extract baseline metrics and calibrate SIMOD in one call.
    
    Args:
        log_path: Path to event log
        simod_model_path: Path to SIMOD discovered model JSON
        
    Returns:
        Tuple of (baseline_metrics, calibrated_model, calibration_factor)
    """
    # Load SIMOD model
    with open(simod_model_path) as f:
        simod_model = json.load(f)
    
    # Extract and calibrate
    calibrator = SIMODCalibrator(log_path)
    baseline_metrics = calibrator.extract_baseline_metrics()
    calibrated_model, factor = calibrator.calibrate_simod_model(simod_model)
    
    return baseline_metrics, calibrated_model, factor


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simod_calibrator.py <log_path> <simod_json_path>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    simod_path = sys.argv[2]
    
    baseline, calibrated, factor = extract_and_calibrate_baseline(log_path, simod_path)
    
    print(f"\n[OK] Baseline metrics extracted:")
    print(f"  - Cases: {baseline['num_traces']}")
    print(f"  - Events: {baseline['num_events']}")
    print(f"  - Activities: {baseline['num_activities']}")
    print(f"  - Mean activity duration: {baseline['overall_mean_duration']:.0f}s")
    print(f"  - Mean case cycle time: {baseline['case_mean_cycle_time']:.0f}s ({baseline['case_mean_cycle_time']/3600:.1f} hours)")
    
    print(f"\n[OK] Calibration factor: {factor:.2f}x")
    print(f"  SIMOD will be scaled by {factor:.2f} to match log ground truth")
    
    # Save calibrated model
    output_path = simod_path.replace('.json', '_calibrated.json')
    with open(output_path, 'w') as f:
        json.dump(calibrated, f, indent=2)
    
    print(f"\n[OK] Calibrated model saved to: {output_path}")
