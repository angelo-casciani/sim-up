"""
Log Processor for O2C Event Logs

Transforms raw O2C event logs into a format compatible with Prosimos and AgentSimulator.
Follows the same processing used by the agent simulator to create logs like LoanApp.csv.

Output format columns:
- case_id: Unique case identifier (integer)
- resource: Resource/agent identifier
- activity_name: Name of the activity
- start_timestamp: Activity start time (ISO format with timezone)
- end_timestamp: Activity end time (ISO format with timezone)
- agent: Integer agent ID
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect column mappings from raw event log.

    Returns a dictionary mapping standard names to actual column names.
    """
    column_candidates = {
        'case_id': ['c_caseid', 'case_id', 'case:concept:name', 'CaseID'],
        'activity_name': ['c_eventname', 'activity_name', 'activity', 'concept:name', 'Activity'],
        'timestamp': ['c_time', 'timestamp', 'time:timestamp', 'end_timestamp', 'Timestamp'],
        'resource': ['EventCreatedByUserId', 'resource', 'org:resource', 'Resource'],
    }

    columns = df.columns.tolist()
    mapping = {}

    for standard_name, candidates in column_candidates.items():
        for candidate in candidates:
            if candidate in columns:
                mapping[standard_name] = candidate
                break

    return mapping


def clean_resource_id(resource_id: str) -> str:
    """
    Clean resource identifier by removing memory address patterns.
    """
    if pd.isna(resource_id):
        return "Unknown"

    resource_str = str(resource_id)

    # Remove memory address patterns like "<memory at 0x7f3c348657c8>"
    if resource_str.startswith("<memory at"):
        return "SystemAgent"

    # Extract meaningful part from long identifiers
    if len(resource_str) > 30 and resource_str.startswith("100"):
        # These are likely composite IDs, extract a shorter hash
        return resource_str[:12]

    return resource_str


def create_agent_mapping(df: pd.DataFrame, resource_col: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mapping between resource names and agent IDs.
    """
    unique_resources = df[resource_col].apply(clean_resource_id).unique()
    resource_to_agent = {res: idx for idx, res in enumerate(sorted(unique_resources))}
    agent_to_resource = {idx: res for res, idx in resource_to_agent.items()}
    return resource_to_agent, agent_to_resource




def parse_timestamp(ts) -> pd.Timestamp:
    """
    Parse timestamp from various formats.
    """
    if pd.isna(ts):
        return pd.NaT

    try:
        parsed = pd.to_datetime(ts, utc=True)
        return parsed
    except:
        return pd.NaT


def process_raw_log(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Process a raw O2C event log and transform it to Prosimos/AgentSimulator format.

    Args:
        input_path: Path to the raw CSV file
        output_path: Path where processed CSV will be saved

    Returns:
        Processed DataFrame
    """
    print(f"Processing: {input_path}")

    # Read raw log
    df = pd.read_csv(input_path)

    # Detect columns
    col_mapping = detect_columns(df)
    print(f"  Detected columns: {col_mapping}")

    # Verify required columns exist
    required = ['case_id', 'activity_name', 'timestamp']
    for req in required:
        if req not in col_mapping:
            raise ValueError(f"Could not find required column: {req}. Available: {df.columns.tolist()}")

    # Create working dataframe with standardized columns
    processed = pd.DataFrame()

    # Map case_id - convert to integer representation
    case_ids = df[col_mapping['case_id']].astype(str)
    unique_cases = sorted(case_ids.unique())
    case_id_map = {case: idx for idx, case in enumerate(unique_cases)}
    processed['case_id'] = case_ids.map(case_id_map)

    # Map activity_name
    processed['activity_name'] = df[col_mapping['activity_name']].astype(str)

    # Parse and sort by timestamp
    processed['timestamp'] = df[col_mapping['timestamp']].apply(parse_timestamp)

    # Handle resource column
    if 'resource' in col_mapping:
        processed['resource_raw'] = df[col_mapping['resource']].apply(clean_resource_id)
    else:
        processed['resource_raw'] = 'SystemAgent'

    # Remove rows with invalid timestamps
    processed = processed.dropna(subset=['timestamp'])

    # Sort by case_id and timestamp
    processed = processed.sort_values(['case_id', 'timestamp']).reset_index(drop=True)

    # Create start and end timestamps
    # Events are treated as instantaneous (start = end)
    processed['start_timestamp'] = processed['timestamp']
    processed['end_timestamp'] = processed['timestamp']

    # Create agent mapping
    resource_to_agent, agent_to_resource = create_agent_mapping(processed, 'resource_raw')
    processed['agent'] = processed['resource_raw'].map(resource_to_agent)

    # Create resource column with cleaned names
    processed['resource'] = processed['resource_raw']

    # Format timestamps to ISO format with timezone
    processed['start_timestamp'] = processed['start_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    processed['end_timestamp'] = processed['end_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

    # Select and order final columns (matching LoanApp.csv format)
    final_columns = ['case_id', 'resource', 'activity_name', 'start_timestamp', 'end_timestamp', 'agent']
    result = processed[final_columns].copy()

    # Remove duplicate consecutive events (same case, activity, and timestamp)
    result = result.drop_duplicates(subset=['case_id', 'activity_name', 'start_timestamp'])

    # Save to output
    result.to_csv(output_path, index=False)
    print(f"  Saved processed log to: {output_path}")
    print(f"  Cases: {result['case_id'].nunique()}, Events: {len(result)}, Resources: {result['resource'].nunique()}")

    return result


def process_all_o2c_logs(logs_dir: str, output_suffix: str = 'processed') -> Dict[str, pd.DataFrame]:
    """
    Process all raw O2C logs in the given directory.

    Args:
        logs_dir: Path to the o2c logs directory
        output_suffix: Suffix for output files (e.g., 'processed' -> 'processed_742427502.csv')

    Returns:
        Dictionary mapping log IDs to processed DataFrames
    """
    logs_dir = Path(logs_dir)
    results = {}

    # Find all subdirectories (each representing a log)
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue

        log_id = subdir.name

        # Look for raw CSV file
        raw_file = None
        for csv_file in subdir.glob('*.csv'):
            # Prefer files starting with 'raw_'
            if csv_file.name.startswith('raw_'):
                raw_file = csv_file
                break
            # Otherwise use the first CSV that's not already processed
            elif not csv_file.name.startswith('processed_'):
                raw_file = csv_file

        if raw_file is None:
            print(f"Skipping {log_id}: No raw CSV found")
            continue

        # Process and save
        output_file = subdir / f"{output_suffix}_{log_id}.csv"

        try:
            result = process_raw_log(str(raw_file), str(output_file))
            results[log_id] = result
        except Exception as e:
            print(f"Error processing {log_id}: {e}")
            continue

    return results


def main():
    """
    Main entry point for processing O2C logs.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process O2C event logs for Prosimos/AgentSimulator')
    parser.add_argument('--logs-dir', type=str,
                        default='logs/o2c',
                        help='Directory containing O2C log folders')
    parser.add_argument('--single-log', type=str,
                        help='Process a single log folder by ID (e.g., 742427502)')
    parser.add_argument('--output-suffix', type=str,
                        default='processed',
                        help='Suffix for output files')

    args = parser.parse_args()

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # src/evaluation/qualitative -> project root

    logs_dir = project_root / args.logs_dir

    if args.single_log:
        log_path = logs_dir / args.single_log
        if not log_path.exists():
            print(f"Log folder not found: {log_path}")
            return

        # Find raw file
        raw_files = list(log_path.glob('raw_*.csv')) or list(log_path.glob('*.csv'))
        if not raw_files:
            print(f"No CSV files found in {log_path}")
            return

        output_file = log_path / f"{args.output_suffix}_{args.single_log}.csv"
        process_raw_log(str(raw_files[0]), str(output_file))
    else:
        print(f"Processing all logs in: {logs_dir}")
        results = process_all_o2c_logs(str(logs_dir), args.output_suffix)
        print(f"\nProcessed {len(results)} logs successfully")


if __name__ == '__main__':
    main()
