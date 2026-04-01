"""
Log Processor for O2C Event Logs

Transforms raw O2C event logs into a format compatible with Prosimos and AgentSimulator.
Follows the same processing used by the agent simulator to create logs like LoanApp.csv.

Output format columns:
- case_id: Unique case identifier (integer)
- resource: Resource/agent identifier (anonymized role-based name like BillingClerk_001)
- activity_name: Name of the activity
- start_timestamp: Activity start time (ISO format with timezone)
- end_timestamp: Activity end time (ISO format with timezone)
- agent: Integer agent ID

Key design decisions:
- Each unique original resource ID (including memory addresses) maps to ONE anonymized resource
- Role is derived from the ACTIVITIES that resource performs (not per-event)
- This preserves the original resource structure while anonymizing
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Set


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


def derive_role_from_activities(activities: Set[str]) -> str:
    """
    Derive a role/resource category from the SET of activities a resource performs.
    This looks at ALL activities the resource does, not just one event.
    """
    activities_lower = [a.lower() for a in activities]
    all_acts = ' '.join(activities_lower)

    # Priority-based role assignment based on activity patterns
    # Invoice/billing activities take priority
    if 'invoice' in all_acts or 'billing' in all_acts:
        return 'BillingClerk'
    # Warehouse/delivery activities
    if any(kw in all_acts for kw in ['delivery', 'picking', 'goods']):
        return 'WarehouseStaff'
    # Sales activities
    if any(kw in all_acts for kw in ['sales', 'order', 'quotation']):
        return 'SalesRep'
    # Credit/return activities
    if any(kw in all_acts for kw in ['credit', 'return']):
        return 'CreditAnalyst'

    return 'ProcessAgent'


def create_resource_mapping(df: pd.DataFrame, resource_col: str, activity_col: str) -> Dict[str, str]:
    """
    Create a mapping from original resource IDs to anonymized role-based names.

    Process:
    1. Group events by original resource ID
    2. For each resource, collect all activities they perform
    3. Derive role from activity patterns
    4. Create anonymized name like BillingClerk_001, WarehouseStaff_002, etc.

    Returns:
        Dict mapping original resource ID -> anonymized resource name
    """
    # Group activities by resource
    resource_activities = df.groupby(resource_col)[activity_col].apply(set)

    # Derive role for each resource
    resource_roles = resource_activities.apply(derive_role_from_activities)

    # Count resources per role for numbering
    role_counters = {}
    resource_mapping = {}

    # Sort resources by their first appearance in the log for consistent numbering
    first_appearance = df.groupby(resource_col).first().reset_index()
    sorted_resources = first_appearance.sort_values('timestamp' if 'timestamp' in first_appearance.columns else activity_col)[resource_col].tolist()

    # If sorting failed, just use the order from resource_roles
    if not sorted_resources:
        sorted_resources = resource_roles.index.tolist()

    for orig_resource in sorted_resources:
        if orig_resource not in resource_roles.index:
            continue
        role = resource_roles[orig_resource]

        if role not in role_counters:
            role_counters[role] = 0
        role_counters[role] += 1

        # Create anonymized name with zero-padded number
        anon_name = f"{role}_{role_counters[role]:03d}"
        resource_mapping[orig_resource] = anon_name

    return resource_mapping


def derive_role_from_activity(activity: str) -> str:
    """
    Derive a role/resource category from a single activity name.
    Used as fallback when resource-level analysis isn't possible.
    """
    activity_lower = activity.lower()

    # Pattern matching based on O2C domain keywords
    if any(kw in activity_lower for kw in ['sales order', 'quotation']):
        return 'SalesRep'
    if any(kw in activity_lower for kw in ['delivery', 'picking', 'goods issue', 'goods receipt', 'goods movement']):
        return 'WarehouseStaff'
    if any(kw in activity_lower for kw in ['invoice', 'billing', 'debit memo', 'pro forma']):
        return 'BillingClerk'
    if any(kw in activity_lower for kw in ['credit', 'return']):
        return 'CreditAnalyst'

    return 'ProcessAgent'


def assign_resource_from_activity(activity: str, case_id: str, num_per_role: int = 3, role_resource_counts: Dict[str, int] = None) -> str:
    """
    Assign a specific resource based on activity pattern and case context.
    Uses deterministic hashing so same case uses consistent resources.
    DEPRECATED: Use create_resource_mapping instead for better resource preservation.

    Args:
        activity: Activity name
        case_id: Case identifier
        num_per_role: Default number of resources per role (used if role_resource_counts not provided)
        role_resource_counts: Optional dict mapping role -> number of resources for that role
    """
    import hashlib

    role = derive_role_from_activity(activity)

    # Use role-specific count if provided, otherwise use default
    if role_resource_counts and role in role_resource_counts:
        n_resources = role_resource_counts[role]
    else:
        n_resources = num_per_role

    # Handle edge case of 0 resources (shouldn't happen but be safe)
    if n_resources <= 0:
        n_resources = 1

    # Hash case_id to distribute work across agents in the same role
    case_hash = int(hashlib.md5(str(case_id).encode()).hexdigest()[:8], 16)
    agent_num = (case_hash % n_resources) + 1

    return f"{role}_{agent_num}"


def calculate_proportional_resources(
    df: pd.DataFrame,
    activity_col: str,
    total_resources: int,
    min_resources_per_role: int = 1
) -> Dict[str, int]:
    """
    Calculate number of resources per role proportional to their workload.

    Args:
        df: DataFrame with activity data
        activity_col: Name of the activity column
        total_resources: Total number of resources to distribute
        min_resources_per_role: Minimum resources for any role (default 1)

    Returns:
        Dict mapping role name -> number of resources
    """
    # Count events per role
    role_events = {}
    for activity in df[activity_col]:
        role = derive_role_from_activity(activity)
        role_events[role] = role_events.get(role, 0) + 1

    total_events = sum(role_events.values())
    n_roles = len(role_events)

    # Calculate proportional allocation
    # Reserve minimum for each role first
    remaining = total_resources - (n_roles * min_resources_per_role)
    if remaining < 0:
        remaining = 0
        min_resources_per_role = max(1, total_resources // n_roles)

    role_resources = {}
    for role, events in role_events.items():
        proportion = events / total_events
        # Allocate minimum plus proportional share of remaining
        allocated = min_resources_per_role + int(remaining * proportion)
        role_resources[role] = max(1, allocated)

    # Adjust to hit exact total (rounding may cause slight differences)
    current_total = sum(role_resources.values())
    if current_total != total_resources:
        # Add/remove from highest workload role
        highest_role = max(role_events.keys(), key=lambda r: role_events[r])
        role_resources[highest_role] += (total_resources - current_total)
        role_resources[highest_role] = max(1, role_resources[highest_role])

    return role_resources


def create_agent_mapping(resources: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mapping between resource names and agent IDs.
    """
    unique_resources = sorted(set(resources))
    resource_to_agent = {res: idx for idx, res in enumerate(unique_resources)}
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


def filter_infrequent_activities(
    df: pd.DataFrame,
    activity_col: str,
    min_frequency_pct: float = 1.0
) -> pd.DataFrame:
    """
    Filter out activities that occur less than min_frequency_pct of total events.

    This reduces BPMN complexity for Prosimos simulation - logs with 60+ activities
    often cause simulation timeouts due to state space explosion in the BPMN model.

    Args:
        df: Event log DataFrame
        activity_col: Name of the activity column
        min_frequency_pct: Minimum percentage of events an activity must have (default 1%)

    Returns:
        Filtered DataFrame with only frequent activities
    """
    original_activities = df[activity_col].nunique()
    original_events = len(df)

    # Calculate activity frequencies
    activity_counts = df[activity_col].value_counts()
    min_count = (min_frequency_pct / 100.0) * original_events

    # Keep activities that meet the threshold
    frequent_activities = activity_counts[activity_counts >= min_count].index.tolist()
    df_filtered = df[df[activity_col].isin(frequent_activities)].copy()

    filtered_activities = df_filtered[activity_col].nunique()
    removed_count = original_activities - filtered_activities

    if removed_count > 0:
        print(f"  [FILTER] Removed {removed_count} infrequent activities ({original_activities} -> {filtered_activities})")
        print(f"  [FILTER] Events retained: {len(df_filtered)}/{original_events} ({len(df_filtered)/original_events*100:.1f}%)")

    return df_filtered


def process_raw_log(
    input_path: str,
    output_path: str,
    filter_activities: bool = False,
    min_activity_frequency_pct: float = 0.5,
    target_events: int = None,
    max_cases: int = None,
    preserve_resources: bool = True,
    filter_singletons: bool = False,
    max_resources_per_role: int = None,
    num_resources_per_role: int = 3,
    total_resources: int = None,
    min_resources_per_role: int = 1
) -> pd.DataFrame:
    """
    Process a raw O2C event log and transform it to Prosimos/AgentSimulator format.

    Key features:
    - Preserves original resource structure (each unique resource ID -> one anonymized resource)
    - Derives roles from activities each resource performs
    - Anonymizes resources as Role_001, Role_002, etc.
    - Removes CompanyCodeName for anonymization

    Args:
        input_path: Path to the raw CSV file
        output_path: Path where processed CSV will be saved
        filter_activities: Whether to filter infrequent activities (default False to keep complexity)
        min_activity_frequency_pct: Minimum activity frequency threshold (default 0.5%)
        target_events: Target number of events (samples cases to achieve this, default None = all)
        max_cases: Maximum number of cases to include (None = all cases)
        preserve_resources: If True, preserve original resource IDs and anonymize them.
                           If False, use old behavior of deriving from activities.
        filter_singletons: If True, remove cases with only 1 event (no process flow)
        max_resources_per_role: Maximum resources to keep per role (keeps most active ones)
        num_resources_per_role: Fixed number of resources per role (used if total_resources not set)
        total_resources: Total resources to distribute proportionally across roles (overrides num_resources_per_role)
        min_resources_per_role: Minimum resources per role when using proportional allocation (default 1)

    Returns:
        Processed DataFrame
    """
    print(f"Processing: {input_path}")

    # Read raw log
    df = pd.read_csv(input_path)
    print(f"  Raw log: {len(df)} events, {df[detect_columns(df).get('case_id', 'c_caseid')].nunique()} cases")

    # Detect columns
    col_mapping = detect_columns(df)
    print(f"  Detected columns: {col_mapping}")

    # Verify required columns exist
    required = ['case_id', 'activity_name', 'timestamp']
    for req in required:
        if req not in col_mapping:
            raise ValueError(f"Could not find required column: {req}. Available: {df.columns.tolist()}")

    # Calculate sampling if target_events specified
    if target_events is not None and max_cases is None:
        total_events = len(df)
        total_cases = df[col_mapping['case_id']].nunique()
        avg_events_per_case = total_events / total_cases
        estimated_cases = int(target_events / avg_events_per_case)
        max_cases = min(estimated_cases, total_cases)
        print(f"  [TARGET] Targeting ~{target_events} events -> sampling ~{max_cases} cases")

    # Sample cases if max_cases specified
    if max_cases is not None:
        unique_cases = df[col_mapping['case_id']].unique()
        if len(unique_cases) > max_cases:
            # Sample cases randomly with fixed seed for reproducibility
            np.random.seed(42)
            sampled_cases = np.random.choice(unique_cases, size=max_cases, replace=False)
            df = df[df[col_mapping['case_id']].isin(sampled_cases)].copy()
            print(f"  [SAMPLE] Sampled {max_cases} cases from {len(unique_cases)} ({max_cases/len(unique_cases)*100:.1f}%)")
            print(f"  [SAMPLE] Events after sampling: {len(df)}")

    # Filter singleton cases (cases with only 1 event - no process flow)
    if filter_singletons:
        case_col = col_mapping['case_id']
        events_per_case = df.groupby(case_col).size()
        singleton_cases = events_per_case[events_per_case == 1].index
        multi_event_cases = events_per_case[events_per_case > 1].index

        n_singletons = len(singleton_cases)
        n_multi = len(multi_event_cases)

        if n_singletons > 0:
            df = df[df[case_col].isin(multi_event_cases)].copy()
            print(f"  [FILTER] Removed {n_singletons} singleton cases (1 event each)")
            print(f"  [FILTER] Remaining: {n_multi} cases, {len(df)} events")

    # Filter infrequent activities (optional, disabled by default to keep complexity)
    if filter_activities:
        n_activities_before = df[col_mapping['activity_name']].nunique()
        if n_activities_before > 30:  # Only filter if many activities
            df = filter_infrequent_activities(
                df,
                col_mapping['activity_name'],
                min_activity_frequency_pct
            )

    # Create working dataframe with standardized columns
    processed = pd.DataFrame()

    # Keep original resource IDs for mapping
    processed['orig_resource'] = df[col_mapping['resource']].astype(str)

    # Map case_id - convert to integer representation
    case_ids = df[col_mapping['case_id']].astype(str)
    unique_cases = sorted(case_ids.unique())
    case_id_map = {case: idx for idx, case in enumerate(unique_cases)}
    processed['case_id'] = case_ids.map(case_id_map)

    # Map activity_name
    processed['activity_name'] = df[col_mapping['activity_name']].astype(str)

    # Parse timestamp
    processed['timestamp'] = df[col_mapping['timestamp']].apply(parse_timestamp)

    # Remove rows with invalid timestamps
    processed = processed.dropna(subset=['timestamp'])

    # Handle resource mapping
    if preserve_resources and 'resource' in col_mapping:
        # NEW APPROACH: Map each unique original resource to an anonymized role-based name
        print(f"  Creating resource mapping (preserving original resource structure)...")
        resource_mapping = create_resource_mapping(
            processed, 'orig_resource', 'activity_name'
        )
        processed['resource'] = processed['orig_resource'].map(resource_mapping)

        # Report resource mapping stats
        role_counts = {}
        for anon_name in resource_mapping.values():
            role = anon_name.rsplit('_', 1)[0]
            role_counts[role] = role_counts.get(role, 0) + 1
        print(f"  Resource mapping: {len(resource_mapping)} unique resources")
        for role, count in sorted(role_counts.items()):
            print(f"    - {role}: {count} resources")

        # Limit resources per role if specified (keep most active ones)
        if max_resources_per_role is not None:
            print(f"  Limiting to {max_resources_per_role} resources per role...")

            # Count events per resource
            resource_event_counts = processed.groupby('resource').size()

            # Group resources by role and keep top N per role
            resources_to_keep = []
            for role in role_counts.keys():
                role_resources = [r for r in resource_event_counts.index if r.startswith(role + '_')]
                role_counts_sorted = resource_event_counts[role_resources].sort_values(ascending=False)
                top_resources = role_counts_sorted.head(max_resources_per_role).index.tolist()
                resources_to_keep.extend(top_resources)
                print(f"    - {role}: keeping {len(top_resources)} of {len(role_resources)} (top by activity)")

            # Filter to keep only events from top resources
            events_before = len(processed)
            processed = processed[processed['resource'].isin(resources_to_keep)].copy()
            events_after = len(processed)
            print(f"  [RESOURCE FILTER] Kept {len(resources_to_keep)} resources, {events_after} events ({events_after/events_before*100:.1f}%)")
    else:
        # OLD APPROACH: Derive resources from activity names only (fewer resources)
        if total_resources is not None:
            # PROPORTIONAL: Allocate resources based on workload per role
            print(f"  Deriving resources proportionally ({total_resources} total, min {min_resources_per_role} per role)")
            role_resource_counts = calculate_proportional_resources(
                processed, 'activity_name', total_resources, min_resources_per_role
            )
            print(f"  Resource allocation by role:")
            for role, count in sorted(role_resource_counts.items(), key=lambda x: -x[1]):
                print(f"    - {role}: {count} resources")

            processed['resource'] = processed.apply(
                lambda row: assign_resource_from_activity(
                    row['activity_name'], str(row['case_id']),
                    role_resource_counts=role_resource_counts
                ),
                axis=1
            )
        else:
            # FIXED: Same number per role regardless of workload
            print(f"  Deriving resources from activity patterns (fixed, {num_resources_per_role} per role)")
            processed['resource'] = processed.apply(
                lambda row: assign_resource_from_activity(row['activity_name'], str(row['case_id']), num_resources_per_role),
                axis=1
            )

    # Sort by case_id and timestamp
    processed = processed.sort_values(['case_id', 'timestamp']).reset_index(drop=True)

    # Create start and end timestamps (events are instantaneous)
    processed['start_timestamp'] = processed['timestamp']
    processed['end_timestamp'] = processed['timestamp']

    # Create agent mapping from final resources
    resource_to_agent, agent_to_resource = create_agent_mapping(processed['resource'].tolist())
    processed['agent'] = processed['resource'].map(resource_to_agent)

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
    print(f"  Final: {result['case_id'].nunique()} cases, {len(result)} events, {result['resource'].nunique()} resources")

    return result


def process_all_o2c_logs(
    logs_dir: str,
    output_suffix: str = 'processed',
    filter_activities: bool = False,
    min_activity_frequency_pct: float = 0.5,
    target_events: int = None,
    max_cases: int = None,
    preserve_resources: bool = True,
    filter_singletons: bool = False,
    max_resources_per_role: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Process all raw O2C logs in the given directory.

    Args:
        logs_dir: Path to the o2c logs directory
        output_suffix: Suffix for output files (e.g., 'processed' -> 'processed_742427502.csv')
        filter_activities: Whether to filter infrequent activities
        min_activity_frequency_pct: Minimum activity frequency threshold
        target_events: Target number of events per log (samples cases to achieve this)
        max_cases: Maximum number of cases per log (None = all cases)
        preserve_resources: If True, preserve original resource IDs
        filter_singletons: If True, remove cases with only 1 event
        max_resources_per_role: Maximum resources to keep per role

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
            result = process_raw_log(
                str(raw_file),
                str(output_file),
                filter_activities=filter_activities,
                min_activity_frequency_pct=min_activity_frequency_pct,
                target_events=target_events,
                max_cases=max_cases,
                preserve_resources=preserve_resources,
                filter_singletons=filter_singletons,
                max_resources_per_role=max_resources_per_role
            )
            results[log_id] = result
        except Exception as e:
            print(f"Error processing {log_id}: {e}")
            import traceback
            traceback.print_exc()
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
                        help='Process a single log folder by ID (e.g., 745347473)')
    parser.add_argument('--output-suffix', type=str,
                        default='processed',
                        help='Suffix for output files')
    parser.add_argument('--filter-activities', action='store_true',
                        help='Enable activity filtering (disabled by default to keep complexity)')
    parser.add_argument('--min-activity-freq', type=float, default=0.5,
                        help='Minimum activity frequency %% to keep (default: 0.5)')
    parser.add_argument('--target-events', type=int, default=None,
                        help='Target number of events (samples cases proportionally)')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Maximum number of cases to include (default: all)')
    parser.add_argument('--no-preserve-resources', action='store_true',
                        help='Use simplified resource derivation (old behavior)')
    parser.add_argument('--filter-singletons', action='store_true',
                        help='Remove cases with only 1 event (no process flow)')
    parser.add_argument('--max-resources-per-role', type=int, default=None,
                        help='Max resources to keep per role (keeps most active, default: all)')
    parser.add_argument('--num-resources-per-role', type=int, default=3,
                        help='Resources per role when using --no-preserve-resources (default: 3)')
    parser.add_argument('--total-resources', type=int, default=None,
                        help='Total resources to distribute proportionally by workload (overrides --num-resources-per-role)')
    parser.add_argument('--min-resources-per-role', type=int, default=1,
                        help='Minimum resources per role when using --total-resources (default: 1)')

    args = parser.parse_args()
    preserve_resources = not args.no_preserve_resources

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
        process_raw_log(
            str(raw_files[0]),
            str(output_file),
            filter_activities=args.filter_activities,
            min_activity_frequency_pct=args.min_activity_freq,
            target_events=args.target_events,
            max_cases=args.max_cases,
            preserve_resources=preserve_resources,
            filter_singletons=args.filter_singletons,
            max_resources_per_role=args.max_resources_per_role,
            num_resources_per_role=args.num_resources_per_role,
            total_resources=args.total_resources,
            min_resources_per_role=args.min_resources_per_role
        )
    else:
        print(f"Processing all logs in: {logs_dir}")
        results = process_all_o2c_logs(
            str(logs_dir),
            args.output_suffix,
            filter_activities=args.filter_activities,
            min_activity_frequency_pct=args.min_activity_freq,
            target_events=args.target_events,
            max_cases=args.max_cases,
            preserve_resources=preserve_resources,
            filter_singletons=args.filter_singletons,
            max_resources_per_role=args.max_resources_per_role
        )
        print(f"\nProcessed {len(results)} logs successfully")


if __name__ == '__main__':
    main()
