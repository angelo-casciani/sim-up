"""Column detection and role label utilities.

These functions were consolidated from duplications across:
- simod_discoverer.py
- agent_simulator_runner.py
- prosimos_runner.py
- improvement_validation_pipeline.py
"""

import re
from typing import Dict, List, Optional, Tuple, Union


def find_column(
    columns: Union[List[str], set],
    patterns: List[str],
    fallback: Optional[str] = None
) -> Optional[str]:
    """Find column matching any pattern (case-insensitive), with exact match priority.

    Args:
        columns: List or set of column names to search
        patterns: List of patterns to match against (in priority order)
        fallback: Default value if no match found

    Returns:
        Matched column name or fallback value
    """
    columns = list(columns) if isinstance(columns, set) else columns

    # First try exact matches (case-insensitive)
    for col in columns:
        if col.lower() in [p.lower() for p in patterns]:
            return col

    # Then try substring matches
    for col in columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern.lower() in col_lower:
                return col

    return fallback


def normalize_role_label(label: Optional[str]) -> str:
    """Normalize role labels by removing suffixes and extracting source from arrows.

    Handles:
    - Removing '_profile' suffix
    - Extracting source from 'source → destination' format

    Args:
        label: Role label string (may contain arrows or suffixes)

    Returns:
        Normalized label string
    """
    role = str(label or '').strip()

    if not role:
        return ""

    # Handle arrow separators - extract the left side (source)
    for separator in ['→', '->']:
        if separator in role:
            role = role.split(separator)[0].strip()
            break

    # Remove _profile suffix
    if role.lower().endswith('_profile'):
        role = role[:-len('_profile')].strip()

    return role


def parse_shift_responsibility_target(target: Optional[str]) -> Tuple[str, str]:
    """Parse shift-responsibility target into (source, destination) labels.

    Args:
        target: Target string, possibly containing arrow separator
                e.g., "RoleA → RoleB" or just "RoleA"

    Returns:
        Tuple of (source_role, destination_role)
        If no arrow present, returns (normalized_target, normalized_target)
    """
    raw = str(target or '').strip()

    if not raw:
        return "", ""

    for separator in ['→', '->']:
        if separator in raw:
            src, dst = raw.split(separator, 1)
            return normalize_role_label(src), normalize_role_label(dst)

    normalized = normalize_role_label(raw)
    return normalized, normalized


def resolve_shift_fraction(value) -> float:
    """Resolve shift fraction from hypothesis value.

    Mapping:
    - Float in (0, 1): used directly as fraction (clamped to [0.1, 0.9])
    - Integer >= 1: mapped to increasing scale (1→0.3, 2→0.5, 3→0.7, >=4→0.9)
    - None / zero / invalid: default 0.5

    Args:
        value: Numeric value representing shift amount

    Returns:
        Shift fraction between 0.1 and 0.9
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5

    # Already a fraction
    if 0 < numeric < 1:
        return max(0.1, min(0.9, numeric))

    # Integer severity level
    if numeric >= 1:
        n = int(numeric)
        return min(0.9, 0.1 + n * 0.2)  # 1→0.3, 2→0.5, 3→0.7, >=4→0.9

    return 0.5


def extract_resource_role(resource_id: str) -> str:
    """Extract the role/type from a resource identifier.

    Handles various naming patterns found in event logs:
    - "Clerk-000001" → "Clerk"
    - "Loan Officer-000002" → "Loan Officer"
    - "ID4618" → "ID4618" (no suffix, keep as-is)
    - "sid-C1BCD679-E133-4F46-9956-013E1F9E0BD1" → "sid" (UUID suffix)
    - "qbp_e79fc9c0-b70d-4c52-acb5-7a2416e6086c" → "qbp" (UUID suffix)
    - "10913.0" → "Resource" (numeric IDs get generic role)
    - "NOT_SET" → "NOT_SET" (special values kept as-is)

    Args:
        resource_id: Resource identifier string

    Returns:
        Extracted role/type name
    """
    if not resource_id:
        return ""

    res_id = str(resource_id).strip()

    # Handle numeric-only IDs (e.g., "10913.0", "8997")
    # These are individual workers, not typed roles
    try:
        float(res_id)
        return "Resource"  # Generic role for numeric IDs
    except ValueError:
        pass

    # Pattern 1: "RoleName-NNNNNN" format (e.g., "Clerk-000001", "Loan Officer-000002")
    # Match: word(s) followed by hyphen and digits
    match = re.match(r'^(.+?)-\d+$', res_id)
    if match:
        return match.group(1).strip()

    # Pattern 2: UUID suffix patterns (sid-UUID, qbp_UUID)
    # Match: prefix followed by separator and UUID-like string
    uuid_pattern = r'^(.+?)[-_][a-fA-F0-9]{8}[-_]?[a-fA-F0-9]{4}[-_]?[a-fA-F0-9]{4}[-_]?[a-fA-F0-9]{4}[-_]?[a-fA-F0-9]{12}$'
    match = re.match(uuid_pattern, res_id)
    if match:
        return match.group(1).strip()

    # Pattern 3: "prefix_UUID" without full UUID format (e.g., "qbp_e79fc9c0...")
    # Check for underscore followed by hex-like string
    if '_' in res_id:
        parts = res_id.split('_', 1)
        if len(parts) == 2 and len(parts[1]) > 8:
            # Check if suffix looks like a UUID/hash
            if re.match(r'^[a-fA-F0-9-]+$', parts[1]):
                return parts[0].strip()

    # Pattern 4: "IDNNNN" format (e.g., "ID4618") - keep as-is, these are individual IDs
    if re.match(r'^ID\d+$', res_id, re.IGNORECASE):
        return res_id  # Each ID is its own "role"

    # Default: return as-is (no recognizable suffix pattern)
    return res_id


def group_resources_by_role(resource_profiles: List[Dict]) -> Dict[str, List[Dict]]:
    """Group resource profiles by their extracted role.

    Args:
        resource_profiles: List of resource profile dicts with 'id' or 'name' keys

    Returns:
        Dict mapping role names to list of matching resource profiles
    """
    groups: Dict[str, List[Dict]] = {}

    for profile in resource_profiles:
        res_id = profile.get('id', profile.get('name', ''))
        role = extract_resource_role(res_id)

        if role not in groups:
            groups[role] = []
        groups[role].append(profile)

    return groups


def find_resources_matching_role(
    resource_profiles: List[Dict],
    target_role: str
) -> Tuple[List[Dict], str]:
    """Find all resource profiles that match a target role.

    Uses fuzzy matching to handle variations in naming:
    - Case-insensitive comparison
    - Substring matching (target in role or role in target)
    - Role extraction for suffix-based IDs

    Args:
        resource_profiles: List of resource profile dicts
        target_role: Target role name to match

    Returns:
        Tuple of (matching_profiles, matched_role_name)
        matched_role_name is the canonical role name found
    """
    if not target_role or not resource_profiles:
        return [], ""

    target_lower = target_role.lower().strip()

    # Group by role first
    role_groups = group_resources_by_role(resource_profiles)

    # Try exact match first (case-insensitive)
    for role, profiles in role_groups.items():
        if role.lower() == target_lower:
            return profiles, role

    # Try substring match (target in role or role in target)
    for role, profiles in role_groups.items():
        role_lower = role.lower()
        if target_lower in role_lower or role_lower in target_lower:
            return profiles, role

    # Try matching against individual resource IDs as fallback
    matching = []
    for profile in resource_profiles:
        res_id = str(profile.get('id', profile.get('name', ''))).lower()
        if target_lower in res_id or res_id in target_lower:
            matching.append(profile)

    if matching:
        # Return with the target as the role name
        return matching, target_role

    return [], ""


def calculate_target_distribution(
    current_profiles: List[Dict],
    target_total: int,
    change_type: str
) -> List[Tuple[Dict, int]]:
    """Calculate how to distribute a target resource count across profiles.

    For 'increase' changes: distributes additional resources, potentially
    creating new profile slots based on existing patterns.

    For 'decrease' changes: reduces resources proportionally, respecting
    minimum of 1 per active profile.

    Args:
        current_profiles: List of current resource profiles for this role
        target_total: Target total number of resources for this role
        change_type: Type of change ('add', 'increase', 'remove', 'decrease', 'set')

    Returns:
        List of (profile, new_amount) tuples
    """
    if not current_profiles:
        return []

    # Calculate current total
    current_total = sum(p.get('amount', 1) for p in current_profiles)

    # Handle 'set' as absolute value
    if 'set' in change_type.lower():
        target = target_total
    elif 'add' in change_type.lower() or 'increase' in change_type.lower():
        target = current_total + target_total
    elif 'remove' in change_type.lower() or 'decrease' in change_type.lower():
        target = max(len(current_profiles), current_total - target_total)
    else:
        # Default: treat as absolute target
        target = target_total

    # Ensure minimum of 1 per profile
    target = max(len(current_profiles), target)

    # Distribute evenly with remainder going to first profiles
    n_profiles = len(current_profiles)
    base_amount = target // n_profiles
    remainder = target % n_profiles

    distribution = []
    for i, profile in enumerate(current_profiles):
        new_amount = base_amount + (1 if i < remainder else 0)
        distribution.append((profile, max(1, new_amount)))

    return distribution
