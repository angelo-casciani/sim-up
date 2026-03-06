"""Column detection and role label utilities.

These functions were consolidated from duplications across:
- simod_discoverer.py
- agent_simulator_runner.py
- prosimos_runner.py
- improvement_validation_pipeline.py
"""

from typing import List, Optional, Tuple, Union


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


def parse_role_redesign_target(target: Optional[str]) -> Tuple[str, str]:
    """Parse role-redesign target into (source, destination) labels.

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
