"""
Calendar Fault Injector for Quantitative Evaluation.

This module injects known faults into simulation models by reducing working hours.
The goal is to create M_fault from M_opt (the optimal/discovered model) and measure
how quickly different search strategies (LLM vs Grid) can recover the optimal configuration.

Fault Injection Strategy:
- Reduce working hours for a target resource's calendar (e.g., 8h → 4h)
- Ground truth: LLM should propose increasing hours back to original

Calendar Structure (from SIMOD/Prosimos):
{
  "id": "AML Investigator-000001_profile_calendar",
  "time_periods": [
    {"from": "MONDAY", "to": "MONDAY", "beginTime": "07:00:00", "endTime": "15:00:00"},
    ...
  ]
}
"""

import copy
import re
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


class CalendarFaultInjector:
    """Inject calendar faults into simulation models for controlled experiments."""

    def __init__(self, fault_hours_reduction: float = 4.0):
        """
        Initialize calendar fault injector.

        Args:
            fault_hours_reduction: Number of hours to reduce from original calendar
        """
        self.fault_hours_reduction = fault_hours_reduction

    def _extract_role_name(self, resource_id: str) -> str:
        """
        Extract role name from resource/calendar ID.

        Examples:
            "Loan Officer-000001_profile_calendar" -> "Loan Officer"
            "Clerk-000003_profile" -> "Clerk"
            "AML Investigator-000002" -> "AML Investigator"
        """
        name = resource_id.replace("_profile_calendar", "").replace("_profile", "").replace("_calendar", "")
        name = re.sub(r'-\d+$', '', name)
        return name.strip()

    def _parse_time(self, time_str: str) -> Tuple[int, int, int]:
        """Parse time string HH:MM:SS to (hour, minute, second)."""
        parts = time_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        sec_parts = parts[2].split('.') if len(parts) > 2 else ['0']
        second = int(sec_parts[0])
        return hour, minute, second

    def _format_time(self, hour: int, minute: int, second: int, preserve_microseconds: str = "") -> str:
        """Format time to HH:MM:SS string."""
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        if preserve_microseconds:
            time_str += '.' + preserve_microseconds
        return time_str

    def _calculate_hours_in_period(self, begin_time: str, end_time: str) -> float:
        """Calculate total hours in a time period."""
        begin_h, begin_m, _ = self._parse_time(begin_time)
        end_h, end_m, _ = self._parse_time(end_time)
        return (end_h - begin_h) + (end_m - begin_m) / 60

    def get_calendar_hours(self, model: Dict, calendar_id: str) -> Optional[Dict]:
        """
        Get hours information for a specific calendar.

        Returns:
            Dict with calendar info or None if not found
        """
        resource_calendars = model.get("prosimos_params", {}).get("resource_calendars", [])

        for calendar in resource_calendars:
            cal_id = calendar.get("id", calendar.get("name", ""))
            if calendar_id.lower() in cal_id.lower() or cal_id.lower() in calendar_id.lower():
                time_periods = calendar.get("time_periods", [])
                if not time_periods:
                    continue

                # Calculate total hours per day (approximate - use first period)
                period = time_periods[0]
                begin_time = period.get("beginTime", "09:00:00")
                end_time = period.get("endTime", "17:00:00")
                hours_per_day = self._calculate_hours_in_period(begin_time, end_time)

                return {
                    "calendar_id": cal_id,
                    "hours_per_day": hours_per_day,
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "num_periods": len(time_periods)
                }

        return None

    def get_all_calendars(self, model: Dict) -> List[Dict]:
        """Get all resource calendars with their hours information."""
        resource_calendars = model.get("prosimos_params", {}).get("resource_calendars", [])
        results = []

        for calendar in resource_calendars:
            cal_id = calendar.get("id", calendar.get("name", ""))
            time_periods = calendar.get("time_periods", [])

            if not time_periods:
                continue

            period = time_periods[0]
            begin_time = period.get("beginTime", "09:00:00")
            end_time = period.get("endTime", "17:00:00")
            hours_per_day = self._calculate_hours_in_period(begin_time, end_time)

            role = self._extract_role_name(cal_id)

            results.append({
                "calendar_id": cal_id,
                "role": role,
                "hours_per_day": hours_per_day,
                "begin_time": begin_time,
                "end_time": end_time,
                "num_periods": len(time_periods)
            })

        return results

    def find_best_calendar_for_fault(
        self,
        baseline_metrics: Dict,
        model: Dict,
        min_hours: float = 6.0
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Find the best calendar for fault injection.

        Selection criteria:
        1. Calendar must have >= min_hours working hours (to allow meaningful reduction)
        2. PRIMARY: Prefer higher utilization resources (bottleneck candidates)
        3. SECONDARY: Prefer longer working hours (more room for reduction)

        Args:
            baseline_metrics: Baseline metrics containing resource_utilization
            model: The simulation model with prosimos_params
            min_hours: Minimum hours required for fault injection

        Returns:
            Tuple of (calendar_id, original_hours, utilization)
        """
        calendars = self.get_all_calendars(model)
        utilization = baseline_metrics.get("resource_utilization", {})

        if not calendars:
            print("  [!] No calendars found in model")
            return None, None, None

        candidates = []
        for cal_info in calendars:
            hours = cal_info["hours_per_day"]

            if hours < min_hours:
                continue

            # Find utilization for this resource
            cal_id = cal_info["calendar_id"]
            role = cal_info["role"]

            # Try to match utilization by calendar ID or role
            util_val = 0.5  # Default
            for util_name, util in utilization.items():
                util_norm = util_name.replace("_profile", "").replace("_calendar", "").lower()
                cal_norm = cal_id.replace("_profile", "").replace("_calendar", "").lower()
                role_norm = role.lower()

                if util_norm in cal_norm or cal_norm in util_norm or role_norm in util_norm:
                    util_val = float(util)
                    break

            # Score: prioritize utilization, then hours
            score = util_val * 100 + hours

            candidates.append({
                "calendar_id": cal_id,
                "role": role,
                "hours": hours,
                "utilization": util_val,
                "score": score
            })

        if not candidates:
            print(f"  [!] No calendars with >= {min_hours}h found")
            return None, None, None

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        print(f"  [OK] Selected calendar for fault injection:")
        print(f"       {best['calendar_id']} ({best['hours']:.1f}h/day, util={best['utilization']:.2%})")
        if len(candidates) > 1:
            other_cands = [(c['calendar_id'], round(c['hours'], 1)) for c in candidates[1:3]]
            print(f"       Other candidates: {other_cands}")

        return best["calendar_id"], best["hours"], best["utilization"]

    def inject_fault(
        self,
        model: Dict,
        target_calendar: str,
        hours_reduction: Optional[float] = None
    ) -> Tuple[Dict, Dict]:
        """
        Inject a fault by reducing working hours for a calendar.

        Args:
            model: The original (optimal) simulation model
            target_calendar: Calendar ID to modify
            hours_reduction: Number of hours to reduce (default: self.fault_hours_reduction)

        Returns:
            Tuple of (faulty_model, fault_info)
        """
        if hours_reduction is None:
            hours_reduction = self.fault_hours_reduction

        faulty_model = copy.deepcopy(model)
        resource_calendars = faulty_model.get("prosimos_params", {}).get("resource_calendars", [])

        target_norm = target_calendar.lower().replace("_profile", "").replace("_calendar", "")

        modified_calendar = None
        original_hours = None
        new_hours = None
        original_end_time = None
        new_end_time = None

        for calendar in resource_calendars:
            cal_id = calendar.get("id", calendar.get("name", ""))
            cal_norm = cal_id.lower().replace("_profile", "").replace("_calendar", "")

            if target_norm not in cal_norm and cal_norm not in target_norm:
                continue

            time_periods = calendar.get("time_periods", [])
            if not time_periods:
                continue

            modified_calendar = cal_id

            for period in time_periods:
                end_time_str = period.get("endTime", "")
                begin_time_str = period.get("beginTime", "")

                if not end_time_str or not begin_time_str:
                    continue

                original_end_time = end_time_str
                original_hours = self._calculate_hours_in_period(begin_time_str, end_time_str)

                # Reduce end time by hours_reduction
                end_h, end_m, end_s = self._parse_time(end_time_str)

                new_end_h = end_h - int(hours_reduction)
                new_end_m = end_m - int((hours_reduction % 1) * 60)

                # Handle minute underflow
                if new_end_m < 0:
                    new_end_h -= 1
                    new_end_m += 60

                # Ensure we don't go below start time
                begin_h, begin_m, _ = self._parse_time(begin_time_str)
                min_end_h = begin_h + 1  # At least 1 hour of work

                if new_end_h < min_end_h:
                    new_end_h = min_end_h
                    new_end_m = 0

                microseconds = end_time_str.split('.')[-1] if '.' in end_time_str else ""
                new_end_time_str = self._format_time(new_end_h, new_end_m, end_s, microseconds)

                period["endTime"] = new_end_time_str
                new_end_time = new_end_time_str
                new_hours = self._calculate_hours_in_period(begin_time_str, new_end_time_str)

            break  # Only modify first matching calendar

        if modified_calendar is None:
            raise ValueError(f"Calendar '{target_calendar}' not found in model")

        fault_info = {
            "target_calendar": modified_calendar,
            "original_hours": original_hours,
            "fault_hours": new_hours,
            "hours_reduced": original_hours - new_hours if original_hours and new_hours else hours_reduction,
            "original_end_time": original_end_time,
            "new_end_time": new_end_time,
            "expected_fix_hours": original_hours - new_hours if original_hours and new_hours else hours_reduction
        }

        print(f"    Reduced {modified_calendar} from {original_hours:.1f}h to {new_hours:.1f}h/day")
        print(f"    (endTime: {original_end_time} → {new_end_time})")

        return faulty_model, fault_info

    def restore_hours(
        self,
        model: Dict,
        target_calendar: str,
        hours_to_add: float
    ) -> Dict:
        """
        Restore hours to a calendar by extending end time.

        Args:
            model: Model with reduced hours
            target_calendar: Calendar ID to modify
            hours_to_add: Number of hours to add back

        Returns:
            Modified model with restored hours
        """
        restored_model = copy.deepcopy(model)
        resource_calendars = restored_model.get("prosimos_params", {}).get("resource_calendars", [])

        target_norm = target_calendar.lower().replace("_profile", "").replace("_calendar", "")

        for calendar in resource_calendars:
            cal_id = calendar.get("id", calendar.get("name", ""))
            cal_norm = cal_id.lower().replace("_profile", "").replace("_calendar", "")

            if target_norm not in cal_norm and cal_norm not in target_norm:
                continue

            time_periods = calendar.get("time_periods", [])

            for period in time_periods:
                end_time_str = period.get("endTime", "")
                if not end_time_str:
                    continue

                end_h, end_m, end_s = self._parse_time(end_time_str)

                new_end_h = end_h + int(hours_to_add)
                new_end_m = end_m + int((hours_to_add % 1) * 60)

                # Handle minute overflow
                if new_end_m >= 60:
                    new_end_h += 1
                    new_end_m -= 60

                # Cap at 23:59:59
                if new_end_h >= 24:
                    new_end_h = 23
                    new_end_m = 59
                    end_s = 59

                microseconds = end_time_str.split('.')[-1] if '.' in end_time_str else ""
                new_end_time_str = self._format_time(new_end_h, new_end_m, end_s, microseconds)

                period["endTime"] = new_end_time_str

            break

        return restored_model
