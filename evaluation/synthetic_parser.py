from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from .common import load_yaml_like_file, normalize_shift_name, parse_nurse_numbers
from .models import CanonicalInstance, Rule


def parse_synthetic_instance(path: Path) -> CanonicalInstance:
    data = load_yaml_like_file(path)
    basic = data["basic_setting"]
    instance = CanonicalInstance(
        instance_id=data["instance_id"],
        problem_type=data["problem_type"],
        source_type="synthetic",
        number_of_nurses=int(basic["number_of_nurses"]),
        number_of_days=int(basic["number_of_days"]),
        shift_types=[normalize_shift_name(value) for value in basic.get("shift_types", [])],
        shift_durations={str(key): int(value) for key, value in basic.get("shift_duration_minutes", {}).items()},
        first_day_of_scheduling_period=basic.get("first_day_of_scheduling_period", "Monday"),
        meta={"all_preferences": data.get("all_preferences", [])},
    )

    for text in data.get("hard_constraints", []):
        rules = _parse_synthetic_hard_constraint(text)
        if rules:
            instance.hard_rules.extend(rules)
        else:
            instance.unsupported_rules.append({"kind": "hard", "text": text})

    return instance


def _parse_synthetic_hard_constraint(text: str) -> Optional[List[Rule]]:
    lowered = text.lower()
    if "exactly one shift or one day off per day" in lowered or "assigned 1 shift per day" in lowered:
        return [Rule(type="assignment_completeness", source_text=text, params={}, is_hard=True)]

    if "cannot work a night shift followed" in lowered and "day shift" in lowered:
        return [
            Rule(
                type="forbidden_succession",
                source_text=text,
                params={"from_shift": "night", "to_shifts": ["day"]},
                is_hard=True,
            )
        ]

    exact_day = re.match(r"Exactly\s+(\d+)\s+nurses?\s+must be assigned to the\s+([A-Za-z0-9_]+)\s+shift on each day\.", text)
    if exact_day:
        return [
            Rule(
                type="daily_shift_coverage_all_days",
                source_text=text,
                params={"shift": exact_day.group(2), "required": int(exact_day.group(1))},
                is_hard=True,
            )
        ]

    exact_day_alt = re.match(r"Exactly\s+(\d+)\s+nurse must be assigned to the\s+([A-Za-z0-9_]+)\s+shift on each day\.", text)
    if exact_day_alt:
        return [
            Rule(
                type="daily_shift_coverage_all_days",
                source_text=text,
                params={"shift": exact_day_alt.group(2), "required": int(exact_day_alt.group(1))},
                is_hard=True,
            )
        ]

    max_total = re.match(r"Nurses?\s+([0-9,\sand]+)\s+can work at most\s+(\d+)\s+shifts during the scheduling period\.", text)
    if max_total:
        nurses = parse_nurse_numbers(text)
        limit = int(max_total.group(2))
        return [Rule(type="max_total_work_shifts", source_text=text, params={"nurse": nurse, "limit": limit}, is_hard=True) for nurse in nurses]

    max_nights = re.match(r"(?:Each nurse|A nurse|The nurse)\s+can work at most\s+(\d+)\s+night shifts during the scheduling period\.", text)
    if max_nights:
        return [Rule(type="max_shift_type_count_all", source_text=text, params={"shift_pattern": "night*", "limit": int(max_nights.group(1))}, is_hard=True)]

    min_nights = re.match(r"Each nurse must work at least\s+(\d+)\s+night shifts during the scheduling period\.", text)
    if min_nights:
        return [Rule(type="min_shift_type_count_all", source_text=text, params={"shift_pattern": "night*", "limit": int(min_nights.group(1))}, is_hard=True)]

    max_consecutive_days = re.match(r"A nurse cannot work more than\s+(\d+)\s+consecutive(?: working)? days\.", text)
    if max_consecutive_days:
        return [Rule(type="max_consecutive_work_days_all", source_text=text, params={"limit": int(max_consecutive_days.group(1))}, is_hard=True)]

    min_consecutive_off = re.match(r"A nurse (?:must have|should receive) at least\s+(\d+)\s+consecutive days off.*", text)
    if min_consecutive_off:
        return [Rule(type="min_consecutive_off_days_all", source_text=text, params={"limit": int(min_consecutive_off.group(1))}, is_hard=True)]

    max_consecutive_nights = re.match(r"A nurse cannot work(?: more than)?\s+(\d+)\s+consecutive night shifts\.", text)
    if max_consecutive_nights:
        return [Rule(type="max_consecutive_shift_type_all", source_text=text, params={"shift_pattern": "night*", "limit": int(max_consecutive_nights.group(1))}, is_hard=True)]

    isolated_night = re.match(r"A nurse cannot work a single isolated night shift between non-night shifts\.", text)
    if isolated_night:
        return [Rule(type="no_isolated_shift_type_all", source_text=text, params={"shift_pattern": "night*"}, is_hard=True)]

    max_weekends = re.match(r"A nurse cannot work more than\s+(\d+)\s+consecutive weekends\.", text)
    if max_weekends:
        return [Rule(type="max_consecutive_worked_weekends_all", source_text=text, params={"limit": int(max_weekends.group(1))}, is_hard=True)]

    weekly_range = re.match(r"Nurses?\s+([0-9,\sand]+)\s+must work between\s+(\d+)\s+and\s+(\d+)\s+shifts per week\.", text)
    if weekly_range:
        nurses = parse_nurse_numbers(text)
        minimum = int(weekly_range.group(2))
        maximum = int(weekly_range.group(3))
        return [
            Rule(type="weekly_total_shift_range", source_text=text, params={"nurse": nurse, "min": minimum, "max": maximum}, is_hard=True)
            for nurse in nurses
        ]

    return None
