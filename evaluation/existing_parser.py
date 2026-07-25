from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .common import (
    expand_days_from_text,
    load_yaml_like_file,
    normalize_shift_name,
    parse_day_numbers,
    parse_inline_shift_pairs,
    parse_nurse_numbers,
    parse_shift_list,
)
from .models import CanonicalInstance, Rule


def parse_existing_instance(path: Path) -> CanonicalInstance:
    data = load_yaml_like_file(path)
    basic = data["basic_setting"]
    instance = CanonicalInstance(
        instance_id=data["instance_id"],
        problem_type=data["problem_type"],
        source_type="existing",
        number_of_nurses=int(basic["number_of_nurses"]),
        number_of_days=int(basic["number_of_days"]),
        shift_types=[normalize_shift_name(value) for value in basic.get("shift_types", [])],
        shift_durations={str(key): int(value) for key, value in basic.get("shift_duration_minutes", {}).items()},
        first_day_of_scheduling_period=basic.get("first_day_of_scheduling_period", "Monday"),
    )

    coverage_penalties = {"under": 0, "over": 0}

    for item in data.get("hard_constraints", []):
        text = _constraint_to_text(item)
        parsed_rules = _parse_existing_hard_constraint(text)
        if parsed_rules:
            instance.hard_rules.extend(parsed_rules)
        else:
            instance.unsupported_rules.append({"kind": "hard", "text": text})

    for item in data.get("soft_constraints", []):
        text = _constraint_to_text(item)
        parsed_rules = _parse_existing_soft_constraint(text, coverage_penalties)
        if parsed_rules is None:
            instance.unsupported_rules.append({"kind": "soft", "text": text})
        else:
            instance.soft_rules.extend(parsed_rules)

    for rule in instance.soft_rules:
        if rule.type == "coverage_target":
            if "under_penalty" not in rule.params:
                rule.params["under_penalty"] = coverage_penalties["under"]
            if "over_penalty" not in rule.params:
                rule.params["over_penalty"] = coverage_penalties["over"]

    if not instance.shift_durations:
        instance.warnings.append("No shift_duration_minutes found in basic_setting.")

    return instance


def _parse_existing_hard_constraint(text: str) -> Optional[List[Rule]]:
    lowered = text.lower()
    if (
        ("exactly one" in lowered and ("per day" in lowered or "each day" in lowered))
        or ("assigned either one" in lowered and "each day" in lowered)
    ):
        return [Rule(type="assignment_completeness", source_text=text, params={}, is_hard=True)]

    if "cannot be assigned more than one shift on the same day" in lowered:
        return [Rule(type="single_assignment_per_day", source_text=text, params={}, is_hard=True)]

    match = re.match(
        r"A\s+([A-Za-z0-9_]+)\s+shift cannot be followed(?: on the next day)? by\s+(.+)\.",
        text,
    )
    if match:
        return [
            Rule(
                type="forbidden_succession",
                source_text=text,
                params={"from_shift": match.group(1), "to_shifts": parse_shift_list(match.group(2))},
                is_hard=True,
            )
        ]

    match = re.match(
        r"A\s+([A-Za-z0-9_]+)\s+shift cannot be followed by any of (?:these|the following)\s+shifts on the next day:\s+(.+)\.?",
        text,
    )
    if match:
        return [
            Rule(
                type="forbidden_succession",
                source_text=text,
                params={"from_shift": match.group(1), "to_shifts": parse_shift_list(match.group(2))},
                is_hard=True,
            )
        ]

    if "fixed days off" in text or "must have" in text and " off." in text:
        nurses = parse_nurse_numbers(text)
        days = parse_day_numbers(text)
        if nurses and days:
            return [
                Rule(
                    type="fixed_days_off",
                    source_text=text,
                    params={"nurse": nurse, "days": days},
                    is_hard=True,
                )
                for nurse in nurses
            ]

    return None


def _parse_existing_soft_constraint(text: str, coverage_penalties: Dict[str, int]) -> Optional[List[Rule]]:
    lowered = text.lower()
    if lowered.startswith("minimize "):
        return []

    penalty_match = re.search(
        r"understaffing incurs a penalty of\s+(\d+).+overstaffing incurs a penalty of\s+(\d+)",
        lowered,
    )
    if penalty_match:
        coverage_penalties["under"] = int(penalty_match.group(1))
        coverage_penalties["over"] = int(penalty_match.group(2))
        return []

    coverage_target = re.match(r"On Day\s+(\d+), the target coverage is\s+(.+)\.", text)
    if coverage_target:
        return [
            Rule(
                type="coverage_target",
                source_text=text,
                params={
                    "day": int(coverage_target.group(1)),
                    "requirements": parse_inline_shift_pairs(coverage_target.group(2)),
                },
                is_hard=False,
            )
        ]

    coverage_line = re.match(r"Day\s+(\d+):\s+([A-Za-z0-9_]+)\s+requires\s+(\d+)\s+nurses?\.", text)
    if coverage_line:
        return [
            Rule(
                type="coverage_target",
                source_text=text,
                params={
                    "day": int(coverage_line.group(1)),
                    "requirements": {coverage_line.group(2): int(coverage_line.group(3))},
                },
                is_hard=False,
            )
        ]

    group_contract = re.match(
        r"Nurses?\s+([0-9,\sand]+):\s+minutes\s+(\d+)-(\d+),\s+max consecutive\s+(\d+),\s+min consecutive\s+(\d+),\s+min off\s+(\d+),\s+max weekends\s+(\d+)\.",
        text,
    )
    if group_contract:
        nurses = [int(value) for value in re.findall(r"\d+", group_contract.group(1))]
        rules: List[Rule] = []
        for nurse in nurses:
            rules.extend(_build_contract_rules(
                nurse=nurse,
                min_minutes=int(group_contract.group(2)),
                max_minutes=int(group_contract.group(3)),
                max_consecutive_work=int(group_contract.group(4)),
                min_consecutive_work=int(group_contract.group(5)),
                min_consecutive_off=int(group_contract.group(6)),
                max_weekends=int(group_contract.group(7)),
                source_text=text,
            ))
        return rules

    ranged_group_contract = re.match(
        r"Nurses?\s+(\d+)[–-](\d+)\s+should work between\s+(\d+)\s+and\s+(\d+)\s+minutes,\s+work no more than\s+(\d+)\s+consecutive days,\s+work at least\s+(\d+)\s+consecutive days,\s+have at least\s+(\d+)\s+consecutive days off,\s+and work at most\s+(\d+)\s+weekends\.",
        text,
    )
    if ranged_group_contract:
        start = int(ranged_group_contract.group(1))
        end = int(ranged_group_contract.group(2))
        rules: List[Rule] = []
        for nurse in range(start, end + 1):
            rules.extend(_build_contract_rules(
                nurse=nurse,
                min_minutes=int(ranged_group_contract.group(3)),
                max_minutes=int(ranged_group_contract.group(4)),
                max_consecutive_work=int(ranged_group_contract.group(5)),
                min_consecutive_work=int(ranged_group_contract.group(6)),
                min_consecutive_off=int(ranged_group_contract.group(7)),
                max_weekends=int(ranged_group_contract.group(8)),
                source_text=text,
            ))
        return rules

    nurse_contract = re.match(
        r"Nurse\s+(\d+)\s+should work between\s+(\d+)\s+and\s+(\d+)\s+minutes, should work no more than\s+(\d+)\s+consecutive days, each sequence of consecutive working days should contain at least\s+(\d+)\s+days, each sequence of consecutive days off should contain at least\s+(\d+)\s+days, and the nurse should work no more than\s+(\d+)\s+weekends\.(.*)",
        text,
    )
    if nurse_contract:
        nurse = int(nurse_contract.group(1))
        rules = _build_contract_rules(
            nurse=nurse,
            min_minutes=int(nurse_contract.group(2)),
            max_minutes=int(nurse_contract.group(3)),
            max_consecutive_work=int(nurse_contract.group(4)),
            min_consecutive_work=int(nurse_contract.group(5)),
            min_consecutive_off=int(nurse_contract.group(6)),
            max_weekends=int(nurse_contract.group(7)),
            source_text=text,
        )
        tail = nurse_contract.group(8).strip()
        if tail:
            eligibility_rules = _parse_eligibility_tail(nurse, tail, text)
            if eligibility_rules:
                rules.extend(eligibility_rules)
        return rules

    eligibility = re.match(r"Nurse\s+(\d+)\s+may work only(?::)?\s+(.+)\.", text)
    if eligibility:
        nurse = int(eligibility.group(1))
        shifts = parse_shift_list(eligibility.group(2))
        return [Rule(type="allowed_shifts", source_text=text, params={"nurse": nurse, "allowed_shifts": shifts}, is_hard=False)]

    semicolon_eligibility = re.match(
        r"Nurse\s+(\d+)\s+may work only\s+(.+?);\s+the maximum assignments for capped shifts are\s+(.+)\.",
        text,
    )
    if semicolon_eligibility:
        nurse = int(semicolon_eligibility.group(1))
        rules = [Rule(type="allowed_shifts", source_text=text, params={"nurse": nurse, "allowed_shifts": parse_shift_list(semicolon_eligibility.group(2))}, is_hard=False)]
        for shift, limit in re.findall(r"([A-Za-z0-9_]+)\s+\((\d+)\)", semicolon_eligibility.group(3)):
            rules.append(Rule(type="shift_max_count", source_text=text, params={"nurse": nurse, "shift": shift, "limit": int(limit)}, is_hard=False))
        return rules

    if text.startswith("Shift-request weights indicate preference importance"):
        return []

    negative_preference = re.match(
        r"Nurse\s+(\d+)\s+prefers not to work(?: a| an)?\s+([A-Za-z0-9_ ]+?)\s+shift\s+on\s+Day\s+(\d+),\s+with a violation penalty of\s+(\d+)\.",
        text,
    )
    if negative_preference:
        return [_build_shift_preference_rule(negative_preference, text, negated=True)]

    positive_preference = re.match(
        r"Nurse\s+(\d+)\s+prefers(?: to work)?(?: a| an)?\s+([A-Za-z0-9_ ]+?)\s+shift\s+on\s+Day\s+(\d+),\s+with a violation penalty of\s+(\d+)\.",
        text,
    )
    if positive_preference:
        return [_build_shift_preference_rule(positive_preference, text, negated=False)]

    generic_positive_weight = re.match(
        r"Nurse\s+(\d+)\s+prefers(?: to work)?\s+([A-Za-z0-9_]+)\s+on\s+Day\s+(\d+)\s+\(weight\s+(\d+)\)\.",
        text,
    )
    if generic_positive_weight:
        return [_build_simple_preference_rule(generic_positive_weight, text, negated=False)]

    generic_negative_weight = re.match(
        r"Nurse\s+(\d+)\s+prefers not to work\s+([A-Za-z0-9_]+)\s+on\s+Day\s+(\d+)\s+\(weight\s+(\d+)\)\.",
        text,
    )
    if generic_negative_weight:
        return [_build_simple_preference_rule(generic_negative_weight, text, negated=True)]

    range_preference = re.match(
        r"Nurse\s+(\d+)\s+prefers(?: not to work)?\s+([A-Za-z0-9_ ]+?)\s+shifts?\s+on\s+Days?\s+(\d+)[–-](\d+)\s+\((?:weight|penalty)\s+(\d+)(?: per violation)?\)\.?",
        text,
    )
    if range_preference:
        nurse = int(range_preference.group(1))
        shift_pattern = _normalize_shift_pattern(range_preference.group(2))
        start = int(range_preference.group(3))
        end = int(range_preference.group(4))
        penalty = int(range_preference.group(5))
        negated = "prefers not to work" in text
        return [
            Rule(
                type="shift_preference",
                source_text=text,
                params={
                    "nurse": nurse,
                    "days": list(range(start, end + 1)),
                    "shift_pattern": shift_pattern,
                    "preferred": not negated,
                    "penalty": penalty,
                },
                is_hard=False,
            )
        ]

    return None


def _build_contract_rules(
    nurse: int,
    min_minutes: int,
    max_minutes: int,
    max_consecutive_work: int,
    min_consecutive_work: int,
    min_consecutive_off: int,
    max_weekends: int,
    source_text: str,
) -> List[Rule]:
    return [
        Rule(type="minutes_range", source_text=source_text, params={"nurse": nurse, "min_minutes": min_minutes, "max_minutes": max_minutes}, is_hard=False),
        Rule(type="max_consecutive_work_days", source_text=source_text, params={"nurse": nurse, "limit": max_consecutive_work}, is_hard=False),
        Rule(type="min_consecutive_work_days", source_text=source_text, params={"nurse": nurse, "limit": min_consecutive_work}, is_hard=False),
        Rule(type="min_consecutive_off_days", source_text=source_text, params={"nurse": nurse, "limit": min_consecutive_off}, is_hard=False),
        Rule(type="max_weekends", source_text=source_text, params={"nurse": nurse, "limit": max_weekends}, is_hard=False),
    ]


def _parse_eligibility_tail(nurse: int, tail: str, source_text: str) -> List[Rule]:
    rules: List[Rule] = []
    eligibility_match = re.search(r"The nurse may work only\s+(.+?)(?:, with|$)", tail)
    if eligibility_match:
        shifts = parse_shift_list(eligibility_match.group(1))
        rules.append(Rule(type="allowed_shifts", source_text=source_text, params={"nurse": nurse, "allowed_shifts": shifts}, is_hard=False))

    for count, shift in re.findall(r"no more than\s+(\d+)\s+([A-Za-z0-9_]+)\s+shifts", tail):
        rules.append(
            Rule(
                type="shift_max_count",
                source_text=source_text,
                params={"nurse": nurse, "shift": shift, "limit": int(count)},
                is_hard=False,
            )
        )
    return rules


def _build_shift_preference_rule(match: re.Match[str], text: str, negated: bool) -> Rule:
    return Rule(
        type="shift_preference",
        source_text=text,
        params={
            "nurse": int(match.group(1)),
            "days": [int(match.group(3))],
            "shift_pattern": _normalize_shift_pattern(match.group(2)),
            "preferred": not negated,
            "penalty": int(match.group(4)),
        },
        is_hard=False,
    )


def _build_simple_preference_rule(match: re.Match[str], text: str, negated: bool) -> Rule:
    return Rule(
        type="shift_preference",
        source_text=text,
        params={
            "nurse": int(match.group(1)),
            "days": [int(match.group(3))],
            "shift_pattern": _normalize_shift_pattern(match.group(2)),
            "preferred": not negated,
            "penalty": int(match.group(4)),
        },
        is_hard=False,
    )


def _normalize_shift_pattern(text: str) -> str:
    cleaned = text.strip().replace(" shift", "").replace(" shifts", "")
    if cleaned in {"morning", "day", "evening", "night", "early", "late"}:
        return f"{cleaned}*"
    return cleaned


def _constraint_to_text(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and len(item) == 1:
        key, value = next(iter(item.items()))
        return f"{key}: {value}"
    return str(item)
