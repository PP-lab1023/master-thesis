from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .canonicalizer import canonicalize_description
from .common import iter_runs, shift_matches_pattern, weekend_pairs
from .models import CanonicalInstance, Rule
from .schedule_loader import load_schedule


def evaluate_description(description_path: Path, schedule_path: Path, source_type: str | None = None) -> Dict[str, Any]:
    instance = canonicalize_description(description_path, source_type=source_type)
    return evaluate_canonical_instance(instance, schedule_path)


def evaluate_canonical_instance(instance: CanonicalInstance, schedule_path: Path) -> Dict[str, Any]:
    schedule = load_schedule(schedule_path)
    context = _build_context(instance, schedule)

    hard_results = [_evaluate_rule(rule, instance, context) for rule in instance.hard_rules]
    soft_results = [_evaluate_rule(rule, instance, context) for rule in instance.soft_rules]

    hard_violations = sum(result["violation_count"] for result in hard_results)
    soft_penalty = sum(result["penalty"] for result in soft_results)
    return {
        "instance_id": instance.instance_id,
        "source_type": instance.source_type,
        "feasible": hard_violations == 0,
        "hard_violation_count": hard_violations,
        "soft_penalty_total": soft_penalty,
        "hard_results": hard_results,
        "soft_results": soft_results,
        "unsupported_rules": instance.unsupported_rules,
        "instance_warnings": instance.warnings,
        "schedule_warnings": schedule.raw_warnings,
    }


def _build_context(instance: CanonicalInstance, schedule: Any) -> Dict[str, Any]:
    assignments = schedule.assignments
    work_shifts = {shift for shift in instance.shift_types if shift.lower() != "off"}
    per_nurse_shift_counts: Dict[int, Counter[str]] = {}
    per_day_shift_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    per_nurse_minutes: Dict[int, int] = defaultdict(int)
    per_nurse_work_flags: Dict[int, List[bool]] = {}
    per_nurse_off_flags: Dict[int, List[bool]] = {}

    for nurse in range(1, instance.number_of_nurses + 1):
        counter: Counter[str] = Counter()
        work_flags: List[bool] = []
        off_flags: List[bool] = []
        for day in range(1, instance.number_of_days + 1):
            shift = assignments.get(nurse, {}).get(day)
            if shift is not None:
                counter[shift] += 1
                per_day_shift_counts[day][shift] += 1
                per_nurse_minutes[nurse] += instance.shift_durations.get(shift, 0)
            is_work = shift in work_shifts
            work_flags.append(is_work)
            off_flags.append(shift == "off")
        per_nurse_shift_counts[nurse] = counter
        per_nurse_work_flags[nurse] = work_flags
        per_nurse_off_flags[nurse] = off_flags

    weekends = weekend_pairs(instance.number_of_days, instance.first_day_of_scheduling_period)
    per_nurse_weekends_worked: Dict[int, int] = {}
    per_nurse_weekend_flags: Dict[int, List[bool]] = {}
    for nurse in range(1, instance.number_of_nurses + 1):
        flags: List[bool] = []
        for saturday, sunday in weekends:
            saturday_shift = assignments.get(nurse, {}).get(saturday)
            sunday_shift = assignments.get(nurse, {}).get(sunday)
            works = saturday_shift in work_shifts or sunday_shift in work_shifts
            flags.append(works)
        per_nurse_weekend_flags[nurse] = flags
        per_nurse_weekends_worked[nurse] = sum(flags)

    return {
        "assignments": assignments,
        "duplicates": schedule.duplicates,
        "work_shifts": work_shifts,
        "per_nurse_shift_counts": per_nurse_shift_counts,
        "per_day_shift_counts": per_day_shift_counts,
        "per_nurse_minutes": per_nurse_minutes,
        "per_nurse_work_flags": per_nurse_work_flags,
        "per_nurse_off_flags": per_nurse_off_flags,
        "per_nurse_weekends_worked": per_nurse_weekends_worked,
        "per_nurse_weekend_flags": per_nurse_weekend_flags,
    }


def _evaluate_rule(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    evaluator = _RULE_EVALUATORS.get(rule.type, _evaluate_unsupported_rule)
    return evaluator(rule, instance, context)


def _build_result(rule: Rule, violation_count: int, details: List[Dict[str, Any]] | None = None, penalty: int = 0) -> Dict[str, Any]:
    return {
        "type": rule.type,
        "is_hard": rule.is_hard,
        "source_text": rule.source_text,
        "violation_count": violation_count,
        "penalty": penalty,
        "satisfied": violation_count == 0,
        "details": details or [],
    }


def _evaluate_assignment_completeness(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    assignments = context["assignments"]
    valid_shifts = set(instance.shift_types)
    for nurse in range(1, instance.number_of_nurses + 1):
        for day in range(1, instance.number_of_days + 1):
            shift = assignments.get(nurse, {}).get(day)
            if shift is None:
                details.append({"nurse": nurse, "day": day, "issue": "missing_assignment"})
            elif shift not in valid_shifts:
                details.append({"nurse": nurse, "day": day, "issue": "invalid_shift", "shift": shift})
    return _build_result(rule, len(details), details)


def _evaluate_single_assignment(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    details = [
        {"nurse": nurse, "day": day, "first": first, "second": second}
        for nurse, day, first, second in context["duplicates"]
    ]
    return _build_result(rule, len(details), details)


def _evaluate_fixed_days_off(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    details: List[Dict[str, Any]] = []
    for day in rule.params["days"]:
        shift = context["assignments"].get(nurse, {}).get(day)
        if shift != "off":
            details.append({"nurse": nurse, "day": day, "assigned_shift": shift})
    return _build_result(rule, len(details), details)


def _evaluate_forbidden_succession(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    from_shift = rule.params["from_shift"]
    to_shifts = rule.params["to_shifts"]
    assignments = context["assignments"]
    for nurse in range(1, instance.number_of_nurses + 1):
        for day in range(1, instance.number_of_days):
            current = assignments.get(nurse, {}).get(day)
            nxt = assignments.get(nurse, {}).get(day + 1)
            if current == from_shift and nxt in to_shifts:
                details.append({"nurse": nurse, "day": day, "from_shift": current, "to_shift": nxt})
    return _build_result(rule, len(details), details)


def _evaluate_minutes_range(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    minutes = context["per_nurse_minutes"][nurse]
    minimum = rule.params["min_minutes"]
    maximum = rule.params["max_minutes"]
    details: List[Dict[str, Any]] = []
    penalty = 0
    if minutes < minimum:
        details.append({"nurse": nurse, "minutes": minutes, "expected_min": minimum})
        penalty += minimum - minutes
    if minutes > maximum:
        details.append({"nurse": nurse, "minutes": minutes, "expected_max": maximum})
        penalty += minutes - maximum
    return _build_result(rule, len(details), details, penalty)


def _evaluate_max_consecutive_work(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params.get("nurse")
    limit = rule.params["limit"]
    nurses = [nurse] if nurse else list(range(1, instance.number_of_nurses + 1))
    details: List[Dict[str, Any]] = []
    for current_nurse in nurses:
        for run in iter_runs(context["per_nurse_work_flags"][current_nurse]):
            if run > limit:
                details.append({"nurse": current_nurse, "run_length": run, "limit": limit})
    return _build_result(rule, len(details), details, sum(max(0, detail["run_length"] - limit) for detail in details))


def _evaluate_min_consecutive_work(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    limit = rule.params["limit"]
    details: List[Dict[str, Any]] = []
    for run in iter_runs(context["per_nurse_work_flags"][nurse]):
        if 0 < run < limit:
            details.append({"nurse": nurse, "run_length": run, "limit": limit})
    return _build_result(rule, len(details), details, sum(limit - detail["run_length"] for detail in details))


def _evaluate_min_consecutive_off(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params.get("nurse")
    limit = rule.params["limit"]
    nurses = [nurse] if nurse else list(range(1, instance.number_of_nurses + 1))
    details: List[Dict[str, Any]] = []
    for current_nurse in nurses:
        for run in iter_runs(context["per_nurse_off_flags"][current_nurse]):
            if 0 < run < limit:
                details.append({"nurse": current_nurse, "run_length": run, "limit": limit})
    return _build_result(rule, len(details), details, sum(limit - detail["run_length"] for detail in details))


def _evaluate_max_weekends(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    limit = rule.params["limit"]
    worked = context["per_nurse_weekends_worked"][nurse]
    if worked <= limit:
        return _build_result(rule, 0, [])
    detail = {"nurse": nurse, "worked_weekends": worked, "limit": limit}
    return _build_result(rule, 1, [detail], worked - limit)


def _evaluate_allowed_shifts(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    allowed = set(rule.params["allowed_shifts"]) | {"off"}
    details: List[Dict[str, Any]] = []
    for day, shift in context["assignments"].get(nurse, {}).items():
        if shift not in allowed:
            details.append({"nurse": nurse, "day": day, "shift": shift})
    return _build_result(rule, len(details), details, len(details))


def _evaluate_shift_max_count(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    shift = rule.params["shift"]
    limit = rule.params["limit"]
    count = context["per_nurse_shift_counts"][nurse][shift]
    if count <= limit:
        return _build_result(rule, 0, [])
    detail = {"nurse": nurse, "shift": shift, "count": count, "limit": limit}
    return _build_result(rule, 1, [detail], count - limit)


def _evaluate_shift_preference(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    days = rule.params["days"]
    pattern = rule.params["shift_pattern"]
    preferred = rule.params["preferred"]
    penalty_weight = rule.params["penalty"]
    details: List[Dict[str, Any]] = []
    for day in days:
        shift = context["assignments"].get(nurse, {}).get(day)
        matches = shift_matches_pattern(shift or "", pattern)
        if preferred and not matches:
            details.append({"nurse": nurse, "day": day, "assigned_shift": shift, "expected_pattern": pattern})
        if not preferred and matches:
            details.append({"nurse": nurse, "day": day, "assigned_shift": shift, "forbidden_pattern": pattern})
    return _build_result(rule, len(details), details, len(details) * penalty_weight)


def _evaluate_coverage_target(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    day = rule.params["day"]
    requirements = rule.params["requirements"]
    under_penalty = int(rule.params.get("under_penalty", 0))
    over_penalty = int(rule.params.get("over_penalty", 0))
    counts = context["per_day_shift_counts"][day]
    details: List[Dict[str, Any]] = []
    total_penalty = 0
    for shift, required in requirements.items():
        actual = counts.get(shift, 0)
        if actual != required:
            short = max(0, required - actual)
            excess = max(0, actual - required)
            total_penalty += short * under_penalty + excess * over_penalty
            details.append({"day": day, "shift": shift, "required": required, "actual": actual, "shortage": short, "excess": excess})
    return _build_result(rule, len(details), details, total_penalty)


def _evaluate_max_total_work_shifts(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    limit = rule.params["limit"]
    work_count = sum(context["per_nurse_work_flags"][nurse])
    if work_count <= limit:
        return _build_result(rule, 0, [])
    detail = {"nurse": nurse, "work_count": work_count, "limit": limit}
    return _build_result(rule, 1, [detail], work_count - limit)


def _evaluate_max_shift_type_count_all(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    pattern = rule.params["shift_pattern"]
    limit = rule.params["limit"]
    details: List[Dict[str, Any]] = []
    for nurse, counter in context["per_nurse_shift_counts"].items():
        count = sum(value for shift, value in counter.items() if shift_matches_pattern(shift, pattern))
        if count > limit:
            details.append({"nurse": nurse, "count": count, "limit": limit, "pattern": pattern})
    return _build_result(rule, len(details), details)


def _evaluate_min_shift_type_count_all(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    pattern = rule.params["shift_pattern"]
    limit = rule.params["limit"]
    details: List[Dict[str, Any]] = []
    for nurse, counter in context["per_nurse_shift_counts"].items():
        count = sum(value for shift, value in counter.items() if shift_matches_pattern(shift, pattern))
        if count < limit:
            details.append({"nurse": nurse, "count": count, "limit": limit, "pattern": pattern})
    return _build_result(rule, len(details), details)


def _evaluate_daily_shift_coverage_all_days(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    shift = rule.params["shift"]
    required = rule.params["required"]
    details: List[Dict[str, Any]] = []
    for day in range(1, instance.number_of_days + 1):
        actual = context["per_day_shift_counts"][day].get(shift, 0)
        if actual != required:
            details.append({"day": day, "shift": shift, "required": required, "actual": actual})
    return _build_result(rule, len(details), details)


def _evaluate_max_consecutive_shift_type_all(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    pattern = rule.params["shift_pattern"]
    limit = rule.params["limit"]
    details: List[Dict[str, Any]] = []
    for nurse in range(1, instance.number_of_nurses + 1):
        flags = [
            shift_matches_pattern(context["assignments"].get(nurse, {}).get(day, ""), pattern)
            for day in range(1, instance.number_of_days + 1)
        ]
        for run in iter_runs(flags):
            if run >= limit:
                details.append({"nurse": nurse, "run_length": run, "limit": limit, "pattern": pattern})
    return _build_result(rule, len(details), details)


def _evaluate_no_isolated_shift_type_all(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    pattern = rule.params["shift_pattern"]
    details: List[Dict[str, Any]] = []
    for nurse in range(1, instance.number_of_nurses + 1):
        for day in range(2, instance.number_of_days):
            prev_shift = context["assignments"].get(nurse, {}).get(day - 1, "")
            current_shift = context["assignments"].get(nurse, {}).get(day, "")
            next_shift = context["assignments"].get(nurse, {}).get(day + 1, "")
            if shift_matches_pattern(current_shift, pattern) and not shift_matches_pattern(prev_shift, pattern) and not shift_matches_pattern(next_shift, pattern):
                details.append({"nurse": nurse, "day": day, "shift": current_shift})
    return _build_result(rule, len(details), details)


def _evaluate_max_consecutive_worked_weekends_all(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    limit = rule.params["limit"]
    details: List[Dict[str, Any]] = []
    for nurse, flags in context["per_nurse_weekend_flags"].items():
        for run in iter_runs(flags):
            if run > limit:
                details.append({"nurse": nurse, "run_length": run, "limit": limit})
    return _build_result(rule, len(details), details)


def _evaluate_weekly_total_shift_range(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    nurse = rule.params["nurse"]
    minimum = rule.params["min"]
    maximum = rule.params["max"]
    details: List[Dict[str, Any]] = []
    work_flags = context["per_nurse_work_flags"][nurse]
    for week_index in range(0, instance.number_of_days, 7):
        week_number = week_index // 7 + 1
        count = sum(work_flags[week_index:week_index + 7])
        if count < minimum or count > maximum:
            details.append({"nurse": nurse, "week": week_number, "count": count, "min": minimum, "max": maximum})
    return _build_result(rule, len(details), details)


def _evaluate_unsupported_rule(rule: Rule, instance: CanonicalInstance, context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": rule.type,
        "is_hard": rule.is_hard,
        "source_text": rule.source_text,
        "violation_count": 0,
        "penalty": 0,
        "satisfied": False,
        "details": [{"issue": "unsupported_rule_type"}],
    }


_RULE_EVALUATORS = {
    "assignment_completeness": _evaluate_assignment_completeness,
    "single_assignment_per_day": _evaluate_single_assignment,
    "fixed_days_off": _evaluate_fixed_days_off,
    "forbidden_succession": _evaluate_forbidden_succession,
    "minutes_range": _evaluate_minutes_range,
    "max_consecutive_work_days": _evaluate_max_consecutive_work,
    "max_consecutive_work_days_all": _evaluate_max_consecutive_work,
    "min_consecutive_work_days": _evaluate_min_consecutive_work,
    "min_consecutive_off_days": _evaluate_min_consecutive_off,
    "min_consecutive_off_days_all": _evaluate_min_consecutive_off,
    "max_weekends": _evaluate_max_weekends,
    "allowed_shifts": _evaluate_allowed_shifts,
    "shift_max_count": _evaluate_shift_max_count,
    "shift_preference": _evaluate_shift_preference,
    "coverage_target": _evaluate_coverage_target,
    "max_total_work_shifts": _evaluate_max_total_work_shifts,
    "max_shift_type_count_all": _evaluate_max_shift_type_count_all,
    "min_shift_type_count_all": _evaluate_min_shift_type_count_all,
    "daily_shift_coverage_all_days": _evaluate_daily_shift_coverage_all_days,
    "max_consecutive_shift_type_all": _evaluate_max_consecutive_shift_type_all,
    "no_isolated_shift_type_all": _evaluate_no_isolated_shift_type_all,
    "max_consecutive_worked_weekends_all": _evaluate_max_consecutive_worked_weekends_all,
    "weekly_total_shift_range": _evaluate_weekly_total_shift_range,
}
