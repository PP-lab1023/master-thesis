from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml


def load_yaml_like_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_shift_name(value: Any) -> str:
    if isinstance(value, bool):
        return "off" if value is False else "on"
    return str(value)


def parse_day_numbers(text: str) -> List[int]:
    return [int(value) for value in re.findall(r"Day\s+(\d+)", text)]


def parse_range_days(text: str) -> List[int]:
    match = re.search(r"Days?\s+(\d+)\s*[–-]\s*(\d+)", text)
    if not match:
        return []
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        start, end = end, start
    return list(range(start, end + 1))


def expand_days_from_text(text: str) -> List[int]:
    range_days = parse_range_days(text)
    if range_days:
        return range_days
    return parse_day_numbers(text)


def parse_shift_list(text: str) -> List[str]:
    cleaned = text.strip().rstrip(".")
    cleaned = cleaned.replace(" and ", ", ")
    parts = [part.strip() for part in cleaned.split(",")]
    return [part for part in parts if part]


def parse_inline_shift_pairs(text: str) -> Dict[str, int]:
    pairs: Dict[str, int] = {}
    for amount, shift in re.findall(r"(\d+)\s+([A-Za-z0-9_]+)", text):
        pairs[shift] = int(amount)
    return pairs


def shift_matches_pattern(shift: str, pattern: str) -> bool:
    if pattern.endswith("*"):
        return shift.startswith(pattern[:-1])
    return shift == pattern


def day_of_week(day: int, first_day: str = "Monday") -> str:
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        offset = weekdays.index(first_day)
    except ValueError:
        offset = 0
    return weekdays[(offset + day - 1) % 7]


def weekend_pairs(number_of_days: int, first_day: str = "Monday") -> List[Sequence[int]]:
    pairs: List[Sequence[int]] = []
    current_saturday = None
    for day in range(1, number_of_days + 1):
        weekday = day_of_week(day, first_day)
        if weekday == "Saturday":
            current_saturday = day
        elif weekday == "Sunday" and current_saturday is not None:
            pairs.append((current_saturday, day))
            current_saturday = None
    return pairs


def iter_runs(values: Iterable[bool]) -> List[int]:
    runs: List[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def parse_nurse_numbers(text: str) -> List[int]:
    if "Nurses " in text:
        payload = text.split("Nurses ", 1)[1].split(" should", 1)[0].split(" must", 1)[0].split(" can", 1)[0]
        payload = payload.replace(" and ", ", ")
        return [int(value) for value in re.findall(r"\d+", payload)]
    match = re.search(r"Nurse\s+(\d+)", text)
    return [int(match.group(1))] if match else []
