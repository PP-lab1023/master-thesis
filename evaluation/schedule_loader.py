from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class NormalizedSchedule:
    assignments: Dict[int, Dict[int, str]]
    duplicates: List[Tuple[int, int, str, str]] = field(default_factory=list)
    raw_warnings: List[str] = field(default_factory=list)


def load_schedule(schedule_path: Path) -> NormalizedSchedule:
    with schedule_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return normalize_schedule(data)


def normalize_schedule(data: Any) -> NormalizedSchedule:
    if isinstance(data, dict) and "assignments" in data:
        return _from_assignment_list(data["assignments"])
    if isinstance(data, dict) and "schedule" in data:
        return _from_nested_mapping(data["schedule"])
    if isinstance(data, dict):
        return _from_nested_mapping(data)
    if isinstance(data, list):
        return _from_assignment_list(data)
    raise ValueError("Unsupported schedule JSON format.")


def _from_assignment_list(items: List[Dict[str, Any]]) -> NormalizedSchedule:
    assignments: Dict[int, Dict[int, str]] = {}
    duplicates: List[Tuple[int, int, str, str]] = []
    for item in items:
        nurse = int(item["nurse"])
        day = int(item["day"])
        shift = str(item["shift"])
        existing = assignments.setdefault(nurse, {}).get(day)
        if existing is not None and existing != shift:
            duplicates.append((nurse, day, existing, shift))
        assignments.setdefault(nurse, {})[day] = shift
    return NormalizedSchedule(assignments=assignments, duplicates=duplicates)


def _from_nested_mapping(mapping: Dict[str, Any]) -> NormalizedSchedule:
    assignments: Dict[int, Dict[int, str]] = {}
    warnings: List[str] = []
    for raw_nurse, row in mapping.items():
        nurse = int(raw_nurse)
        if isinstance(row, list):
            assignments[nurse] = {day + 1: str(shift) for day, shift in enumerate(row)}
            continue
        if isinstance(row, dict):
            assignments[nurse] = {int(day): str(shift) for day, shift in row.items()}
            continue
        warnings.append(f"Unsupported row format for nurse {raw_nurse}.")
    return NormalizedSchedule(assignments=assignments, raw_warnings=warnings)
