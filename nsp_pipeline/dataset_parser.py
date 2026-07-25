from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _normalize_shift_name(shift: Any) -> str:
    """
    处理 YAML 里 `off` 被解析成布尔值的情况。
    """
    if isinstance(shift, bool):
        return "off" if shift is False else "on"
    return str(shift)


def load_dataset_description(description_path: Path) -> Dict[str, Any]:
    """
    读取 `sythetic/*/description.txt` 或 `existing/*/description.txt`
    这种 YAML 风格的问题描述文件，并归一化为统一结构。
    """
    with description_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    required = {"instance_id", "problem_type", "basic_setting", "hard_constraints"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Dataset description is missing fields: {sorted(missing)}")

    if "all_preferences" in data:
        data["dataset_source"] = "synthetic"
        data["all_preferences"] = _normalize_preference_items(data["all_preferences"])
        return data

    if "soft_constraints" in data:
        data["dataset_source"] = "existing"
        data["all_preferences"] = [
            {"id": f"SC{index + 1}", "text": _normalize_constraint_text(item)}
            for index, item in enumerate(data.get("soft_constraints", []))
        ]
        return data

    raise ValueError("Dataset description must contain either `all_preferences` or `soft_constraints`.")


def _normalize_constraint_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and len(item) == 1:
        key, value = next(iter(item.items()))
        return f"{key}: {value}"
    return str(item)


def _normalize_preference_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            normalized.append({"id": f"P{index}", "text": str(item)})
            continue
        normalized.append(
            {
                "id": str(item.get("id", f"P{index}")),
                "text": str(item.get("text", "")),
            }
        )
    return normalized


def split_preferences(problem: Dict[str, Any], initial_count: int = 8, extra_count: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按实验设定拆分 preference:
    - 前 8 条用于初始建模
    - 后 2 条用于后续增量修正
    """
    preferences = problem.get("all_preferences", [])
    if len(preferences) < initial_count + extra_count:
        raise ValueError(
            f"Expected at least {initial_count + extra_count} preferences, found {len(preferences)}."
        )

    return preferences[:initial_count], preferences[initial_count:initial_count + extra_count]


def build_runtime_data(problem: Dict[str, Any], active_preferences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    构造传给生成 solver 的运行时数据。

    这里的数据结构尽量简单、扁平，方便 LLM 生成的代码直接读取。
    """
    basic = problem["basic_setting"]
    raw_shift_types = basic.get("shift_types", [])
    shift_types = [_normalize_shift_name(shift) for shift in raw_shift_types]

    return {
        "InstanceId": problem["instance_id"],
        "ProblemType": problem["problem_type"],
        "NumberOfNurses": basic["number_of_nurses"],
        "NumberOfDays": basic["number_of_days"],
        "ShiftTypes": shift_types,
        "WorkShiftTypes": [shift for shift in shift_types if shift.lower() != "off"],
        "HardConstraints": problem["hard_constraints"],
        "ActivePreferences": active_preferences,
        "DatasetSource": problem.get("dataset_source", "unknown"),
    }


def build_problem_text(problem: Dict[str, Any], active_preferences: List[Dict[str, Any]]) -> str:
    """
    把结构化问题描述转成更适合 prompt 阅读的文本。
    """
    basic = problem["basic_setting"]
    raw_shift_types = basic.get("shift_types", [])
    shift_types = [_normalize_shift_name(shift) for shift in raw_shift_types]
    lines = [
        f"Instance ID: {problem['instance_id']}",
        f"Problem Type: {problem['problem_type']}",
        f"Dataset Source: {problem.get('dataset_source', 'unknown')}",
        f"Number of nurses: {basic['number_of_nurses']}",
        f"Number of days: {basic['number_of_days']}",
        f"Shift types: {', '.join(shift_types)}",
        "",
        "Hard constraints:",
    ]
    lines.extend(f"- {constraint}" for constraint in problem["hard_constraints"])
    lines.append("")
    section_title = "Active nurse preferences:" if problem.get("dataset_source") == "synthetic" else "Active soft constraints:"
    lines.append(section_title)
    lines.extend(f"- {item['id']}: {item['text']}" for item in active_preferences)
    return "\n".join(lines)
