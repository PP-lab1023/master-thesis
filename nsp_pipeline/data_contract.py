import json
from pathlib import Path
from typing import Any, Dict


def load_experiment_data(data_path: Path) -> Dict[str, Any]:
    """
    读取新的实验输入格式，并转换成 solver 内部使用的扁平字典。

    外部 JSON 采用更清晰的分层结构:
    - instance: 数据集元信息
    - dimensions: 规模信息
    - sets: 一些集合或标签
    - parameters: 真正参与建模的输入参数

    但为了不把所有 prompt 和 solver 约定一起改掉，这里会在加载时
    统一转换成内部兼容格式，例如 `N`, `D`, `availableSlots` 等。
    """
    with data_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    dimensions = raw["dimensions"]
    parameters = raw["Parameters"]
    sets = raw.get("sets", {})
    instance = raw.get("instance", {})

    return {
        "InstanceId": instance.get("instance_id", data_path.stem),
        "Description": instance.get("description", ""),
        "ShiftNames": sets.get("shift_names", []),
        "N": dimensions["num_nurses"],
        "D": dimensions["num_days"],
        "T": dimensions["num_shifts"],
        "demandSlot": parameters["demand"],
        "MinDays": parameters["min_work_days"],
        "MaxDays": parameters["max_work_days"],
        "MaxConsecutiveWorkDays": parameters["max_consecutive_work_days"],
        "MaxNightShifts": parameters["max_night_shifts"],
    }

