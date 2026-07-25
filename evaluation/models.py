from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Rule:
    type: str
    source_text: str
    params: Dict[str, Any] = field(default_factory=dict)
    is_hard: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalInstance:
    instance_id: str
    problem_type: str
    source_type: str
    number_of_nurses: int
    number_of_days: int
    shift_types: List[str]
    shift_durations: Dict[str, int] = field(default_factory=dict)
    first_day_of_scheduling_period: str = "Monday"
    hard_rules: List[Rule] = field(default_factory=list)
    soft_rules: List[Rule] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    unsupported_rules: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "problem_type": self.problem_type,
            "source_type": self.source_type,
            "number_of_nurses": self.number_of_nurses,
            "number_of_days": self.number_of_days,
            "shift_types": self.shift_types,
            "shift_durations": self.shift_durations,
            "first_day_of_scheduling_period": self.first_day_of_scheduling_period,
            "hard_rules": [rule.to_dict() for rule in self.hard_rules],
            "soft_rules": [rule.to_dict() for rule in self.soft_rules],
            "meta": self.meta,
            "unsupported_rules": self.unsupported_rules,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalInstance":
        return cls(
            instance_id=data["instance_id"],
            problem_type=data["problem_type"],
            source_type=data["source_type"],
            number_of_nurses=data["number_of_nurses"],
            number_of_days=data["number_of_days"],
            shift_types=list(data.get("shift_types", [])),
            shift_durations=dict(data.get("shift_durations", {})),
            first_day_of_scheduling_period=data.get("first_day_of_scheduling_period", "Monday"),
            hard_rules=[Rule(**rule) for rule in data.get("hard_rules", [])],
            soft_rules=[Rule(**rule) for rule in data.get("soft_rules", [])],
            meta=dict(data.get("meta", {})),
            unsupported_rules=list(data.get("unsupported_rules", [])),
            warnings=list(data.get("warnings", [])),
        )
