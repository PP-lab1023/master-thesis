from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .existing_parser import parse_existing_instance
from .models import CanonicalInstance
from .synthetic_parser import parse_synthetic_instance


def detect_source_type(description_path: Path) -> str:
    lowered = str(description_path).lower()
    if "/existing/" in lowered:
        return "existing"
    if "/sythetic/" in lowered or "/synthetic/" in lowered:
        return "synthetic"
    raise ValueError(f"Could not infer source type from path: {description_path}")


def canonicalize_description(description_path: Path, source_type: Optional[str] = None) -> CanonicalInstance:
    source = source_type or detect_source_type(description_path)
    if source == "existing":
        return parse_existing_instance(description_path)
    if source == "synthetic":
        return parse_synthetic_instance(description_path)
    raise ValueError(f"Unsupported source type: {source}")


def write_canonical_json(instance: CanonicalInstance, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(instance.to_dict(), file, ensure_ascii=False, indent=2)


def load_canonical_json(path: Path) -> CanonicalInstance:
    with path.open("r", encoding="utf-8") as file:
        return CanonicalInstance.from_dict(json.load(file))
