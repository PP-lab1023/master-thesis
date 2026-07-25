from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .canonicalizer import canonicalize_description, load_canonical_json, write_canonical_json
from .evaluator import evaluate_canonical_instance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic evaluation utilities for nurse scheduling instances.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonicalize = subparsers.add_parser("canonicalize", help="Convert description.txt to canonical instance JSON.")
    canonicalize.add_argument("description_path", type=Path)
    canonicalize.add_argument("--source-type", choices=["existing", "synthetic"])
    canonicalize.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a schedule JSON against a canonical JSON instance.")
    evaluate.add_argument("canonical_json", type=Path)
    evaluate.add_argument("schedule_json", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to save the full evaluation JSON payload.",
    )

    return parser


def _top_violated_rules(results: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    failed = [item for item in results if item.get("violation_count", 0) > 0]
    failed.sort(key=lambda item: item.get("violation_count", 0), reverse=True)
    return failed[:limit]


def _format_evaluation_summary(result: Dict[str, Any]) -> str:
    lines = [
        "Schedule validation summary",
        f"Instance: {result['instance_id']}",
        f"Source type: {result['source_type']}",
        f"Validation passed: {'YES' if result['feasible'] else 'NO'}",
        f"Hard violations: {result['hard_violation_count']}",
        f"Soft penalty: {result['soft_penalty_total']}",
    ]

    unsupported_count = len(result.get("unsupported_rules", []))
    if unsupported_count:
        lines.append(f"Unsupported rules: {unsupported_count}")

    instance_warnings = result.get("instance_warnings", [])
    schedule_warnings = result.get("schedule_warnings", [])
    if instance_warnings:
        lines.append(f"Instance warnings: {len(instance_warnings)}")
    if schedule_warnings:
        lines.append(f"Schedule warnings: {len(schedule_warnings)}")

    if result["feasible"]:
        lines.append("Conclusion: the schedule satisfies all hard constraints.")
    else:
        lines.append("Conclusion: the schedule does not satisfy all hard constraints.")
        top_failed = _top_violated_rules(result.get("hard_results", []))
        if top_failed:
            lines.append("Main failing hard constraints:")
            for item in top_failed:
                source_text = str(item.get("source_text", "")).strip()
                short_text = source_text if len(source_text) <= 120 else source_text[:117] + "..."
                lines.append(f"- {item['type']}: {item['violation_count']} violation(s)")
                if short_text:
                    lines.append(f"  {short_text}")

    if result["soft_penalty_total"] > 0:
        top_soft = _top_violated_rules(result.get("soft_results", []), limit=3)
        if top_soft:
            lines.append("Main soft-constraint penalties:")
            for item in top_soft:
                source_text = str(item.get("source_text", "")).strip()
                short_text = source_text if len(source_text) <= 120 else source_text[:117] + "..."
                lines.append(f"- {item['type']}: penalty {item['penalty']}")
                if short_text:
                    lines.append(f"  {short_text}")

    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "canonicalize":
        instance = canonicalize_description(args.description_path, source_type=args.source_type)
        write_canonical_json(instance, args.output)
        return

    if args.command == "evaluate":
        instance = load_canonical_json(args.canonical_json)
        result = evaluate_canonical_instance(instance, args.schedule_json)
        summary = _format_evaluation_summary(result)
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(summary + "\n", encoding="utf-8")
        if args.json_output:
            args.json_output.write_text(payload + "\n", encoding="utf-8")
        print(summary)


if __name__ == "__main__":
    main()
