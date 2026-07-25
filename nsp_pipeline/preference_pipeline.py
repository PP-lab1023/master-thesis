import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from nsp_pipeline.dataset_parser import (
    build_problem_text,
    build_runtime_data,
    load_dataset_description,
    split_preferences,
)
from nsp_pipeline.llm import DEFAULT_OPENROUTER_MODEL, apply_cot_prompt, request_llm
from nsp_pipeline.parsers import extract_json_object, extract_python_code
from nsp_pipeline.preference_prompts import (
    CONSTRAINT_CODE_PROMPT,
    CONSTRAINT_REPAIR_PROMPT,
    PREFERENCE_PARSE_PROMPT,
    PREFERENCE_REVIEW_PROMPT,
    TARGETED_PREFERENCE_REVIEW_PROMPT,
)
from nsp_pipeline.runner import execute_solver
from nsp_pipeline.solver_template import assemble_solver_code


class PreferenceCorrectionPipeline:
    """
    实现新的实验流程:
    1. 先用 hard constraints + 前 k 条 preferences 逐条生成约束代码
    2. 再把剩余 preferences 按顺序一条条加入
    3. 每加入一条 preference，都先检查当前排班表是否满足
    4. 只有当当前表不满足该 preference 时，才生成新的约束代码并重跑 solver
    """

    def __init__(
        self,
        description_path: Path,
        output_dir: Path,
        model: str = DEFAULT_OPENROUTER_MODEL,
        review_model: str | None = None,
        repair_model: str | None = None,
        initial_preferences_count: int = 8,
        review_format: str = "table",
        review_strategy: str = "direct",
        retries: int = 3,
        use_cot: bool = False,
        self_consistency_samples: int = 1,
        self_consistency_temperature: float = 0.7,
    ):
        self.description_path = description_path
        self.output_dir = output_dir
        self.model = model
        self.review_model = review_model or model
        self.repair_model = repair_model or model
        self.initial_preferences_count = initial_preferences_count
        self.review_format = review_format
        self.review_strategy = review_strategy
        self.retries = retries
        self.use_cot = use_cot
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.self_consistency_temperature = self_consistency_temperature
        self.problem = load_dataset_description(description_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_prompt(self, prompt: str) -> str:
        return apply_cot_prompt(prompt) if self.use_cot else prompt

    def _llm_temperature(self, sample_index: int) -> float:
        if self.self_consistency_samples <= 1:
            return 0.0
        return self.self_consistency_temperature

    def _call_json(self, prompt: str, model: str) -> Dict[str, Any]:
        prepared_prompt = self._prepare_prompt(prompt)
        if self.self_consistency_samples <= 1:
            last_error = None
            for _ in range(self.retries):
                try:
                    response = request_llm(prepared_prompt, model=model, temperature=0.0)
                    return extract_json_object(response)
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"Failed to parse JSON response after {self.retries} attempts: {last_error}")

        parsed_candidates: List[Dict[str, Any]] = []
        serialized_counts: Counter[str] = Counter()
        last_error = None
        for sample_index in range(self.self_consistency_samples):
            try:
                response = request_llm(
                    prepared_prompt,
                    model=model,
                    temperature=self._llm_temperature(sample_index),
                )
                candidate = extract_json_object(response)
                parsed_candidates.append(candidate)
                serialized_counts[json.dumps(candidate, ensure_ascii=False, sort_keys=True)] += 1
            except Exception as exc:
                last_error = exc
        if not parsed_candidates:
            raise RuntimeError(
                f"Failed to parse any JSON response across {self.self_consistency_samples} self-consistency samples: {last_error}"
            )
        best_serialized, _ = serialized_counts.most_common(1)[0]
        return json.loads(best_serialized)

    def _score_code_candidate(
        self,
        code: str,
        runtime_data: Optional[Dict[str, Any]],
        active_constraints: Optional[List[str]],
    ) -> Tuple[int, int]:
        if runtime_data is None or active_constraints is None:
            return (0, 0)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".py",
                dir=self.output_dir,
                delete=False,
            ) as handle:
                handle.write(code)
                temp_path = Path(handle.name)
            execution = execute_solver(temp_path, runtime_data, active_constraints)
            return (1 if execution.get("ok") else 0, 1 if execution.get("result") else 0)
        except Exception:
            return (0, 0)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _call_code(
        self,
        prompt: str,
        model: str,
        runtime_data: Optional[Dict[str, Any]] = None,
        active_constraints: Optional[List[str]] = None,
    ) -> str:
        prepared_prompt = self._prepare_prompt(prompt)
        if self.self_consistency_samples <= 1:
            last_error = None
            for _ in range(self.retries):
                try:
                    response = request_llm(prepared_prompt, model=model, temperature=0.0)
                    return extract_python_code(response)
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"Failed to parse code response after {self.retries} attempts: {last_error}")

        code_candidates: List[str] = []
        candidate_counts: Counter[str] = Counter()
        last_error = None
        for sample_index in range(self.self_consistency_samples):
            try:
                response = request_llm(
                    prepared_prompt,
                    model=model,
                    temperature=self._llm_temperature(sample_index),
                )
                code = extract_python_code(response)
                code_candidates.append(code)
                candidate_counts[code.strip()] += 1
            except Exception as exc:
                last_error = exc
        if not code_candidates:
            raise RuntimeError(
                f"Failed to parse any code response across {self.self_consistency_samples} self-consistency samples: {last_error}"
            )

        best_code = None
        best_score: Tuple[int, int, int] = (-1, -1, -1)
        for code in code_candidates:
            execution_score = self._score_code_candidate(code, runtime_data, active_constraints)
            frequency = candidate_counts[code.strip()]
            score = (execution_score[0], execution_score[1], frequency)
            if score > best_score:
                best_score = score
                best_code = code
        return best_code or code_candidates[0]

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _write_text(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def _render_schedule_representation(self, schedule_table: Dict[str, Any]) -> str:
        """
        按实验设置把排班结果渲染成给 review LLM 看的表示方式。
        """
        if self.review_format == "json":
            return json.dumps(schedule_table, ensure_ascii=False, indent=2)

        if self.review_format == "table":
            days = schedule_table.get("days", [])
            nurses = schedule_table.get("nurses", [])
            rows = schedule_table.get("rows", [])
            if not days or not nurses or not rows:
                return "(empty schedule table)"
            table_df = pd.DataFrame(data=rows, index=nurses, columns=days)
            return table_df.to_string()

        raise ValueError(f"Unsupported review_format: {self.review_format}")

    def _format_execution_payload(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据当前 review_format，统一 execution.json 里 `schedule_table` 的表现形式。
        """
        if self.review_format != "table":
            return execution

        formatted = dict(execution)
        result = formatted.get("result")
        if not isinstance(result, dict):
            return formatted

        schedule_table = result.get("schedule_table")
        if isinstance(schedule_table, dict):
            updated_result = dict(result)
            updated_result["schedule_table"] = self._render_schedule_representation(schedule_table)
            formatted["result"] = updated_result
        return formatted

    def _implemented_constraints_text(self, items: List[Dict[str, str]]) -> str:
        if not items:
            return "(none yet)"
        return "\n".join(f'- {item["id"]}: {item["text"]}' for item in items)

    def _parse_preference(self, problem_text: str, preference: Dict[str, str]) -> Dict[str, Any]:
        return self._call_json(
            PREFERENCE_PARSE_PROMPT.format(
                problem_text=problem_text,
                preference_id=preference["id"],
                preference_text=preference["text"],
            ),
            model=self.review_model,
        )

    def _extract_observed_facts(self, parsed_preference: Dict[str, Any], schedule_table: Dict[str, Any]) -> Dict[str, Any]:
        nurses = schedule_table.get("nurses", [])
        days = schedule_table.get("days", [])
        rows = schedule_table.get("rows", [])

        nurse_name = parsed_preference.get("nurse_name")
        target_days = parsed_preference.get("target_days_1based", [])
        observed_entries: List[Dict[str, Any]] = []

        if nurse_name not in nurses:
            return {
                "nurse_name": nurse_name,
                "found_nurse": False,
                "entries": observed_entries,
            }

        nurse_idx = nurses.index(nurse_name)
        for day_1based in target_days:
            day_label = f"Day {day_1based}"
            assigned_shift = None
            if day_label in days and nurse_idx < len(rows):
                day_idx = days.index(day_label)
                if day_idx < len(rows[nurse_idx]):
                    assigned_shift = rows[nurse_idx][day_idx]
            observed_entries.append(
                {
                    "day_1based": day_1based,
                    "day_label": day_label,
                    "assigned_shift": assigned_shift,
                }
            )

        return {
            "nurse_name": nurse_name,
            "found_nurse": True,
            "entries": observed_entries,
        }

    def _review_preferences_direct(
        self,
        full_problem_text: str,
        schedule_representation: str,
        new_preference_text: str,
    ) -> Dict[str, Any]:
        return self._call_json(
            PREFERENCE_REVIEW_PROMPT.format(
                problem_text=full_problem_text,
                schedule_format=self.review_format,
                schedule_representation=schedule_representation,
                new_preference_text=new_preference_text,
            ),
            model=self.review_model,
        )

    def _review_preference_targeted(
        self,
        full_problem_text: str,
        preference: Dict[str, str],
        schedule_table: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        parsed = self._parse_preference(full_problem_text, preference)
        observed = self._extract_observed_facts(parsed, schedule_table)
        judgment = self._call_json(
            TARGETED_PREFERENCE_REVIEW_PROMPT.format(
                problem_text=full_problem_text,
                parsed_preference_json=json.dumps(parsed, ensure_ascii=False, indent=2),
                observed_facts_json=json.dumps(observed, ensure_ascii=False, indent=2),
            ),
            model=self.review_model,
        )

        review = {
            "all_satisfied": bool(judgment.get("satisfied", False)),
            "summary": judgment.get("reason", ""),
            "violations": [] if judgment.get("satisfied", False) else [
                {
                    "preference_id": judgment.get("preference_id", preference["id"]),
                    "reason": judgment.get("reason", ""),
                    "evidence": "; ".join(judgment.get("evidence", [])),
                }
            ],
            "suggestions": judgment.get("suggestions", []),
        }
        artifact = {
            "preference": preference,
            "parsed_preference": parsed,
            "observed_facts": observed,
            "judgment": judgment,
        }
        return review, artifact

    def _run_review_for_preference(
        self,
        full_problem_text: str,
        preference: Dict[str, str],
        schedule_table: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None, str]:
        schedule_representation = self._render_schedule_representation(schedule_table)
        if self.review_strategy == "direct":
            review = self._review_preferences_direct(
                full_problem_text=full_problem_text,
                schedule_representation=schedule_representation,
                new_preference_text=f'- {preference["id"]}: {preference["text"]}',
            )
            return review, None, schedule_representation
        if self.review_strategy == "targeted":
            review, parsed_artifact = self._review_preference_targeted(
                full_problem_text=full_problem_text,
                preference=preference,
                schedule_table=schedule_table,
            )
            return review, parsed_artifact, schedule_representation
        raise ValueError(f"Unsupported review_strategy: {self.review_strategy}")

    def _generate_constraint_block(
        self,
        problem_text: str,
        runtime_data: Dict[str, Any],
        implemented_items: List[Dict[str, str]],
        constraint_kind: str,
        constraint_id: str,
        constraint_text: str,
    ) -> str:
        return self._call_code(
            CONSTRAINT_CODE_PROMPT.format(
                problem_text=problem_text,
                runtime_data_json=json.dumps(runtime_data, ensure_ascii=False, indent=2),
                constraint_kind=constraint_kind,
                constraint_id=constraint_id,
                constraint_text=constraint_text,
                implemented_constraints_text=self._implemented_constraints_text(implemented_items),
            ),
            model=self.model,
            runtime_data=runtime_data,
            active_constraints=[item["text"] for item in implemented_items] + [constraint_text],
        )

    def _repair_constraint_block(
        self,
        problem_text: str,
        runtime_data: Dict[str, Any],
        schedule_representation: str,
        review_json: Dict[str, Any],
        constraint_kind: str,
        constraint_id: str,
        constraint_text: str,
        current_constraint_code: str,
    ) -> str:
        return self._call_code(
            CONSTRAINT_REPAIR_PROMPT.format(
                problem_text=problem_text,
                runtime_data_json=json.dumps(runtime_data, ensure_ascii=False, indent=2),
                schedule_format=self.review_format,
                schedule_representation=schedule_representation,
                review_json=json.dumps(review_json, ensure_ascii=False, indent=2),
                constraint_kind=constraint_kind,
                constraint_id=constraint_id,
                constraint_text=constraint_text,
                current_constraint_code=current_constraint_code,
            ),
            model=self.repair_model,
            runtime_data=runtime_data,
            active_constraints=[constraint_text],
        )

    def run(self) -> Dict[str, Any]:
        all_preferences = self.problem["all_preferences"]
        if not 0 <= self.initial_preferences_count <= len(all_preferences):
            raise ValueError(
                f"initial_preferences_count must be between 0 and {len(all_preferences)}, got {self.initial_preferences_count}."
            )

        initial_preferences = all_preferences[: self.initial_preferences_count]
        new_preferences = all_preferences[self.initial_preferences_count :]
        initial_runtime_data = build_runtime_data(self.problem, initial_preferences)
        initial_problem_text = build_problem_text(self.problem, initial_preferences)

        hard_constraint_items = [
            {"id": f"HC{index + 1}", "text": text, "kind": "hard_constraint"}
            for index, text in enumerate(self.problem["hard_constraints"])
        ]
        initial_preference_items = [
            {"id": item["id"], "text": item["text"], "kind": "preference"}
            for item in initial_preferences
        ]

        manifest: Dict[str, Any] = {
            "description_path": str(self.description_path.resolve()),
            "model": self.model,
            "review_model": self.review_model,
            "repair_model": self.repair_model,
            "use_cot": self.use_cot,
            "self_consistency_samples": self.self_consistency_samples,
            "self_consistency_temperature": self.self_consistency_temperature,
            "initial_preferences_count": self.initial_preferences_count,
            "remaining_preferences_count": len(new_preferences),
            "review_format": self.review_format,
            "review_strategy": self.review_strategy,
            "steps": [],
        }

        self._write_json(self.output_dir / "step_00_runtime_data.json", initial_runtime_data)

        implemented_items: List[Dict[str, str]] = []
        persistent_blocks: List[Dict[str, str]] = []
        for item in hard_constraint_items + initial_preference_items:
            block = self._generate_constraint_block(
                problem_text=initial_problem_text,
                runtime_data=initial_runtime_data,
                implemented_items=implemented_items,
                constraint_kind=item["kind"],
                constraint_id=item["id"],
                constraint_text=item["text"],
            )
            persistent_blocks.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "text": item["text"],
                    "code": block,
                }
            )
            implemented_items.append({"id": item["id"], "text": item["text"]})

        self._write_json(self.output_dir / "step_00_constraint_blocks.json", {"blocks": persistent_blocks})
        initial_code = assemble_solver_code([item["code"] for item in persistent_blocks])
        initial_code_path = self.output_dir / "step_00_initial_solver.py"
        self._write_text(initial_code_path, initial_code)
        initial_execution = execute_solver(
            initial_code_path,
            initial_runtime_data,
            self.problem["hard_constraints"] + [item["text"] for item in initial_preferences],
        )
        initial_execution_to_save = self._format_execution_payload(initial_execution)
        self._write_json(self.output_dir / "step_00_execution.json", initial_execution_to_save)
        manifest["steps"].append(
            {
                "step": 0,
                "kind": "initial_generation",
                "active_preference_ids": [item["id"] for item in initial_preferences],
                "runtime_data_path": str((self.output_dir / "step_00_runtime_data.json").resolve()),
                "constraint_blocks_path": str((self.output_dir / "step_00_constraint_blocks.json").resolve()),
                "solver_path": str(initial_code_path.resolve()),
                "execution_path": str((self.output_dir / "step_00_execution.json").resolve()),
            }
        )

        active_preferences = list(initial_preferences)
        incremental_blocks: List[Dict[str, str]] = []
        current_execution = initial_execution

        for step_index, preference in enumerate(new_preferences, start=1):
            full_runtime_data = build_runtime_data(self.problem, active_preferences + [preference])
            full_problem_text = build_problem_text(self.problem, active_preferences + [preference])
            schedule_table = {}
            if current_execution.get("ok") and current_execution.get("result"):
                schedule_table = current_execution["result"].get("schedule_table", {})
            review, parsed_artifact, schedule_representation = self._run_review_for_preference(
                full_problem_text=full_problem_text,
                preference=preference,
                schedule_table=schedule_table,
            )
            schedule_path = self.output_dir / f"step_{step_index:02d}_schedule_for_review.txt"
            self._write_text(schedule_path, schedule_representation)
            review_path = self.output_dir / f"step_{step_index:02d}_review.json"
            runtime_data_path = self.output_dir / f"step_{step_index:02d}_runtime_data.json"
            self._write_json(runtime_data_path, full_runtime_data)

            constraint_blocks_path = self.output_dir / f"step_{step_index:02d}_constraint_blocks.json"
            code_path = self.output_dir / f"step_{step_index:02d}_repaired_solver.py"
            execution_path = self.output_dir / f"step_{step_index:02d}_execution.json"
            review_payload: Dict[str, Any] = {
                "step": step_index,
                "preference_id": preference["id"],
                "initial_review": review,
            }
            if parsed_artifact is not None:
                review_payload["initial_parsed_preference"] = parsed_artifact

            execution_payload: Dict[str, Any] = {
                "step": step_index,
                "preference_id": preference["id"],
                "before_update": self._format_execution_payload(current_execution),
            }

            if review.get("all_satisfied", False):
                self._write_json(
                    constraint_blocks_path,
                    {
                        "persistent_blocks": persistent_blocks,
                        "incremental_blocks": incremental_blocks,
                        "skipped_generation_for_preference_id": preference["id"],
                        "reason": "Current schedule already satisfies this new preference.",
                    },
                )
                self._write_text(
                    code_path,
                    "# No solver regeneration was needed for this step because the current schedule already satisfied the new preference.\n",
                )
                execution = current_execution
                execution_payload["after_first_update"] = self._format_execution_payload(current_execution)
                execution_payload["after_second_update"] = None
                review_payload["after_first_update_review"] = review
                review_payload["after_second_update_review"] = None
                self._write_json(review_path, review_payload)
                self._write_json(execution_path, execution_payload)
                step_status = "already_satisfied"
            else:
                new_code = self._generate_constraint_block(
                    problem_text=full_problem_text,
                    runtime_data=full_runtime_data,
                    implemented_items=[
                        {"id": entry["id"], "text": entry["text"]}
                        for entry in persistent_blocks + incremental_blocks
                    ],
                    constraint_kind="preference",
                    constraint_id=preference["id"],
                    constraint_text=preference["text"],
                )
                incremental_blocks.append(
                    {
                        "id": preference["id"],
                        "kind": "preference",
                        "text": preference["text"],
                        "code": new_code,
                    }
                )
                self._write_json(
                    constraint_blocks_path,
                    {"persistent_blocks": persistent_blocks, "incremental_blocks": incremental_blocks},
                )
                repaired_code = assemble_solver_code(
                    [item["code"] for item in persistent_blocks] + [item["code"] for item in incremental_blocks]
                )
                self._write_text(code_path, repaired_code)
                first_execution = execute_solver(
                    code_path,
                    full_runtime_data,
                    self.problem["hard_constraints"] + [item["text"] for item in active_preferences + [preference]],
                )
                execution = first_execution
                first_review = {
                    "all_satisfied": False,
                    "summary": "Execution failed before post-update review.",
                    "violations": [],
                    "suggestions": [],
                }
                first_parsed_artifact = None

                updated_schedule_table = {}
                if first_execution.get("ok") and first_execution.get("result"):
                    updated_schedule_table = first_execution["result"].get("schedule_table", {})
                    first_review, first_parsed_artifact, _ = self._run_review_for_preference(
                        full_problem_text=full_problem_text,
                        preference=preference,
                        schedule_table=updated_schedule_table,
                    )

                review_payload["after_first_update_review"] = first_review
                if first_parsed_artifact is not None:
                    review_payload["after_first_update_parsed_preference"] = first_parsed_artifact
                execution_payload["after_first_update"] = self._format_execution_payload(first_execution)

                second_review = None
                second_parsed_artifact = None
                second_execution_payload = None

                if first_execution.get("ok") and not first_review.get("all_satisfied", False):
                    repaired_block_code = self._repair_constraint_block(
                        problem_text=full_problem_text,
                        runtime_data=full_runtime_data,
                        schedule_representation=self._render_schedule_representation(updated_schedule_table),
                        review_json=first_review,
                        constraint_kind="preference",
                        constraint_id=preference["id"],
                        constraint_text=preference["text"],
                        current_constraint_code=incremental_blocks[-1]["code"],
                    )
                    incremental_blocks[-1]["code"] = repaired_block_code
                    self._write_json(
                        constraint_blocks_path,
                        {"persistent_blocks": persistent_blocks, "incremental_blocks": incremental_blocks},
                    )
                    twice_repaired_code = assemble_solver_code(
                        [item["code"] for item in persistent_blocks] + [item["code"] for item in incremental_blocks]
                    )
                    self._write_text(code_path, twice_repaired_code)
                    second_execution = execute_solver(
                        code_path,
                        full_runtime_data,
                        self.problem["hard_constraints"] + [item["text"] for item in active_preferences + [preference]],
                    )
                    execution = second_execution
                    second_execution_payload = self._format_execution_payload(second_execution)

                    second_schedule_table = {}
                    if second_execution.get("ok") and second_execution.get("result"):
                        second_schedule_table = second_execution["result"].get("schedule_table", {})
                        second_review, second_parsed_artifact, _ = self._run_review_for_preference(
                            full_problem_text=full_problem_text,
                            preference=preference,
                            schedule_table=second_schedule_table,
                        )
                    else:
                        second_review = {
                            "all_satisfied": False,
                            "summary": "Execution failed after second update.",
                            "violations": [],
                            "suggestions": [],
                        }

                review_payload["after_second_update_review"] = second_review
                if second_parsed_artifact is not None:
                    review_payload["after_second_update_parsed_preference"] = second_parsed_artifact
                execution_payload["after_second_update"] = second_execution_payload
                self._write_json(review_path, review_payload)
                self._write_json(execution_path, execution_payload)

                if second_review is not None:
                    step_status = "repaired_twice_after_violation"
                else:
                    step_status = "repaired_once_after_violation"

            manifest["steps"].append(
                {
                    "step": step_index,
                    "kind": "incremental_preference_addition",
                    "added_preference_id": preference["id"],
                    "status": step_status,
                    "review_path": str(review_path.resolve()),
                    "schedule_representation_path": str(schedule_path.resolve()),
                    "runtime_data_path": str(runtime_data_path.resolve()),
                    "constraint_blocks_path": str(constraint_blocks_path.resolve()),
                    "solver_path": str(code_path.resolve()),
                    "execution_path": str(execution_path.resolve()),
                }
            )
            active_preferences.append(preference)
            current_execution = execution

        final_schedule_json_path = self.output_dir / "final_schedule.json"
        final_schedule_text_path = self.output_dir / "final_schedule.txt"
        final_schedule_table = {}
        if current_execution.get("ok") and current_execution.get("result"):
            final_schedule_table = current_execution["result"].get("schedule_table", {})

        self._write_json(final_schedule_json_path, {"schedule_table": final_schedule_table})
        self._write_text(final_schedule_text_path, self._render_schedule_representation(final_schedule_table))
        manifest["final_schedule_json_path"] = str(final_schedule_json_path.resolve())
        manifest["final_schedule_text_path"] = str(final_schedule_text_path.resolve())

        manifest_path = self.output_dir / "manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nurse preference correction pipeline.")
    parser.add_argument(
        "--description",
        default="sythetic/1/description.txt",
        help="Path to the dataset description file. Supports both sythetic/*/description.txt and existing/*/description.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/preference_correction_run",
        help="Directory used to save generated code, reviews, and execution logs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help="Model used for initial per-constraint generation.",
    )
    parser.add_argument(
        "--review-model",
        default=None,
        help="Optional model override for review / preference parsing steps. Defaults to --model.",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Optional model override for repair-style code generation steps. Defaults to --model.",
    )
    parser.add_argument(
        "--initial-preferences-count",
        type=int,
        default=8,
        help="How many preferences are provided initially. The remaining preferences are then added one by one.",
    )
    parser.add_argument(
        "--review-format",
        choices=["json", "table", "both"],
        default="table",
        help="How the schedule is shown to the review/repair LLM: structured JSON, text table, or both as separate runs.",
    )
    parser.add_argument(
        "--review-strategy",
        choices=["direct", "targeted"],
        default="direct",
        help="`direct`: give the whole schedule to the review LLM. `targeted`: first parse each new preference, then programmatically extract the affected nurse/day cells before asking the review LLM.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="How many times to repeat the full experiment pipeline. Default is 100.",
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Wrap LLM prompts with an internal chain-of-thought style instruction while still requiring only final-format output.",
    )
    parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=1,
        help="How many independent LLM samples to draw for each generation/review call. Values >1 enable self-consistency aggregation.",
    )
    parser.add_argument(
        "--self-consistency-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used when self-consistency is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse the dataset description and print the derived experiment split.",
    )
    args = parser.parse_args()

    if args.dry_run:
        output_dir = Path(args.output_dir).resolve()
        description_path = Path(args.description).resolve()
        preview_pipeline = PreferenceCorrectionPipeline(
            description_path=description_path,
            output_dir=output_dir,
            model=args.model,
            review_model=args.review_model,
            repair_model=args.repair_model,
            initial_preferences_count=args.initial_preferences_count,
            review_format="json" if args.review_format == "both" else args.review_format,
            review_strategy=args.review_strategy,
            use_cot=args.use_cot,
            self_consistency_samples=args.self_consistency_samples,
            self_consistency_temperature=args.self_consistency_temperature,
        )
        all_preferences = preview_pipeline.problem["all_preferences"]
        initial_preferences = all_preferences[: args.initial_preferences_count]
        new_preferences = all_preferences[args.initial_preferences_count :]
        summary = {
            "description_path": str(preview_pipeline.description_path),
            "output_dir": str(output_dir),
            "hard_constraint_count": len(preview_pipeline.problem["hard_constraints"]),
            "initial_preference_ids": [item["id"] for item in initial_preferences],
            "new_preference_ids": [item["id"] for item in new_preferences],
            "initial_preferences_count": args.initial_preferences_count,
            "review_format": args.review_format,
            "review_strategy": args.review_strategy,
            "runs": args.runs,
            "generation_mode": "one-constraint-at-a-time",
            "use_cot": args.use_cot,
            "self_consistency_samples": args.self_consistency_samples,
            "self_consistency_temperature": args.self_consistency_temperature,
        }
        if args.review_format == "both":
            summary["planned_output_dirs"] = {
                "json": str((output_dir / "json_review").resolve()),
                "table": str((output_dir / "table_review").resolve()),
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.review_format == "both":
        for run_index in range(1, args.runs + 1):
            print(f"{run_index} START", flush=True)
            for review_format in ["json", "table"]:
                pipeline = PreferenceCorrectionPipeline(
                    description_path=Path(args.description).resolve(),
                    output_dir=(Path(args.output_dir).resolve() / f"run_{run_index:03d}" / f"{review_format}_review"),
                    model=args.model,
                    review_model=args.review_model,
                    repair_model=args.repair_model,
                    initial_preferences_count=args.initial_preferences_count,
                    review_format=review_format,
                    review_strategy=args.review_strategy,
                    use_cot=args.use_cot,
                    self_consistency_samples=args.self_consistency_samples,
                    self_consistency_temperature=args.self_consistency_temperature,
                )
                pipeline.run()
            print(f"{run_index} OK", flush=True)
        return

    for run_index in range(1, args.runs + 1):
        print(f"{run_index} START", flush=True)
        pipeline = PreferenceCorrectionPipeline(
            description_path=Path(args.description).resolve(),
            output_dir=Path(args.output_dir).resolve() / f"run_{run_index:03d}",
            model=args.model,
            review_model=args.review_model,
            repair_model=args.repair_model,
            initial_preferences_count=args.initial_preferences_count,
            review_format=args.review_format,
            review_strategy=args.review_strategy,
            use_cot=args.use_cot,
            self_consistency_samples=args.self_consistency_samples,
            self_consistency_temperature=args.self_consistency_temperature,
        )
        pipeline.run()
        print(f"{run_index} OK", flush=True)


if __name__ == "__main__":
    main()
