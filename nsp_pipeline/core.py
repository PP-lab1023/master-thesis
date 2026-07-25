import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nsp_pipeline.data_contract import load_experiment_data
from nsp_pipeline.llm import DEFAULT_OPENROUTER_MODEL, apply_cot_prompt, request_llm
from nsp_pipeline.parsers import extract_json_object, extract_python_code, format_constraints
from nsp_pipeline.prompts import (
    INCREMENTAL_CODE_PROMPT,
    INCREMENTAL_EXTRACTION_PROMPT,
    INITIAL_CODE_PROMPT,
    INITIAL_EXTRACTION_PROMPT,
)
from nsp_pipeline.runner import execute_solver


class NSPPipeline:
    """
    整个实验 pipeline 的核心类。
    """

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        model: str = DEFAULT_OPENROUTER_MODEL,
        retries: int = 3,
        use_cot: bool = False,
        self_consistency_samples: int = 1,
        self_consistency_temperature: float = 0.7,
    ):
        """
        初始化 pipeline,并在一开始完成配置检查。
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.model = model
        self.retries = retries
        self.use_cot = use_cot
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.self_consistency_temperature = self_consistency_temperature
        self.config = self._load_config()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """
        读取并校验实验配置文件。
        """
        with self.config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        constraints = config.get("constraints", [])
        if len(constraints) != 10:
            raise ValueError(f"Expected exactly 10 constraints, found {len(constraints)}.")

        base_count = int(config.get("base_constraints_count", 4))
        if base_count != 4:
            raise ValueError("This experiment pipeline expects the first 4 constraints to be the base constraints.")

        required = {"problem_description", "data_path", "constraints"}
        missing = required - config.keys()
        if missing:
            raise ValueError(f"Missing required config fields: {sorted(missing)}")

        return config

    def _prepare_prompt(self, prompt: str) -> str:
        return apply_cot_prompt(prompt) if self.use_cot else prompt

    def _llm_temperature(self, sample_index: int) -> float:
        if self.self_consistency_samples <= 1:
            return 0.0
        return self.self_consistency_temperature

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        """
        调用 LLM,并把返回结果解析为 JSON。
        """
        prepared_prompt = self._prepare_prompt(prompt)
        if self.self_consistency_samples <= 1:
            last_error = None
            for _ in range(self.retries):
                try:
                    response = request_llm(prepared_prompt, model=self.model, temperature=0.0)
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
                    model=self.model,
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
        validation_data: Optional[Dict[str, Any]],
        active_constraints: Optional[List[str]],
    ) -> Tuple[int, int]:
        if validation_data is None or active_constraints is None:
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
            execution = execute_solver(temp_path, validation_data, active_constraints)
            return (1 if execution.get("ok") else 0, 1 if execution.get("result") else 0)
        except Exception:
            return (0, 0)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _call_llm_code(
        self,
        prompt: str,
        validation_data: Optional[Dict[str, Any]] = None,
        active_constraints: Optional[List[str]] = None,
    ) -> str:
        """
        调用 LLM,并把返回结果解析为 Python 代码。
        """
        prepared_prompt = self._prepare_prompt(prompt)
        if self.self_consistency_samples <= 1:
            last_error = None
            for _ in range(self.retries):
                try:
                    response = request_llm(prepared_prompt, model=self.model, temperature=0.0)
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
                    model=self.model,
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
            execution_score = self._score_code_candidate(code, validation_data, active_constraints)
            frequency = candidate_counts[code.strip()]
            score = (execution_score[0], execution_score[1], frequency)
            if score > best_score:
                best_score = score
                best_code = code
        return best_code or code_candidates[0]

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        """
        把字典写成格式化 JSON 文件。
        """
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _save_solver_code(self, step_index: int, label: str, code: str) -> Path:
        """
        保存某一步生成的 solver 代码。
        """
        path = self.output_dir / f"step_{step_index:02d}_{label}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def _load_data(self) -> Dict[str, Any]:
        """
        读取实验输入数据。

        注意:
        - 磁盘上的 JSON 使用新的分层实验格式
        - 这里会把它转换成 solver 内部更稳定的扁平字段
        """
        data_path = (self.config_path.parent / self.config["data_path"]).resolve()
        return load_experiment_data(data_path)

    def run(self) -> Dict[str, Any]:
        """
        执行整条实验主流程。
        """
        problem_description = self.config["problem_description"]
        constraints = self.config["constraints"]
        base_constraints = constraints[:4]
        incremental_constraints = constraints[4:]
        data = self._load_data()

        manifest: Dict[str, Any] = {
            "config_path": str(self.config_path.resolve()),
            "model": self.model,
            "use_cot": self.use_cot,
            "self_consistency_samples": self.self_consistency_samples,
            "self_consistency_temperature": self.self_consistency_temperature,
            "steps": [],
        }

        schema = self._call_llm_json(
            INITIAL_EXTRACTION_PROMPT.format(
                problem_description=problem_description,
                constraints_text=format_constraints(base_constraints),
            )
        )
        self._write_json(self.output_dir / "step_00_schema.json", schema)

        solver_code = self._call_llm_code(
            INITIAL_CODE_PROMPT.format(
                problem_description=problem_description,
                constraints_text=format_constraints(base_constraints),
                schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
            ),
            validation_data=data,
            active_constraints=[item["text"] for item in base_constraints],
        )
        solver_path = self._save_solver_code(0, "base_solver", solver_code)
        execution = execute_solver(solver_path, data, [item["text"] for item in base_constraints])
        self._write_json(self.output_dir / "step_00_execution.json", execution)
        manifest["steps"].append(
            {
                "step": 0,
                "constraint_id": None,
                "schema_path": str((self.output_dir / "step_00_schema.json").resolve()),
                "solver_path": str(solver_path.resolve()),
                "execution_path": str((self.output_dir / "step_00_execution.json").resolve()),
            }
        )

        active_constraints = list(base_constraints)
        for offset, new_constraint in enumerate(incremental_constraints, start=1):
            active_constraints.append(new_constraint)
            schema_update = self._call_llm_json(
                INCREMENTAL_EXTRACTION_PROMPT.format(
                    problem_description=problem_description,
                    current_schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
                    new_constraint=new_constraint["text"],
                )
            )
            schema = {
                "Parameters": schema_update.get("Parameters", {}),
                "Variables": schema_update.get("Variables", {}),
            }
            step_schema_path = self.output_dir / f"step_{offset:02d}_{new_constraint['id']}_schema.json"
            self._write_json(step_schema_path, schema_update)

            solver_code = self._call_llm_code(
                INCREMENTAL_CODE_PROMPT.format(
                    problem_description=problem_description,
                    constraints_text=format_constraints(active_constraints),
                    schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
                    new_constraint=new_constraint["text"],
                    current_solver_code=solver_code,
                ),
                validation_data=data,
                active_constraints=[item["text"] for item in active_constraints],
            )
            solver_path = self._save_solver_code(offset, new_constraint["id"], solver_code)
            execution = execute_solver(solver_path, data, [item["text"] for item in active_constraints])
            execution_path = self.output_dir / f"step_{offset:02d}_{new_constraint['id']}_execution.json"
            self._write_json(execution_path, execution)
            manifest["steps"].append(
                {
                    "step": offset,
                    "constraint_id": new_constraint["id"],
                    "schema_path": str(step_schema_path.resolve()),
                    "solver_path": str(solver_path.resolve()),
                    "execution_path": str(execution_path.resolve()),
                }
            )

        manifest_path = self.output_dir / "manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest
