import contextlib
import importlib.util
import io
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def execute_solver(code_path: Path, data: Dict[str, Any], active_constraints: List[str]) -> Dict[str, Any]:
    """
    动态导入生成出来的 solver 文件，并执行其中的 `nsp_solver(data)`。
    """
    module_name = f"generated_nsp_solver_{int(time.time() * 1000)}"
    stdout_buffer = io.StringIO()
    start_time = time.time()
    try:
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import generated solver from {code_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with contextlib.redirect_stdout(stdout_buffer):
            spec.loader.exec_module(module)
            if not hasattr(module, "nsp_solver"):
                raise AttributeError("Generated solver file does not define nsp_solver(data).")
            raw_result = module.nsp_solver(deepcopy(data))

        elapsed = time.time() - start_time
        return {
            "ok": True,
            "runtime_seconds": round(elapsed, 4),
            "stdout": stdout_buffer.getvalue(),
            "result": raw_result,
            "active_constraints": active_constraints,
        }
    except Exception:
        elapsed = time.time() - start_time
        return {
            "ok": False,
            "runtime_seconds": round(elapsed, 4),
            "stdout": stdout_buffer.getvalue(),
            "error": traceback.format_exc(),
            "active_constraints": active_constraints,
        }
    finally:
        sys.modules.pop(module_name, None)

