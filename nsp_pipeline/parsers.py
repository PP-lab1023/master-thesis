import json
from typing import Any, Dict, List


def format_constraints(constraints: List[Dict[str, Any]]) -> str:
    """
    把约束列表格式化成 prompt 里更容易阅读的多行文本。
    """
    return "\n".join(f'{item["id"]}. {item["text"]}' for item in constraints)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    从 LLM 返回的文本里提取一个完整 JSON 对象。
    """
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:index + 1])
    raise ValueError("Incomplete JSON object in LLM response.")


def extract_python_code(text: str) -> str:
    """
    从 LLM 输出中提取 Python 代码。
    """
    candidate = text.strip()
    if "```python" in candidate:
        return candidate.split("```python", 1)[1].split("```", 1)[0].strip() + "\n"
    if "```" in candidate:
        return candidate.split("```", 1)[1].split("```", 1)[0].strip() + "\n"
    return candidate + ("\n" if not candidate.endswith("\n") else "")

