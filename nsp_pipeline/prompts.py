"""
集中存放所有给 LLM 的 prompt 模板。

这样做的好处是:
- prompt 和业务逻辑解耦
- 以后只想改提示词时，不需要翻主流程代码
"""

INITIAL_EXTRACTION_PROMPT = """
You are an expert in the Nurse Scheduling Problem (NSP) and mathematical optimization.

Task:
You are given:
1. A problem description.
2. The current active constraints.

You must extract the optimization model's parameters and decision variables that are needed for the current stage only.

Problem description:
-----
{problem_description}
-----

Current active constraints:
-----
{constraints_text}
-----

Output format:
Return exactly one JSON object with this schema:
{{
  "Parameters": {{
    "Symbol": {{
      "shape": "[] or [N,D,T] style",
      "type": "int | float | binary | bool",
      "definition": "concise meaning"
    }}
  }},
  "Variables": {{
    "Symbol": {{
      "shape": "[] or [N,D,T] style",
      "type": "int | float | binary | bool",
      "definition": "concise meaning"
    }}
  }}
}}

Rules:
- Include core size parameters when needed, such as N, D, T.
- Include only information needed for the current active constraints.
- If a soft constraint needs auxiliary variables for penalties, include them.
- Use CamelCase symbols.
- Return JSON only.
"""


INCREMENTAL_EXTRACTION_PROMPT = """
You are updating an NSP optimization model after one new constraint is revealed.

Problem description:
-----
{problem_description}
-----

Current parameters and variables:
-----
{current_schema_json}
-----

New constraint:
-----
{new_constraint}
-----

Determine whether the new constraint requires new parameters or variables.

Return exactly one JSON object with this schema:
{{
  "analysis": {{
    "needs_new_parameters": true,
    "needs_new_variables": false,
    "reason": "short explanation"
  }},
  "Parameters": {{
    "Symbol": {{
      "shape": "[] or [N,D,T] style",
      "type": "int | float | binary | bool",
      "definition": "concise meaning"
    }}
  }},
  "Variables": {{
    "Symbol": {{
      "shape": "[] or [N,D,T] style",
      "type": "int | float | binary | bool",
      "definition": "concise meaning"
    }}
  }}
}}

Rules:
- Reuse existing symbols whenever possible.
- Keep all still-needed parameters and variables in the returned schema.
- Add auxiliary variables if the new constraint is soft or needs a linearization helper.
- Return JSON only.
"""


INITIAL_CODE_PROMPT = """
You are an expert in OR-Tools CP-SAT and NSP modeling.

Write a complete, self-contained Python solver file for the current NSP stage.

Problem description:
-----
{problem_description}
-----

Current active constraints:
-----
{constraints_text}
-----

Current parameters and variables:
-----
{schema_json}
-----

Requirements:
- Use `from ortools.sat.python import cp_model`.
- Implement `def nsp_solver(data):`.
- `data` is the runtime input dict already loaded by the pipeline.
- Do not read files inside `nsp_solver`.
- Build the model for the current active constraints only.
- The runtime `data` is a FLAT dictionary, not the schema JSON.
- Read fields directly from `data`, for example:
  - `N = data["N"]`
  - `D = data["D"]`
  - `T = data["T"]`
  - `availableSlots = data["availableSlots"]`
  - `demandSlot = data["demandSlot"]`
  - `MinDays = data["MinDays"]`
  - `MaxDays = data["MaxDays"]`
  - `requestOn = data["requestOn"]`
  - `requestOff = data["requestOff"]`
- Do NOT use nested access like `data["Parameters"][...]` or `data["Variables"][...]`.
- The `schema_json` shown above is only a modeling reference for you, not the runtime layout of `data`.
- Return a JSON-serializable dict with at least:
  {{
    "status": "OPTIMAL | FEASIBLE | INFEASIBLE | MODEL_INVALID | UNKNOWN",
    "objective": number or null,
    "selected_constraints": [list of plain-text constraints used in this stage]
  }}
- If you create assignment outputs, keep them JSON-serializable.
- Keep the code concise and runnable.

Return format:
```python
# full file here
```
"""


INCREMENTAL_CODE_PROMPT = """
You are incrementally editing an existing OR-Tools NSP solver.

Your goal is to minimally modify the current solver so that it supports one new constraint.

Problem description:
-----
{problem_description}
-----

All active constraints after adding the new one:
-----
{constraints_text}
-----

Updated parameters and variables:
-----
{schema_json}
-----

New constraint to add now:
-----
{new_constraint}
-----

Current solver code:
```python
{current_solver_code}
```

Editing rules:
- Keep the existing structure and names whenever possible.
- Add only the smallest necessary changes for the new constraint.
- Do not redesign the whole solver.
- Preserve `def nsp_solver(data):`.
- Keep the return value JSON-serializable.
- The final answer must still be a complete runnable Python file.
- The runtime `data` remains a FLAT dictionary.
- Read fields directly from `data`, such as `data["N"]`, `data["MaxDays"]`, `data["requestOn"]`.
- Do NOT introduce or keep accesses like `data["Parameters"][...]` or `data["Variables"][...]`.
- The `schema_json` above is only a modeling reference, not the runtime structure of `data`.

Return format:
```python
# updated full file here
```
"""
