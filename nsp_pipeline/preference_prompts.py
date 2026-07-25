"""
用于“先 8 条 preference 建模，再追加 2 条 preference 并进行 3 轮检查-修正”的 prompt。
"""

CONSTRAINT_CODE_PROMPT = """
You are an expert in nurse scheduling and OR-Tools CP-SAT.

Write only the Python code snippet for ONE constraint.

Problem summary:
-----
{problem_text}
-----

Runtime input data:
-----
{runtime_data_json}
-----

Constraint kind: {constraint_kind}
Constraint id: {constraint_id}
Constraint text:
-----
{constraint_text}
-----

Current already-implemented constraints:
-----
{implemented_constraints_text}
-----

Requirements:
- Write code only for this single constraint.
- Do not write code for any other constraint.
- Do not write imports.
- Do not write `def nsp_solver(data):`.
- Do not write solver creation, solver execution, result formatting, `schedule_table`, or `return result`.
- Do not define `shifts`; it already exists.
- Assume these variables are already defined: `num_nurses`, `num_days`, `shift_types`, `work_shift_types`, `hard_constraints`, `active_preferences`, `model`, `shifts`.
- Access shift variables with tuple indexing: `shifts[(nurse, day, shift_index)]`.
- Do not use nested indexing like `shifts[nurse][day][shift_index]` unless absolutely necessary.
- Hard constraints and preferences are both encoded as hard feasibility constraints.
- Do not create penalty variables.
- Do not create an objective.
- Use direct explicit constraints that match the exact meaning of this one constraint.
- If the constraint mentions a specific nurse/day/shift, encode that exact nurse/day/shift requirement directly.
- The snippet should be self-contained for this one constraint, but should coexist with previously generated snippets.

OR-Tools CP-SAT safety rules:
- `model.AddBoolOr(...)` and `model.AddBoolAnd(...)` may only contain BoolVar literals or their negations.
- Never put linear constraints or expressions such as `x + y <= 1`, `a == b`, or sums inside `AddBoolOr(...)` or `AddBoolAnd(...)`.
- Never put comparisons such as `var == 0`, `var == 1`, `expr == 0`, or `expr == 1` inside `AddBoolOr(...)` or `AddBoolAnd(...)`.
- Never write `model.Add(var.Not())` or `model.Add(literal.Not())`.
- `model.Add(...)` must receive a real constraint such as `var == 0`, `var == 1`, `sum(...) <= k`, etc.
- If you want to represent `var == 0` inside a boolean clause, use `var.Not()`.
- If you want to represent `var == 1` inside a boolean clause, use `var` directly.
- For sliding-window or counting constraints such as “not 5 consecutive off/day/night”, prefer linear constraints like `model.Add(sum(...) <= 4)` instead of `AddBoolOr(...)`.
- If unsure, prefer `model.Add(...)` with linear inequalities.
- Correct examples:
  - `model.AddBoolOr([shifts[(n, d, off_idx)].Not(), shifts[(n, d + 1, off_idx)].Not()])`
  - `model.Add(sum(shifts[(n, start_day + i, off_idx)] for i in range(5)) <= 4)`
  - `model.Add(shifts[(n, d, night_idx)] == 0)`
- Incorrect examples:
  - `model.AddBoolOr([shifts[(n, d, off_idx)] == 0, shifts[(n, d + 1, off_idx)] == 0])`
  - `model.AddBoolOr([shifts[(n, start_day + i, day_idx)] == 0 for i in range(5)])`
  - `model.Add(shifts[(n, d, night_idx)].Not())`

Return only the Python code snippet for this single constraint inside a ```python fenced block.
"""


PREFERENCE_REVIEW_PROMPT = """
You are reviewing an existing nurse schedule after one new nurse preference was revealed.

Problem summary:
-----
{problem_text}
-----

Existing schedule representation format: {schedule_format}

Existing schedule representation:
-----
{schedule_representation}
-----

Newly added nurse preference:
-----
{new_preference_text}
-----

Task:
1. Check whether the current schedule satisfies the newly added preference.
2. If not, explain exactly where the schedule conflicts with them.
3. Propose concrete schedule correction suggestions that would improve compliance while trying to preserve feasibility.

Return exactly one JSON object with this schema:
{{
  "all_satisfied": true,
  "summary": "short summary",
  "violations": [
    {{
      "preference_id": "P9",
      "reason": "why the schedule violates or risks violating it",
      "evidence": "reference to nurse/day/shift in the table"
    }}
  ],
  "suggestions": [
    "Move Nurse 2 from night to off on Day 12 if feasible.",
    "..."
  ]
}}

Return JSON only.
"""


PREFERENCE_PARSE_PROMPT = """
You are extracting structured information from one nurse preference.

Problem summary:
-----
{problem_text}
-----

Preference to parse:
-----
{preference_id}: {preference_text}
-----

Return exactly one JSON object with this schema:
{{
  "preference_id": "P9",
  "nurse_name": "Nurse 2",
  "nurse_index_1based": 2,
  "target_days_1based": [12],
  "request_type": "require_off | avoid_shift | require_shift | prefer_shift_over_other | keep_free | multi_day_off",
  "target_shift": "day | night | off | null",
  "other_shift": "day | night | off | null",
  "notes": "short explanation of the parsed intent"
}}

Rules:
- Use 1-based day indices in `target_days_1based`.
- If the request is for rest / day off / break, set `target_shift` to `off`.
- If the request is "avoid night", use `request_type = "avoid_shift"` and `target_shift = "night"`.
- If the request says "rather work night than day", use `request_type = "prefer_shift_over_other"`, `target_shift = "night"`, `other_shift = "day"`.
- If the request covers multiple days, include all of them in `target_days_1based`.
- Return JSON only.
"""


TARGETED_PREFERENCE_REVIEW_PROMPT = """
You are checking whether one structured nurse preference is satisfied by the extracted schedule facts.

Problem summary:
-----
{problem_text}
-----

Parsed preference:
-----
{parsed_preference_json}
-----

Observed schedule facts for the affected nurse/day cells:
-----
{observed_facts_json}
-----

Task:
1. Determine whether the extracted schedule facts satisfy the parsed preference.
2. If not, explain exactly why.
3. If not satisfied, propose concrete schedule correction suggestions.

Return exactly one JSON object with this schema:
{{
  "preference_id": "P9",
  "satisfied": false,
  "reason": "short explanation",
  "evidence": [
    "Nurse 2 is assigned night on Day 12."
  ],
  "suggestions": [
    "Move Nurse 2 from night to off on Day 12 if feasible."
  ]
}}

Return JSON only.
"""


CONSTRAINT_REPAIR_PROMPT = """
You are incrementally revising the Python code snippet for ONE nurse-scheduling constraint.

Problem summary:
-----
{problem_text}
-----

Runtime input data:
-----
{runtime_data_json}
-----

Current schedule representation format: {schedule_format}

Current schedule representation:
-----
{schedule_representation}
-----

Schedule review result:
-----
{review_json}
-----

Constraint kind: {constraint_kind}
Constraint id: {constraint_id}
Constraint text:
-----
{constraint_text}
-----

Current code snippet for this one constraint:
```python
{current_constraint_code}
```

Task:
- Revise only this one constraint snippet.
- Make the minimum necessary change so the model better satisfies this constraint while preserving feasibility.
- Do not rewrite unrelated constraints.
- Do not write imports.
- Do not write `def nsp_solver(data):`.
- Do not write solver creation, solver execution, result formatting, `schedule_table`, or `return result`.
- Do not define `shifts`; it already exists.
- Assume these variables are already defined: `num_nurses`, `num_days`, `shift_types`, `work_shift_types`, `hard_constraints`, `active_preferences`, `model`, `shifts`.
- Access shift variables with tuple indexing: `shifts[(nurse, day, shift_index)]`.
- Do not use nested indexing like `shifts[nurse][day][shift_index]` unless absolutely necessary.
- Keep this a hard feasibility constraint.
- Do not create penalty variables.
- Do not create an objective.
- Return only the revised snippet for this one constraint.

OR-Tools CP-SAT safety rules:
- `model.AddBoolOr(...)` and `model.AddBoolAnd(...)` may only contain BoolVar literals or their negations.
- Never put linear constraints or expressions such as `x + y <= 1`, `a == b`, or sums inside `AddBoolOr(...)` or `AddBoolAnd(...)`.
- Never put comparisons such as `var == 0`, `var == 1`, `expr == 0`, or `expr == 1` inside `AddBoolOr(...)` or `AddBoolAnd(...)`.
- Never write `model.Add(var.Not())` or `model.Add(literal.Not())`.
- `model.Add(...)` must receive a real constraint such as `var == 0`, `var == 1`, `sum(...) <= k`, etc.
- If you want to represent `var == 0` inside a boolean clause, use `var.Not()`.
- If you want to represent `var == 1` inside a boolean clause, use `var` directly.
- For sliding-window or counting constraints such as “not 5 consecutive off/day/night”, prefer linear constraints like `model.Add(sum(...) <= 4)` instead of `AddBoolOr(...)`.
- If unsure, prefer `model.Add(...)` with linear inequalities.
- Correct examples:
  - `model.AddBoolOr([shifts[(n, d, off_idx)].Not(), shifts[(n, d + 1, off_idx)].Not()])`
  - `model.Add(sum(shifts[(n, start_day + i, off_idx)] for i in range(5)) <= 4)`
  - `model.Add(shifts[(n, d, night_idx)] == 0)`
- Incorrect examples:
  - `model.AddBoolOr([shifts[(n, d, off_idx)] == 0, shifts[(n, d + 1, off_idx)] == 0])`
  - `model.AddBoolOr([shifts[(n, start_day + i, day_idx)] == 0 for i in range(5)])`
  - `model.Add(shifts[(n, d, night_idx)].Not())`

Return only the revised Python code snippet for this single constraint inside a ```python fenced block.
"""
