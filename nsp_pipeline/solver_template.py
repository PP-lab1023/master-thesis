from textwrap import indent


FIXED_SOLVER_TEMPLATE = """from ortools.sat.python import cp_model


class _ShiftAccessor:
    def __init__(self, flat_vars, prefix=()):
        self._flat_vars = flat_vars
        self._prefix = prefix

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._flat_vars[key]
        new_prefix = self._prefix + (key,)
        if len(new_prefix) == 3:
            return self._flat_vars[new_prefix]
        return _ShiftAccessor(self._flat_vars, new_prefix)


def _build_schedule_table(solver, shifts, num_nurses, num_days, shift_types):
    table = {{
        "days": [f"Day {{day + 1}}" for day in range(num_days)],
        "nurses": [f"Nurse {{nurse + 1}}" for nurse in range(num_nurses)],
        "rows": [],
    }}
    for nurse in range(num_nurses):
        row = []
        for day in range(num_days):
            assigned_shift = "unassigned"
            for shift_index, shift_name in enumerate(shift_types):
                if solver.Value(shifts[(nurse, day, shift_index)]) == 1:
                    assigned_shift = shift_name
                    break
            row.append(assigned_shift)
        table["rows"].append(row)
    return table


def nsp_solver(data):
    num_nurses = data["NumberOfNurses"]
    num_days = data["NumberOfDays"]
    shift_types = data["ShiftTypes"]
    work_shift_types = data["WorkShiftTypes"]
    hard_constraints = data["HardConstraints"]
    active_preferences = data["ActivePreferences"]

    model = cp_model.CpModel()

    shift_vars = {{}}
    for nurse in range(num_nurses):
        for day in range(num_days):
            for shift_index, shift_name in enumerate(shift_types):
                shift_vars[(nurse, day, shift_index)] = model.NewBoolVar(
                    f"shift_n{{nurse}}_d{{day}}_s{{shift_name}}"
                )
    shifts = _ShiftAccessor(shift_vars)

    # Every nurse must receive exactly one state (day / night / off) per day.
    for nurse in range(num_nurses):
        for day in range(num_days):
            model.Add(
                sum(shift_vars[(nurse, day, shift_index)] for shift_index in range(len(shift_types))) == 1
            )

{constraint_code}

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)

    result = {{
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "schedule_table": {{
            "days": [f"Day {{day + 1}}" for day in range(num_days)],
            "nurses": [f"Nurse {{nurse + 1}}" for nurse in range(num_nurses)],
            "rows": [],
        }},
        "selected_hard_constraints": hard_constraints,
        "selected_preferences": [item["text"] for item in active_preferences],
    }}

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["schedule_table"] = _build_schedule_table(
            solver=solver,
            shifts=shift_vars,
            num_nurses=num_nurses,
            num_days=num_days,
            shift_types=shift_types,
        )

    return result
"""


def assemble_solver_code(constraint_blocks: list[str]) -> str:
    """
    将逐条生成的 constraint 代码片段嵌入固定 solver 模板。
    """
    normalized_blocks = [block.strip("\n") for block in constraint_blocks if block.strip()]
    merged = "\n\n".join(normalized_blocks)
    return FIXED_SOLVER_TEMPLATE.format(constraint_code=indent(merged, "    ")) + "\n"
