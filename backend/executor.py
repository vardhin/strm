"""
Program executor for NSRR.

Runs composed programs described by a candidate dict against inputs.
Validates candidates against input/output examples.

A candidate is:
    {
        "primary_id":   int,
        "secondary_id": int | None,
        "tertiary_id":  int | None,
        "comp_type":    "none" | "sequential" | "nested" | "parallel" | "loop_direct" | "loop_binary",
    }
"""

import math
from typing import Any

import registry as reg


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run(state: dict, candidate: dict, inputs: list[int]) -> Any:
    """Execute a candidate program on the given inputs."""
    result = _run_base(state, candidate, inputs)

    # Apply fitted constants if present
    constants = candidate.get("constants")
    if constants:
        if candidate.get("const_mode") == "additive":
            result = result + constants[0]
        else:
            result = result * constants[0]

    return result


def _run_base(state: dict, candidate: dict, inputs: list) -> Any:
    """Execute the base composition without constant adjustment."""
    comp = candidate["comp_type"]
    pid = candidate["primary_id"]

    if comp == "none":
        routing = candidate.get("routing")
        if routing:
            inputs = [inputs[i] for i in routing[0]]
        return reg.execute(state, pid, inputs)

    sid = candidate["secondary_id"]

    if comp == "sequential":
        routing = candidate.get("routing")
        if routing:
            inp_s = [inputs[i] for i in routing[0]]
        else:
            inp_s = inputs
        intermediate = reg.execute(state, sid, inp_s)
        return reg.execute(state, pid, [intermediate])

    if comp == "nested":
        # primary(secondary(x), secondary(y), ...)
        routing = candidate.get("routing")
        if routing:
            items = [inputs[i] for i in routing[0]]
        else:
            items = inputs
        transformed = [reg.execute(state, sid, [x]) for x in items]
        return reg.execute(state, pid, transformed)

    if comp == "parallel":
        # tertiary(primary(routed_inputs), secondary(routed_inputs))
        tid = candidate.get("tertiary_id")
        routing = candidate.get("routing")
        if routing:
            inp1 = [inputs[i] for i in routing[0]]
            inp2 = [inputs[i] for i in routing[1]]
        else:
            inp1 = inputs
            inp2 = inputs
        r1 = reg.execute(state, pid, inp1)
        r2 = reg.execute(state, sid, inp2)
        if tid is not None:
            return reg.execute(state, tid, [r1, r2])
        return r1

    if comp == "loop_direct":
        # LOOP(body_fn=sid, count=inputs[1], init=inputs[0])
        loop_id = state["loop_id"]
        return reg.execute(state, loop_id, [sid, inputs[1], inputs[0]])

    if comp == "loop_binary":
        # LOOP(body_fn=sid, count=inputs[1], init=0, step_arg=inputs[0])
        # For MUL: ADD(accum, a) repeated b times, starting from 0
        loop_id = state["loop_id"]
        return reg.execute(state, loop_id, [sid, inputs[1], 0, inputs[0]])

    raise ValueError(f"Unknown composition type: {comp}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(state: dict, candidate: dict,
             examples: list[tuple[list, Any]]) -> bool:
    """Check if a candidate produces the correct output for every example.

    Uses math.isclose for float comparison (rel_tol=1e-6, abs_tol=1e-9).
    """
    for inputs, expected in examples:
        try:
            got = run(state, candidate, inputs)
            if not math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9):
                return False
        except Exception:
            return False
    return True


def r_squared(state: dict, func_id: int,
              examples: list[tuple[list, Any]]) -> float:
    """Compute R² score for a registered function against examples."""
    actuals = []
    predictions = []
    for inputs, expected in examples:
        try:
            got = reg.execute(state, func_id, inputs)
            actuals.append(float(expected))
            predictions.append(float(got))
        except Exception:
            return -float("inf")

    if not actuals:
        return -float("inf")

    mean_actual = sum(actuals) / len(actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -float("inf")

    return 1.0 - ss_res / ss_tot
