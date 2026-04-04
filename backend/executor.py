"""
Program executor for NSSR.

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

from typing import Any

import registry as reg


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run(state: dict, candidate: dict, inputs: list[int]) -> Any:
    """Execute a candidate program on the given inputs."""
    comp = candidate["comp_type"]
    pid = candidate["primary_id"]

    if comp == "none":
        return reg.execute(state, pid, inputs)

    sid = candidate["secondary_id"]

    if comp == "sequential":
        intermediate = reg.execute(state, sid, inputs)
        return reg.execute(state, pid, [intermediate])

    if comp == "nested":
        # primary(secondary(x), secondary(y), ...)
        transformed = [reg.execute(state, sid, [x]) for x in inputs]
        return reg.execute(state, pid, transformed)

    if comp == "parallel":
        # tertiary(primary(inputs), secondary(inputs))
        tid = candidate.get("tertiary_id")
        r1 = reg.execute(state, pid, inputs)
        r2 = reg.execute(state, sid, inputs)
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
             examples: list[tuple[list[int], Any]]) -> bool:
    """Check if a candidate produces the correct output for every example."""
    for inputs, expected in examples:
        try:
            if run(state, candidate, inputs) != expected:
                return False
        except Exception:
            return False
    return True
