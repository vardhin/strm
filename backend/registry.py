"""
Function registry for NSSR.

Manages primitives and learned (composed) functions.
State is a plain dict passed to every function — no globals, no classes.

Registry state shape:
    {
        "functions":  {int: callable},   # id -> fn(inputs) -> result
        "metadata":   {int: dict},       # id -> {name, arity, layer}
        "next_id":    int,               # next available function id
        "loop_id":    int | None,        # cached id of LOOP primitive
    }
"""

import sqlite3
from typing import Any

import db


# ---------------------------------------------------------------------------
# Primitive definitions
# ---------------------------------------------------------------------------
# Each primitive is (arity, factory).
# factory(execute_fn) -> callable(inputs) -> result
#
# The indirection through execute_fn lets LOOP/WHILE/ACCUM call back into
# the registry without circular imports or self-references.
# arity=-1 means variadic.
# ---------------------------------------------------------------------------

def _make_primitives() -> dict[str, tuple[int, Any]]:
    """Return {name: (arity, factory)} for all built-in primitives."""

    def _factory(fn):
        """Wrap a simple function that doesn't need execute_fn."""
        return lambda _exec: fn

    return {
        # -- Bitwise --
        "OR":    (2, _factory(lambda inp: inp[0] | inp[1])),
        "AND":   (2, _factory(lambda inp: inp[0] & inp[1])),
        "NOT":   (1, _factory(lambda inp: ~inp[0])),

        # -- Arithmetic --
        "INC":   (1, _factory(lambda inp: inp[0] + 1)),
        "DEC":   (1, _factory(lambda inp: inp[0] - 1)),

        # -- Division (float) --
        "DIV":   (2, _factory(lambda inp: inp[0] / inp[1] if inp[1] != 0 else 0.0)),

        # -- Comparison --
        "LT":    (2, _factory(lambda inp: int(inp[0] < inp[1]))),
        "LTE":   (2, _factory(lambda inp: int(inp[0] <= inp[1]))),
        "GT":    (2, _factory(lambda inp: int(inp[0] > inp[1]))),
        "GTE":   (2, _factory(lambda inp: int(inp[0] >= inp[1]))),
        "EQ":    (2, _factory(lambda inp: int(inp[0] == inp[1]))),
        "NEQ":   (2, _factory(lambda inp: int(inp[0] != inp[1]))),

        # -- Control --
        "COND":  (3, _factory(lambda inp: inp[1] if inp[0] != 0 else inp[2])),
        "CONST": (1, _factory(lambda inp: inp[0])),

        # -- Iteration (these need execute_fn) --
        "LOOP":  (-1, _loop_factory),
        "WHILE": (-1, _while_factory),
        "ACCUM": (-1, _accum_factory),
    }


def _loop_factory(execute_fn):
    """LOOP(body_fn_id, count, init_value, [step_arg]) -> apply body_fn `count` times.

    3 args: unary body  — result = body(result)           each iteration
    4 args: binary body — result = body(result, step_arg) each iteration

    This lets LOOP handle both ADD (unary INC) and MUL (binary ADD):
      ADD(a,b) = LOOP(INC, b, a)        — 3 args, unary
      MUL(a,b) = LOOP(ADD, b, 0, a)     — 4 args, binary: ADD(accum, a) repeated b times
    """
    def loop_impl(inputs):
        if len(inputs) < 3:
            raise ValueError(f"LOOP expects 3-4 args [body_fn_id, count, init, (step_arg)], got {len(inputs)}")
        body_fn_id, count, init_value = int(inputs[0]), inputs[1], inputs[2]
        step_arg = inputs[3] if len(inputs) >= 4 else None
        result = init_value
        for _ in range(int(count)):
            if step_arg is not None:
                result = execute_fn(body_fn_id, [result, step_arg])
            else:
                result = execute_fn(body_fn_id, [result])
        return result
    return loop_impl


def _while_factory(execute_fn):
    """WHILE(cond_fn_id, body_fn_id, state, limit) -> final state."""
    def while_impl(inputs):
        if len(inputs) != 4:
            raise ValueError(f"WHILE expects 4 args [cond_fn, body_fn, state, limit], got {len(inputs)}")
        cond_fn_id, body_fn_id, state, limit = int(inputs[0]), int(inputs[1]), inputs[2], inputs[3]
        for _ in range(int(limit)):
            if execute_fn(cond_fn_id, [state]) == 0:
                break
            state = execute_fn(body_fn_id, [state])
        return state
    return while_impl


def _accum_factory(execute_fn):
    """ACCUM(cond_fn_id, body_fn_id, state, counter, limit) -> counter."""
    def accum_impl(inputs):
        if len(inputs) != 5:
            raise ValueError(f"ACCUM expects 5 args [cond_fn, body_fn, state, counter, limit], got {len(inputs)}")
        cond_fn_id, body_fn_id = int(inputs[0]), int(inputs[1])
        state, counter, limit = inputs[2], inputs[3], inputs[4]
        for _ in range(int(limit)):
            if execute_fn(cond_fn_id, [state]) == 0:
                break
            state = execute_fn(body_fn_id, [state])
            counter += 1
        return counter
    return accum_impl


PRIMITIVES = _make_primitives()


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    return {"functions": {}, "metadata": {}, "next_id": 0, "loop_id": None}


def execute(state: dict, func_id: int, inputs: list) -> Any:
    """Execute a registered function by id."""
    fn = state["functions"].get(func_id)
    if fn is None:
        raise ValueError(f"Function id {func_id} not found in registry")
    return fn(inputs)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_registry(conn: sqlite3.Connection) -> dict:
    """Register all primitives into the DB and return a fresh registry state."""
    state = _empty_state()

    # Closure so LOOP/WHILE/ACCUM can call back into execute()
    def _exec(fid, inp):
        return execute(state, fid, inp)

    for name, (arity, factory) in PRIMITIVES.items():
        fid = state["next_id"]
        state["functions"][fid] = factory(_exec)
        state["metadata"][fid] = {"name": name, "arity": arity, "layer": 0}
        db.add_primitive(conn, fid, name, arity)
        if name == "LOOP":
            state["loop_id"] = fid
        state["next_id"] += 1

    return state


def load_registry(conn: sqlite3.Connection) -> dict:
    """Rebuild registry state from an existing database."""
    state = _empty_state()

    def _exec(fid, inp):
        return execute(state, fid, inp)

    all_funcs = db.get_all_functions(conn)

    for row in all_funcs:
        fid = row["id"]
        name = row["name"]
        arity = row["arity"]
        layer = row["layer"]

        state["metadata"][fid] = {"name": name, "arity": arity, "layer": layer}

        if layer == 0:
            # Primitive — look up its factory
            prim = PRIMITIVES.get(name)
            if prim is None:
                raise ValueError(f"Unknown primitive in DB: {name}")
            _, factory = prim
            state["functions"][fid] = factory(_exec)
            if name == "LOOP":
                state["loop_id"] = fid
        else:
            # Learned — build a composition closure
            composition = db.get_composition(conn, fid)
            constants = db.get_constants(conn, fid)
            const_mode = db.get_const_mode(conn, fid)
            state["functions"][fid] = _make_composed_fn(state, composition, _exec,
                                                        constants, const_mode)
            state["metadata"][fid]["constants"] = constants
            state["metadata"][fid]["const_mode"] = const_mode

        state["next_id"] = max(state["next_id"], fid + 1)

    return state


# ---------------------------------------------------------------------------
# Learning new functions
# ---------------------------------------------------------------------------

def register_learned(conn: sqlite3.Connection, state: dict, name: str,
                     arity: int, composition: list[tuple[int, list[int]]],
                     constants: list[float] | None = None,
                     const_mode: str = "multiplicative") -> int:
    """Register a new composed function. Returns the new function id."""
    # Check for duplicate name
    for fid, meta in state["metadata"].items():
        if meta["name"] == name:
            return fid

    def _exec(fid, inp):
        return execute(state, fid, inp)

    fid = state["next_id"]
    layer = db.add_learned(conn, fid, name, arity, composition, constants,
                           const_mode)

    state["functions"][fid] = _make_composed_fn(state, composition, _exec,
                                                constants, const_mode)
    state["metadata"][fid] = {"name": name, "arity": arity, "layer": layer,
                              "constants": constants, "const_mode": const_mode}
    state["next_id"] = fid + 1

    return fid


# ---------------------------------------------------------------------------
# Composition execution
# ---------------------------------------------------------------------------

def _make_composed_fn(state: dict, composition: list[tuple[int, list[int]]],
                      execute_fn, constants: list[float] | None = None,
                      const_mode: str = "multiplicative"):
    """Build a callable that runs a composition step-by-step.

    Composition is a list of (child_func_id, arg_indices).
    arg_indices index into an `available_values` list that starts with the
    original inputs and grows as each step appends its result.

    Special arg index -1 means literal 0 (used by LOOP for MUL).

    If constants is provided, a final scaling/offset is applied:
      multiplicative: result *= constants[0]
      additive:       result += constants[0]
    """
    loop_id = state["loop_id"]

    def composed(inputs):
        available = list(inputs)

        for child_id, args in composition:
            if child_id == loop_id and len(args) in (3, 4):
                body_fn_id = args[0]
                count = available[args[1]]
                init_val = 0 if args[2] == -1 else available[args[2]]
                loop_args = [body_fn_id, count, init_val]
                if len(args) == 4:
                    loop_args.append(available[args[3]])
                result = execute_fn(child_id, loop_args)
            else:
                step_inputs = [available[i] for i in args]
                result = execute_fn(child_id, step_inputs)

            available.append(result)

        result = available[-1]

        if constants:
            if const_mode == "additive":
                result = result + constants[0]
            else:
                result = result * constants[0]

        return result

    return composed


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def vocab_size(state: dict) -> int:
    """Number of registered functions (needed by the model for head sizing)."""
    return len(state["functions"])


def get_name(state: dict, func_id: int) -> str:
    """Get function name by id."""
    return state["metadata"][func_id]["name"]


def get_names(state: dict, func_ids: list[int]) -> list[str]:
    """Get multiple function names."""
    return [state["metadata"][fid]["name"] for fid in func_ids]
