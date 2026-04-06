"""
NSSR — Neuro-Symbolic Recursive Regression

Single entry point. Runs the full learning pipeline:
  1. Initialize registry with primitives
  2. Pre-train TRM on curriculum (identity tasks)
  3. Progressively learn higher-level functions:
     Logic  -> NAND, XOR
     Arith  -> ADD, SUB, MUL
"""

import os
import torch
import sqlite3

import db
import registry as reg
import executor as exe
import search
import train
import simplify
from model import TRM, create_model, fresh_carry, resize_heads


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "input_dim": 32,
    "seq_len": 8,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 2,          # tiny network — "less is more"
    "n_recursions": 6,      # n latent z-refinement steps per recursion
    "T": 3,                 # T-1 no-grad + 1 grad pass → effective depth = T*(n+1)*n_layers = 42
    "n_sup": 16,            # deep supervision steps per training example
    "dropout": 0.1,
    "lr": 1e-4,             # paper uses 1e-4 with AdamW
    "checkpoint_dir": "checkpoints",
}


# ---------------------------------------------------------------------------
# Candidate -> Composition conversion
# ---------------------------------------------------------------------------

def build_composition(candidate: dict, input_arity: int,
                      loop_id: int | None) -> list[tuple[int, list[int]]]:
    """Convert a search candidate to a storable composition list.

    A composition is [(func_id, [arg_indices]), ...] where arg_indices
    reference positions in an `available_values` list (inputs ++ prior results).

    Routing: candidate["routing"] is a list of column-index lists.
    For non-parallel types, routing[0] selects which input columns to use.
    For parallel, routing[0] and routing[1] select columns for each branch.
    """
    comp_type = candidate["comp_type"]
    pid = candidate["primary_id"]
    sid = candidate.get("secondary_id")
    tid = candidate.get("tertiary_id")
    routing = candidate.get("routing")

    if comp_type == "none":
        cols = routing[0] if routing else list(range(input_arity))
        return [(pid, cols)]

    if comp_type == "loop_direct":
        # LOOP(body_fn=sid, count=input[1], init=input[0])
        return [(pid, [sid, 1, 0])]

    if comp_type == "loop_binary":
        # MUL(a,b) = LOOP(ADD, count=b, init=0, step_arg=a)
        return [(pid, [sid, 1, -1, 0])]

    if comp_type == "sequential":
        # secondary(routed_inputs) -> primary(result)
        cols = routing[0] if routing else list(range(input_arity))
        return [
            (sid, cols),
            (pid, [input_arity]),
        ]

    if comp_type == "nested":
        # primary(secondary(x_i), secondary(x_j), ...)
        cols = routing[0] if routing else list(range(input_arity))
        steps = [(sid, [c]) for c in cols]
        combiner_args = [input_arity + i for i in range(len(cols))]
        steps.append((pid, combiner_args))
        return steps

    if comp_type == "parallel":
        # tertiary(primary(routed_inputs), secondary(routed_inputs))
        if routing:
            comp = [
                (pid, routing[0]),
                (sid, routing[1]),
            ]
        else:
            comp = [
                (pid, list(range(input_arity))),
                (sid, list(range(input_arity))),
            ]
        if tid is not None:
            comp.append((tid, [input_arity, input_arity + 1]))
        return comp

    raise ValueError(f"Unknown composition type: {comp_type}")


# ---------------------------------------------------------------------------
# Learning pipeline
# ---------------------------------------------------------------------------

def _init_replay_buffer(state: dict):
    """Initialize the replay buffer with curriculum tasks if not already present."""
    if "replay_buffer" not in state:
        state["replay_buffer"] = list(curriculum_tasks(state))
        print(f"  replay buffer initialized with {len(state['replay_buffer'])} curriculum tasks")


def _is_duplicate_discovery(state: dict, candidate: dict) -> bool:
    """Check if this candidate is structurally identical to an existing discovery.

    Only considers it a duplicate if same function IDs, comp_type, AND routing.
    MUL(x,y) is NOT the same as MUL(x,x) — different routing.
    """
    for entry in state.get("replay_buffer", []):
        target = entry["target"]
        if (target.get("primary_id") == candidate.get("primary_id") and
            target.get("secondary_id") == candidate.get("secondary_id") and
            target.get("tertiary_id") == candidate.get("tertiary_id") and
            target.get("comp_type") == candidate.get("comp_type") and
            target.get("routing") == candidate.get("routing") and
            target.get("constants") == candidate.get("constants")):
            return True
    return False


def learn(conn: sqlite3.Connection, state: dict, model: TRM,
          optimizer: torch.optim.Optimizer, name: str,
          examples: list[tuple[list, float | int]], *,
          max_search_steps: int = 10, max_depth: int = 3,
          num_epochs: int = 30, max_retries: int = 2) -> tuple[bool, torch.optim.Optimizer, float]:
    """Full pipeline: search -> simplify -> register -> replay train.

    If search fails, re-trains on accumulated knowledge and retries.
    Returns (success, optimizer, r2_score).
    """
    input_dim = CONFIG["input_dim"]
    seq_len = CONFIG["seq_len"]

    # Ensure replay buffer exists; pre-train if this is the first call
    fresh = "replay_buffer" not in state
    _init_replay_buffer(state)
    if fresh:
        print("  loading curriculum knowledge...")
        train.train_on_replay(model, optimizer, state["replay_buffer"],
                              input_dim=input_dim, seq_len=seq_len,
                              epochs_per_task=2, n_sup=CONFIG["n_sup"])

    print(f"\n{'=' * 50}")
    print(f"Learning: {name}")
    print(f"{'=' * 50}")

    # Search with retries: if search fails, re-train on what we know and try again
    x_input = search.format_examples(examples, input_dim=input_dim, seq_len=seq_len)
    candidate = None
    for attempt in range(1 + max_retries):
        candidate = search.guided(state, model, examples, x_input,
                                  max_steps=max_search_steps, max_depth=max_depth,
                                  temperature_boost=attempt * 1.0)
        if candidate is not None:
            break
        if attempt < max_retries:
            print(f"  search failed (attempt {attempt + 1}), retrying with higher temperature...")

    if candidate is None:
        print(f"  could not find composition for {name}")
        return False, optimizer, 0.0

    constants = candidate.get("constants")
    const_mode = candidate.get("const_mode", "multiplicative")
    if constants:
        print(f"  found: {_fmt_candidate(state, candidate)}  (k={constants[0]}, {const_mode})")
    else:
        print(f"  found: {_fmt_candidate(state, candidate)}")

    # Phase 2: check for duplicate before doing any work
    if _is_duplicate_discovery(state, candidate):
        print(f"  duplicate of existing knowledge — skipping")
        # Find existing fid by name
        fid = None
        for f, m in state["metadata"].items():
            if m["name"] == name:
                fid = f
                break
        if fid is not None:
            r2 = exe.r_squared(state, fid, examples)
            print(f"  verification: R²={r2:.6f}")
            return r2 > 0.999, optimizer, r2
        return True, optimizer, 1.0

    # Phase 3: simplify (strips unused input columns)
    input_arity = len(examples[0][0])
    composition = build_composition(candidate, input_arity, state["loop_id"])
    composition, effective_arity, used_cols = simplify.simplify(
        state, composition, examples, constants, const_mode)

    # Phase 4: register
    old_vocab = reg.vocab_size(state)
    fid = reg.register_learned(conn, state, name, effective_arity, composition,
                               constants, const_mode)
    if used_cols is not None:
        state["metadata"][fid]["used_cols"] = used_cols
    new_vocab = reg.vocab_size(state)
    print(f"  registered {name} as id={fid}")

    # Phase 5: resize model if vocab grew
    if new_vocab > old_vocab:
        resize_heads(model, old_vocab, new_vocab)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    state["replay_buffer"].append({
        "examples": examples,
        "target": dict(candidate),
    })
    # Also store clean version (stripped columns) if applicable
    if used_cols is not None and len(used_cols) < input_arity:
        clean_examples = [
            ([inp[c] for c in used_cols], out)
            for inp, out in examples
        ]
        clean_target = dict(candidate)
        clean_target.pop("routing", None)
        clean_target.pop("null_columns", None)
        state["replay_buffer"].append({
            "examples": clean_examples,
            "target": clean_target,
        })
    print(f"  replay buffer: {len(state['replay_buffer'])} tasks")

    # Phase 6: consolidate all knowledge — curriculum + all discoveries
    print(f"  consolidating knowledge ({len(state['replay_buffer'])} known facts)...")
    train.train_on_replay(model, optimizer, state["replay_buffer"],
                          input_dim=input_dim, seq_len=seq_len,
                          epochs_per_task=2, n_sup=CONFIG["n_sup"])

    # Verify (R² score) — use stripped examples if columns were removed
    if used_cols is not None:
        verify_examples = [([inp[c] for c in used_cols], exp) for inp, exp in examples]
    else:
        verify_examples = examples
    r2 = exe.r_squared(state, fid, verify_examples)
    ok = r2 > 0.999
    print(f"  verification: R²={r2:.6f} ({'passed' if ok else 'FAILED'})")
    return ok, optimizer, r2


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def curriculum_tasks(state: dict) -> list[dict]:
    """Known facts: examples for each primitive and composition type.

    These are already-solved equations. The TRM loads them as knowledge
    so it knows what functions do, how compositions work, and how to
    use NULL to ignore junk columns.
    """
    import random as _rng
    _rng_state = _rng.getstate()
    _rng.seed(777)

    def _junk():
        return round(_rng.uniform(-100, 100), 2)

    tasks = []

    def _find(name):
        for fid, m in state["metadata"].items():
            if m["name"] == name:
                return fid
        return None

    def _add_task(fid, examples):
        """Add a clean task + a noisy variant with NULL columns."""
        target = {
            "primary_id": fid, "secondary_id": None,
            "tertiary_id": None, "comp_type": "none",
        }
        # Clean version
        tasks.append({"target": dict(target), "examples": examples})

        # Noisy version: insert junk columns, teach TRM to use NULL
        arity = len(examples[0][0])
        if arity == 1:
            # [x] -> [junk, x, junk] — NULL cols 0,2, route col 1
            noisy_examples = [
                ([_junk(), inp[0], _junk()], out) for inp, out in examples
            ]
            noisy_target = dict(target)
            noisy_target["routing"] = [[1]]
            noisy_target["null_columns"] = [0, 2]
            tasks.append({"target": noisy_target, "examples": noisy_examples})
        elif arity == 2:
            # [a, b] -> [a, junk, b, junk] — NULL cols 1,3, route cols 0,2
            noisy_examples = [
                ([inp[0], _junk(), inp[1], _junk()], out) for inp, out in examples
            ]
            noisy_target = dict(target)
            noisy_target["routing"] = [[0, 2]]
            noisy_target["null_columns"] = [1, 3]
            tasks.append({"target": noisy_target, "examples": noisy_examples})

    or_id = _find("OR")
    and_id = _find("AND")
    not_id = _find("NOT")
    add_id = _find("ADD")
    sub_id = _find("SUB")
    mul_id = _find("MUL")
    inc_id = _find("INC")
    dec_id = _find("DEC")

    if or_id is not None:
        _add_task(or_id, [([2, 3], 2|3), ([1, 4], 1|4), ([0, 7], 0|7), ([5, 5], 5|5),
                          ([6, 3], 6|3), ([3, 12], 3|12), ([7, 1], 7|1), ([4, 4], 4|4),
                          ([2, 6], 2|6), ([1, 1], 1|1)])
    if and_id is not None:
        _add_task(and_id, [([2, 3], 2&3), ([1, 4], 1&4), ([7, 3], 7&3), ([5, 5], 5&5),
                           ([6, 3], 6&3), ([3, 12], 3&12), ([7, 1], 7&1), ([4, 4], 4&4),
                           ([2, 6], 2&6), ([1, 1], 1&1)])
    if not_id is not None:
        _add_task(not_id, [([0], ~0), ([1], ~1), ([5], ~5), ([7], ~7), ([3], ~3),
                           ([2], ~2), ([4], ~4), ([6], ~6), ([8], ~8), ([10], ~10)])
    if add_id is not None:
        _add_task(add_id, [([0, 0], 0), ([1, 2], 3), ([5, 3], 8), ([7, 10], 17), ([4, 6], 10),
                           ([3, 3], 6), ([8, 2], 10), ([6, 7], 13), ([2, 9], 11), ([1, 1], 2)])
    if sub_id is not None:
        _add_task(sub_id, [([5, 2], 3), ([10, 3], 7), ([7, 7], 0), ([8, 4], 4), ([12, 5], 7),
                           ([9, 1], 8), ([6, 3], 3), ([15, 8], 7), ([4, 2], 2), ([3, 3], 0)])
    if mul_id is not None:
        _add_task(mul_id, [([0, 5], 0), ([1, 5], 5), ([2, 3], 6), ([4, 5], 20), ([3, 7], 21),
                           ([2, 2], 4), ([5, 5], 25), ([3, 4], 12), ([6, 2], 12), ([1, 8], 8)])
    if inc_id is not None:
        _add_task(inc_id, [([0], 1), ([1], 2), ([5], 6), ([10], 11), ([8], 9),
                           ([3], 4), ([7], 8), ([12], 13), ([2], 3), ([15], 16)])
    if dec_id is not None:
        _add_task(dec_id, [([1], 0), ([2], 1), ([6], 5), ([11], 10), ([9], 8),
                           ([4], 3), ([8], 7), ([13], 12), ([3], 2), ([16], 15)])

    # --- Composition tasks: teach the TRM that compositions exist ---
    # Without these, the model only ever sees comp_type="none" and never
    # explores nested/sequential/parallel during search.

    # Sequential: primary(secondary(inputs))
    # INC(MUL(a, b)) = a*b + 1
    if inc_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": inc_id, "secondary_id": mul_id,
                       "tertiary_id": None, "comp_type": "sequential"},
            "examples": [([2, 3], 7), ([1, 5], 6), ([4, 2], 9), ([3, 3], 10), ([2, 6], 13),
                         ([5, 1], 6), ([3, 4], 13), ([1, 1], 2), ([6, 2], 13), ([2, 5], 11)],
        })
    # DEC(ADD(a, b)) = a+b - 1
    if dec_id is not None and add_id is not None:
        tasks.append({
            "target": {"primary_id": dec_id, "secondary_id": add_id,
                       "tertiary_id": None, "comp_type": "sequential"},
            "examples": [([2, 3], 4), ([5, 1], 5), ([4, 4], 7), ([10, 2], 11), ([3, 6], 8),
                         ([1, 1], 1), ([7, 3], 9), ([2, 8], 9), ([6, 6], 11), ([4, 1], 4)],
        })
    # SUB(MUL(a, b)) = a*b - 1  (another sequential)
    if dec_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": dec_id, "secondary_id": mul_id,
                       "tertiary_id": None, "comp_type": "sequential"},
            "examples": [([2, 3], 5), ([1, 5], 4), ([4, 2], 7), ([3, 3], 8), ([5, 2], 9),
                         ([2, 2], 3), ([3, 4], 11), ([1, 1], 0), ([6, 2], 11), ([4, 3], 11)],
        })

    # Nested: primary(secondary(x1), secondary(x2), ...)
    # ADD(INC(a), INC(b)) = (a+1) + (b+1) = a+b+2
    if add_id is not None and inc_id is not None:
        tasks.append({
            "target": {"primary_id": add_id, "secondary_id": inc_id,
                       "tertiary_id": None, "comp_type": "nested"},
            "examples": [([1, 2], 5), ([3, 4], 9), ([0, 0], 2), ([5, 5], 12), ([2, 7], 11),
                         ([4, 1], 7), ([6, 3], 11), ([1, 1], 4), ([3, 8], 13), ([7, 2], 11)],
        })
    # MUL(DEC(a), DEC(b)) = (a-1) * (b-1)
    if mul_id is not None and dec_id is not None:
        tasks.append({
            "target": {"primary_id": mul_id, "secondary_id": dec_id,
                       "tertiary_id": None, "comp_type": "nested"},
            "examples": [([3, 4], 6), ([5, 3], 8), ([2, 6], 5), ([4, 4], 9), ([6, 3], 10),
                         ([7, 2], 6), ([3, 3], 4), ([2, 2], 1), ([5, 5], 16), ([4, 6], 15)],
        })
    # MUL(INC(a), INC(b)) = (a+1) * (b+1)  (another nested)
    if mul_id is not None and inc_id is not None:
        tasks.append({
            "target": {"primary_id": mul_id, "secondary_id": inc_id,
                       "tertiary_id": None, "comp_type": "nested"},
            "examples": [([1, 2], 6), ([3, 4], 20), ([0, 0], 1), ([2, 2], 9), ([4, 1], 10),
                         ([1, 5], 12), ([5, 3], 24), ([2, 1], 6), ([3, 3], 16), ([0, 4], 5)],
        })

    # Parallel: tertiary(primary(route1), secondary(route2))
    # ADD(INC(a), MUL(b, c)) = (a+1) + b*c
    if add_id is not None and inc_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": inc_id, "secondary_id": mul_id,
                       "tertiary_id": add_id, "comp_type": "parallel",
                       "routing": [[0], [1, 2]]},
            "examples": [([1, 2, 3], 8), ([0, 3, 4], 13), ([2, 1, 5], 8), ([3, 2, 2], 8),
                         ([4, 3, 1], 8), ([1, 1, 1], 3), ([5, 2, 3], 12), ([0, 4, 2], 9),
                         ([2, 3, 3], 12), ([3, 1, 4], 8)],
        })
    # SUB(MUL(a,b), ADD(a,b)) = a*b - (a+b)
    if sub_id is not None and mul_id is not None and add_id is not None:
        tasks.append({
            "target": {"primary_id": mul_id, "secondary_id": add_id,
                       "tertiary_id": sub_id, "comp_type": "parallel"},
            "examples": [([3, 4], 5), ([5, 2], 3), ([4, 3], 5), ([6, 2], 4), ([3, 5], 7),
                         ([2, 2], 0), ([4, 4], 8), ([7, 3], 11), ([5, 5], 15), ([2, 6], 4)],
        })
    # MUL(INC(a), MUL(b, c)) = (a+1) * b*c  — parallel with routing
    if mul_id is not None and inc_id is not None:
        tasks.append({
            "target": {"primary_id": inc_id, "secondary_id": mul_id,
                       "tertiary_id": mul_id, "comp_type": "parallel",
                       "routing": [[0], [1, 2]]},
            "examples": [([1, 2, 3], 12), ([0, 3, 4], 12), ([2, 1, 5], 15), ([3, 2, 2], 16),
                         ([1, 4, 2], 16), ([0, 2, 5], 10), ([4, 1, 3], 15), ([2, 2, 2], 12),
                         ([3, 3, 3], 36), ([1, 5, 1], 10)],
        })
    # ADD(MUL(a,b), MUL(a,c)) = a*b + a*c  — parallel, same col in both routes
    if add_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": mul_id, "secondary_id": mul_id,
                       "tertiary_id": add_id, "comp_type": "parallel",
                       "routing": [[0, 1], [0, 2]]},
            "examples": [([2, 3, 4], 14), ([1, 5, 3], 8), ([3, 2, 1], 9), ([4, 1, 2], 12),
                         ([2, 2, 3], 10), ([5, 1, 1], 10), ([3, 4, 2], 18), ([1, 1, 1], 2),
                         ([2, 5, 5], 20), ([4, 3, 3], 24)],
        })

    _rng.setstate(_rng_state)
    return tasks


# ---------------------------------------------------------------------------
# Learning targets
# ---------------------------------------------------------------------------

TARGETS = [
    ("NAND", [([0, 0], ~(0&0)), ([0, 1], ~(0&1)), ([1, 0], ~(1&0)), ([1, 1], ~(1&1)),
              ([3, 5], ~(3&5)), ([7, 3], ~(7&3)), ([15, 10], ~(15&10))]),

    ("XOR",  [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
              ([5, 3], 5^3), ([7, 2], 7^2), ([15, 10], 15^10)]),
]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: TRM, optimizer: torch.optim.Optimizer, state: dict):
    path = os.path.join(CONFIG["checkpoint_dir"], "model.pt")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "n_functions": reg.vocab_size(state),
    }, path)
    print(f"  checkpoint saved to {path}")


def load_checkpoint(model: TRM, optimizer: torch.optim.Optimizer, state: dict):
    path = os.path.join(CONFIG["checkpoint_dir"], "model.pt")
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, weights_only=False)
        old_n = ckpt["n_functions"]
        new_n = reg.vocab_size(state)
        if old_n == new_n:
            model.load_state_dict(ckpt["model_state"], strict=False)
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass
        elif old_n < new_n:
            tmp = create_model(input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                               d_model=CONFIG["d_model"], n_functions=old_n,
                               n_layers=CONFIG["n_layers"],
                               n_recursions=CONFIG["n_recursions"], T=CONFIG["T"])
            tmp.load_state_dict(ckpt["model_state"], strict=False)
            resize_heads(tmp, old_n, new_n)
            model.load_state_dict(tmp.state_dict(), strict=False)
        else:
            print(f"  Stale checkpoint ({old_n} > {new_n} funcs), starting fresh.")
            return False
        print(f"  checkpoint loaded ({old_n} -> {new_n} functions)")
        return True
    except Exception as e:
        print(f"  Warning: could not load checkpoint: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_candidate(state: dict, c: dict) -> str:
    p = reg.get_name(state, c["primary_id"])
    s = reg.get_name(state, c["secondary_id"]) if c.get("secondary_id") is not None else None
    t = reg.get_name(state, c["tertiary_id"]) if c.get("tertiary_id") is not None else None

    if c["comp_type"] == "none":
        return p
    if c["comp_type"] == "loop_direct":
        return f"LOOP({s}, count=b, init=a)"
    if c["comp_type"] == "loop_binary":
        return f"LOOP({s}, count=b, init=0, step=a)"
    if c["comp_type"] == "sequential":
        return f"{p}({s}(...))"
    if c["comp_type"] == "nested":
        return f"{p}({s}(x), {s}(y))"
    if c["comp_type"] == "parallel":
        return f"{t}({p}(...), {s}(...))" if t else f"parallel({p}, {s})"
    return str(c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("NSSR — Neuro-Symbolic Recursive Regression\n")

    # Database + registry
    conn = db.init_db(os.path.join(CONFIG["checkpoint_dir"], "symbolic.db"))
    state = reg.init_registry(conn)
    print(f"primitives: {list(reg.get_names(state, list(state['metadata'].keys())))}")

    # Model + optimizer
    n_funcs = reg.vocab_size(state)
    model = create_model(input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                         d_model=CONFIG["d_model"], n_functions=n_funcs,
                         n_layers=CONFIG["n_layers"],
                         n_recursions=CONFIG["n_recursions"], T=CONFIG["T"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # Load checkpoint if available
    load_checkpoint(model, optimizer, state)

    # Phase 1: curriculum pre-training
    print("\n--- Curriculum pre-training ---")
    for task in curriculum_tasks(state):
        train.train_on_examples(model, optimizer, task["examples"], task["target"],
                                input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                                num_epochs=20, n_sup=CONFIG["n_sup"])

    # Phase 2: progressive learning
    print("\n--- Progressive learning ---")
    for name, examples in TARGETS:
        depth = 3 if name in ("XOR", "MUL") else 2
        epochs = 50 if name in ("XOR", "MUL") else 30
        _, optimizer, _ = learn(conn, state, model, optimizer, name, examples,
                                max_depth=depth, num_epochs=epochs)

    # Save
    save_checkpoint(model, optimizer, state)
    db.print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
