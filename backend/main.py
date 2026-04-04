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
    "seq_len": 4,
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
    """
    comp_type = candidate["comp_type"]
    pid = candidate["primary_id"]
    sid = candidate.get("secondary_id")
    tid = candidate.get("tertiary_id")

    if comp_type == "none":
        return [(pid, list(range(input_arity)))]

    if comp_type == "loop_direct":
        # LOOP(body_fn=sid, count=input[1], init=input[0])
        return [(pid, [sid, 1, 0])]

    if comp_type == "loop_binary":
        # MUL(a,b) = LOOP(ADD, count=b, init=0, step_arg=a)
        # 4-arg LOOP in composition: [body_fn_id, count_idx, init_idx, step_arg_idx]
        # init_idx uses a special marker -1 meaning "literal 0"
        return [(pid, [sid, 1, -1, 0])]

    if comp_type == "sequential":
        # secondary(inputs) -> primary(result)
        return [
            (sid, list(range(input_arity))),
            (pid, [input_arity]),
        ]

    if comp_type == "nested":
        # primary(secondary(x0), secondary(x1))
        return [
            (sid, [0]),
            (sid, [1]),
            (pid, [input_arity, input_arity + 1]),
        ]

    if comp_type == "parallel":
        # tertiary(primary(inputs), secondary(inputs))
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

def learn(conn: sqlite3.Connection, state: dict, model: TRM,
          optimizer: torch.optim.Optimizer, name: str,
          examples: list[tuple[list[int], int]], *,
          max_search_steps: int = 10, max_depth: int = 3,
          num_epochs: int = 30) -> tuple[bool, torch.optim.Optimizer]:
    """Full pipeline: search -> simplify -> register -> resize -> train.

    Returns (success, optimizer) — optimizer may be recreated after resize.
    """
    input_dim = CONFIG["input_dim"]
    seq_len = CONFIG["seq_len"]

    print(f"\n{'=' * 50}")
    print(f"Learning: {name}")
    print(f"{'=' * 50}")

    # Phase 1: search
    x_input = search.format_examples(examples, input_dim=input_dim, seq_len=seq_len)
    candidate = search.guided(state, model, examples, x_input,
                              max_steps=max_search_steps, max_depth=max_depth)
    if candidate is None:
        candidate = search.exhaustive(state, examples, max_depth=max_depth)

    if candidate is None:
        print(f"  could not find composition for {name}")
        return False, optimizer

    print(f"  found: {_fmt_candidate(state, candidate)}")

    # Phase 2: simplify
    input_arity = len(examples[0][0])
    composition = build_composition(candidate, input_arity, state["loop_id"])
    composition = simplify.simplify(state, composition, examples)

    # Phase 3: register
    old_vocab = reg.vocab_size(state)
    fid = reg.register_learned(conn, state, name, input_arity, composition)
    new_vocab = reg.vocab_size(state)
    print(f"  registered {name} as id={fid}")

    # Phase 4: resize model if vocab grew
    if new_vocab > old_vocab:
        resize_heads(model, old_vocab, new_vocab)
        # Must create fresh optimizer — old one holds stale param references
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # Phase 5: train
    train.train_on_examples(model, optimizer, examples, candidate,
                            input_dim=input_dim, seq_len=seq_len,
                            num_epochs=num_epochs, n_sup=CONFIG["n_sup"])

    # Verify
    ok = all(reg.execute(state, fid, inp) == exp for inp, exp in examples)
    print(f"  verification: {'passed' if ok else 'FAILED'}")
    return ok, optimizer


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

def curriculum_tasks(state: dict) -> list[dict]:
    """Pre-training tasks: identity mappings for each primitive."""
    tasks = []

    def _find(name):
        for fid, m in state["metadata"].items():
            if m["name"] == name:
                return fid
        return None

    or_id = _find("OR")
    and_id = _find("AND")
    not_id = _find("NOT")
    inc_id = _find("INC")
    dec_id = _find("DEC")

    if or_id is not None:
        tasks.append({"target": {"primary_id": or_id, "secondary_id": None, "tertiary_id": None, "comp_type": "none"},
                       "examples": [([2, 3], 2|3), ([1, 4], 1|4), ([0, 7], 0|7), ([5, 5], 5|5)]})
    if and_id is not None:
        tasks.append({"target": {"primary_id": and_id, "secondary_id": None, "tertiary_id": None, "comp_type": "none"},
                       "examples": [([2, 3], 2&3), ([1, 4], 1&4), ([7, 3], 7&3), ([5, 5], 5&5)]})
    if not_id is not None:
        tasks.append({"target": {"primary_id": not_id, "secondary_id": None, "tertiary_id": None, "comp_type": "none"},
                       "examples": [([0], ~0), ([1], ~1), ([5], ~5), ([7], ~7)]})
    if inc_id is not None:
        tasks.append({"target": {"primary_id": inc_id, "secondary_id": None, "tertiary_id": None, "comp_type": "none"},
                       "examples": [([0], 1), ([1], 2), ([5], 6), ([10], 11)]})
    if dec_id is not None:
        tasks.append({"target": {"primary_id": dec_id, "secondary_id": None, "tertiary_id": None, "comp_type": "none"},
                       "examples": [([1], 0), ([2], 1), ([6], 5), ([11], 10)]})

    return tasks


# ---------------------------------------------------------------------------
# Learning targets
# ---------------------------------------------------------------------------

TARGETS = [
    ("NAND", [([0, 0], ~(0&0)), ([0, 1], ~(0&1)), ([1, 0], ~(1&0)), ([1, 1], ~(1&1)),
              ([3, 5], ~(3&5)), ([7, 3], ~(7&3)), ([15, 10], ~(15&10))]),

    ("XOR",  [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
              ([5, 3], 5^3), ([7, 2], 7^2), ([15, 10], 15^10)]),

    ("ADD",  [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 2),
              ([2, 3], 5), ([4, 2], 6), ([5, 5], 10), ([7, 3], 10)]),

    ("SUB",  [([5, 2], 3), ([10, 3], 7), ([7, 7], 0), ([8, 4], 4),
              ([15, 5], 10), ([20, 8], 12)]),

    ("MUL",  [([0, 0], 0), ([0, 5], 0), ([1, 5], 5), ([2, 3], 6),
              ([3, 4], 12), ([4, 5], 20), ([5, 5], 25)]),
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
    ckpt = torch.load(path, weights_only=False)
    old_n = ckpt["n_functions"]
    new_n = reg.vocab_size(state)
    if old_n != new_n:
        # Load into a temp model with old vocab, then resize
        tmp = create_model(input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                           d_model=CONFIG["d_model"], n_functions=old_n,
                           n_layers=CONFIG["n_layers"],
                           n_recursions=CONFIG["n_recursions"], T=CONFIG["T"])
        tmp.load_state_dict(ckpt["model_state"], strict=False)
        # Copy weights into current model
        model.load_state_dict(tmp.state_dict(), strict=False)
        resize_heads(model, old_n, new_n)
    else:
        model.load_state_dict(ckpt["model_state"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
    print(f"  checkpoint loaded ({old_n} -> {new_n} functions)")
    return True


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
        _, optimizer = learn(conn, state, model, optimizer, name, examples,
                             max_depth=depth, num_epochs=epochs)

    # Save
    save_checkpoint(model, optimizer, state)
    db.print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
