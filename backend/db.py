"""
Database operations for the NSRR symbolic function registry.

All functions take a sqlite3.Connection as their first argument.
Call `init_db` once to get a connection with the schema applied.
"""

import sqlite3
import json
from pathlib import Path

from schema import ALL_TABLES, ALL_INDEXES


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def init_db(db_path: str = "checkpoints/symbolic.db") -> sqlite3.Connection:
    """Create (or open) the database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for table in ALL_TABLES:
        conn.execute(table)
    for index in ALL_INDEXES:
        conn.execute(index)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def add_primitive(conn: sqlite3.Connection, func_id: int, name: str, arity: int) -> None:
    """Insert a primitive function (layer 0, no composition)."""
    conn.execute(
        "INSERT OR REPLACE INTO functions (id, name, arity, layer, composition) "
        "VALUES (?, ?, ?, 0, NULL)",
        (func_id, name, arity),
    )
    conn.commit()


def add_learned(conn: sqlite3.Connection, func_id: int, name: str, arity: int,
                composition: list[tuple[int, list[int]]],
                constants: list[float] | None = None,
                const_mode: str = "multiplicative") -> int:
    """Insert a learned function. Layer is auto-calculated from children.

    composition: list of (child_id, arg_indices) pairs.
    constants: optional list of float constants (e.g. [0.5] for KE = 0.5*m*v²).
    const_mode: "multiplicative" or "additive".
    Returns the computed layer.
    """
    if not composition:
        raise ValueError(f"Learned function '{name}' must have a non-empty composition")

    child_ids = [child_id for child_id, _ in composition]
    placeholders = ",".join("?" * len(child_ids))
    row = conn.execute(
        f"SELECT MAX(layer) AS max_layer FROM functions WHERE id IN ({placeholders})",
        child_ids,
    ).fetchone()

    max_child_layer = row["max_layer"] if row["max_layer"] is not None else -1
    layer = max_child_layer + 1

    comp_json = json.dumps([[cid, args] for cid, args in composition])
    const_json = json.dumps(constants) if constants else None

    conn.execute(
        "INSERT OR REPLACE INTO functions (id, name, arity, layer, composition, constants, const_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (func_id, name, arity, layer, comp_json, const_json, const_mode),
    )
    conn.commit()
    return layer


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_function(conn: sqlite3.Connection, func_id: int) -> dict | None:
    """Return a single function row as a dict, or None."""
    row = conn.execute("SELECT * FROM functions WHERE id = ?", (func_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_composition(conn: sqlite3.Connection, func_id: int) -> list[tuple[int, list[int]]]:
    """Return the composition steps for a function, or [] for primitives."""
    func = get_function(conn, func_id)
    if func is None or func["composition"] is None:
        return []
    return [(cid, args) for cid, args in func["composition"]]


def get_constants(conn: sqlite3.Connection, func_id: int) -> list[float] | None:
    """Return the constants list for a function, or None if no constants."""
    func = get_function(conn, func_id)
    if func is None or func.get("constants") is None:
        return None
    return func["constants"]


def get_const_mode(conn: sqlite3.Connection, func_id: int) -> str:
    """Return the const_mode for a function, default 'multiplicative'."""
    func = get_function(conn, func_id)
    if func is None:
        return "multiplicative"
    return func.get("const_mode") or "multiplicative"


def get_all_functions(conn: sqlite3.Connection) -> list[dict]:
    """Return every function, ordered by layer then id."""
    rows = conn.execute("SELECT * FROM functions ORDER BY layer, id").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_functions_by_layer(conn: sqlite3.Connection, layer: int) -> list[dict]:
    """Return all functions at a specific layer."""
    rows = conn.execute(
        "SELECT * FROM functions WHERE layer = ? ORDER BY id", (layer,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_functions(conn: sqlite3.Connection) -> int:
    """Total number of registered functions."""
    return conn.execute("SELECT COUNT(*) FROM functions").fetchone()[0]


def max_layer(conn: sqlite3.Connection) -> int:
    """Highest layer number, or -1 if the table is empty."""
    val = conn.execute("SELECT MAX(layer) FROM functions").fetchone()[0]
    return val if val is not None else -1


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection) -> None:
    """Print a human-readable summary of the function database."""
    top = max_layer(conn)
    if top < 0:
        print("\n(empty — no functions registered)\n")
        return

    print("\n" + "=" * 60)
    print("Symbolic Function Database")
    print("=" * 60)

    for layer_num in range(top + 1):
        funcs = get_functions_by_layer(conn, layer_num)
        if not funcs:
            continue

        label = "PRIMITIVES" if layer_num == 0 else f"LAYER {layer_num}"
        print(f"\n{label} ({len(funcs)} functions):")

        for f in funcs:
            if layer_num == 0:
                print(f"  {f['id']}: {f['name']}(arity={f['arity']})")
            else:
                child_names = []
                for cid, _ in f["composition"]:
                    child = get_function(conn, cid)
                    child_names.append(child["name"] if child else f"?{cid}")
                print(f"  {f['id']}: {f['name']}(arity={f['arity']}) = {' -> '.join(child_names)}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a Row to a plain dict, parsing composition and constants JSON."""
    d = dict(row)
    if d["composition"] is not None:
        d["composition"] = json.loads(d["composition"])
    if d.get("constants") is not None:
        d["constants"] = json.loads(d["constants"])
    return d
