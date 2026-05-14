"""
Schema definition for the NSRR symbolic function database.

Single table design: every function (primitive or learned) lives in `functions`.
Primitives have layer=0 and composition=NULL.
Learned functions have layer>0 and composition=JSON array of [child_id, [arg_indices]].
"""

FUNCTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS functions (
    id          INTEGER PRIMARY KEY,
    name        TEXT    UNIQUE NOT NULL,
    arity       INTEGER NOT NULL,
    layer       INTEGER NOT NULL,
    composition TEXT,
    constants   TEXT,
    const_mode  TEXT DEFAULT 'multiplicative',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

LAYER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_functions_layer ON functions(layer)
"""

ALL_TABLES = [FUNCTIONS_TABLE]
ALL_INDEXES = [LAYER_INDEX]
