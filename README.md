# NSSR -- Neuro-Symbolic Recursive Regression

NSSR is a system that **discovers symbolic mathematical expressions from input/output examples**. It combines a tiny recursive transformer (TRM) with a symbolic function registry, letting a neural network *guide* a combinatorial search over function compositions. The system progressively learns higher-level functions by composing primitives (ADD, MUL, etc.) and previously-discovered functions, building a growing vocabulary of reusable symbolic building blocks.

**Core idea**: Instead of training a large neural network to approximate a function, NSSR uses a small recursive transformer to *predict which symbolic compositions are likely correct*, then validates them exactly. The result is a fully interpretable symbolic expression, not a black-box model.

---

## Architecture Overview

```
Input/Output Examples
        |
        v
  [format_examples]  (search.py)     -- encode as float tensors
        |
        v
  [TRM Model]  (model.py)            -- recursive transformer predicts compositions
        |
        v
  [Guided Search]  (search.py)       -- generate candidates from TRM predictions
        |                               validate against examples
        v
  [Simplify]  (simplify.py)          -- prune redundant steps, strip unused columns
        |
        v
  [Register]  (registry.py + db.py)  -- store as new reusable function
        |
        v
  [Replay Train]  (train.py)         -- retrain TRM on all known functions
        |
        v
  [Next Task...]                      -- vocabulary grows, harder functions become reachable
```

---

## File-by-File Documentation

### `schema.py` -- Database Schema Definition

Defines the SQLite schema for persisting the symbolic function registry. Single-table design where every function (primitive or learned) lives in one table.

**Constants:**

| Name | Description |
|---|---|
| `FUNCTIONS_TABLE` | SQL `CREATE TABLE` statement for the `functions` table. Columns: `id` (primary key), `name` (unique), `arity`, `layer`, `composition` (JSON or NULL for primitives), `constants` (JSON or NULL), `const_mode` (default `'multiplicative'`), `created_at` (timestamp). |
| `LAYER_INDEX` | SQL `CREATE INDEX` on the `layer` column for efficient layer-based queries. |
| `ALL_TABLES` | List containing `FUNCTIONS_TABLE`. Used by `db.init_db()` to ensure schema exists. |
| `ALL_INDEXES` | List containing `LAYER_INDEX`. Used by `db.init_db()` to ensure indexes exist. |

**Design notes:**
- Primitives have `layer=0` and `composition=NULL`.
- Learned (composed) functions have `layer>0` and `composition` is a JSON array of `[child_id, [arg_indices]]` pairs.
- `constants` stores optional fitted constants (e.g., `[0.5]` for KE = 0.5 * m * v^2).
- `const_mode` is either `"multiplicative"` (result *= k) or `"additive"` (result += k).

---

### `db.py` -- Database Operations

All SQLite operations for the symbolic function registry. Every function takes a `sqlite3.Connection` as its first argument. Stateless -- no module-level globals.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `init_db` | `(db_path: str = "checkpoints/symbolic.db") -> sqlite3.Connection` | Creates or opens the database, ensures the schema (tables + indexes) exists, and returns a connection with `row_factory=sqlite3.Row` enabled. Creates parent directories if needed. |
| `add_primitive` | `(conn, func_id: int, name: str, arity: int) -> None` | Inserts a primitive function (layer=0, no composition) using `INSERT OR REPLACE`. |
| `add_learned` | `(conn, func_id: int, name: str, arity: int, composition: list[tuple[int, list[int]]], constants=None, const_mode="multiplicative") -> int` | Inserts a learned function. Auto-calculates `layer` as `max(child layers) + 1`. Serializes `composition` and `constants` to JSON. Returns the computed layer number. Raises `ValueError` if composition is empty. |
| `get_function` | `(conn, func_id: int) -> dict \| None` | Returns a single function row as a dict (with parsed JSON fields), or `None` if not found. |
| `get_composition` | `(conn, func_id: int) -> list[tuple[int, list[int]]]` | Returns the composition steps for a function, or `[]` for primitives. |
| `get_constants` | `(conn, func_id: int) -> list[float] \| None` | Returns the constants list for a function, or `None`. |
| `get_const_mode` | `(conn, func_id: int) -> str` | Returns the `const_mode` for a function, defaulting to `"multiplicative"`. |
| `get_all_functions` | `(conn) -> list[dict]` | Returns every function ordered by layer then id. |
| `get_functions_by_layer` | `(conn, layer: int) -> list[dict]` | Returns all functions at a specific layer, ordered by id. |
| `count_functions` | `(conn) -> int` | Total number of registered functions. |
| `max_layer` | `(conn) -> int` | Highest layer number, or `-1` if the table is empty. |
| `print_summary` | `(conn) -> None` | Prints a human-readable summary of the entire function database, organized by layer. Shows composition chains for learned functions. |
| `_row_to_dict` | `(row: sqlite3.Row) -> dict` | **Internal.** Converts a Row to a plain dict, parsing `composition` and `constants` from JSON strings back into Python objects. |

---

### `registry.py` -- Function Registry

The runtime function registry. Manages all primitives and learned (composed) functions. State is a plain dict passed to every function -- no globals, no classes.

#### Registry State Shape

```python
{
    "functions":  {int: callable},   # id -> fn(inputs) -> result
    "metadata":   {int: dict},       # id -> {name, arity, layer, ...}
    "next_id":    int,               # next available function id
    "loop_id":    int | None,        # cached id of LOOP primitive
}
```

#### Primitives

All primitives are defined via `_make_primitives()`. Each is `(arity, factory)` where `factory(execute_fn) -> callable(inputs) -> result`. The `execute_fn` indirection lets iteration primitives (LOOP, WHILE, ACCUM) call back into the registry without circular references.

| Name | Arity | Description |
|---|---|---|
| `OR` | 2 | Bitwise OR: `inp[0] \| inp[1]` |
| `AND` | 2 | Bitwise AND: `inp[0] & inp[1]` |
| `NOT` | 1 | Bitwise NOT: `~inp[0]` |
| `ADD` | 2 | Addition: `inp[0] + inp[1]` |
| `SUB` | 2 | Subtraction: `inp[0] - inp[1]` |
| `MUL` | 2 | Multiplication: `inp[0] * inp[1]` |
| `INC` | 1 | Increment: `inp[0] + 1` |
| `DEC` | 1 | Decrement: `inp[0] - 1` |
| `DIV` | 2 | Float division: `inp[0] / inp[1]` (returns 0.0 if divisor is 0) |
| `LT` | 2 | Less than: `int(inp[0] < inp[1])` |
| `LTE` | 2 | Less than or equal: `int(inp[0] <= inp[1])` |
| `GT` | 2 | Greater than: `int(inp[0] > inp[1])` |
| `GTE` | 2 | Greater than or equal: `int(inp[0] >= inp[1])` |
| `EQ` | 2 | Equality: `int(inp[0] == inp[1])` |
| `NEQ` | 2 | Not equal: `int(inp[0] != inp[1])` |
| `COND` | 3 | Conditional: `inp[1] if inp[0] != 0 else inp[2]` |
| `CONST` | 1 | Identity/constant: returns `inp[0]` |
| `NULL` | 1 | Column eraser: always returns 0. Marks an input column as unused during search. |
| `LOOP` | -1 (variadic) | Iteration. 3-arg form: `body(result)` repeated `count` times from `init`. 4-arg form: `body(result, step_arg)` repeated `count` times. Max 1000 iterations. Enables expressing ADD as `LOOP(INC, b, a)` and MUL as `LOOP(ADD, b, 0, a)`. |
| `WHILE` | -1 (variadic) | Conditional iteration: `WHILE(cond_fn, body_fn, state, limit)`. Runs `body(state)` while `cond(state) != 0`, up to `limit` iterations (max 1000). |
| `ACCUM` | -1 (variadic) | Counting iteration: `ACCUM(cond_fn, body_fn, state, counter, limit)`. Like WHILE but increments and returns a counter. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `_make_primitives` | `() -> dict[str, tuple[int, Any]]` | **Internal.** Builds the `{name: (arity, factory)}` dictionary for all built-in primitives. The factory pattern allows iteration primitives to receive an `execute_fn` callback for recursive evaluation. |
| `_loop_factory` | `(execute_fn) -> callable` | **Internal.** Factory for the LOOP primitive. Supports unary (3-arg) and binary (4-arg) loop bodies. Enforces a 1000-iteration safety limit. |
| `_while_factory` | `(execute_fn) -> callable` | **Internal.** Factory for the WHILE primitive. Takes `(cond_fn_id, body_fn_id, state, limit)`. Runs body while condition is truthy, up to limit. |
| `_accum_factory` | `(execute_fn) -> callable` | **Internal.** Factory for the ACCUM primitive. Like WHILE but counts iterations and returns the counter. |
| `_empty_state` | `() -> dict` | **Internal.** Returns a blank registry state dict. |
| `execute` | `(state: dict, func_id: int, inputs: list) -> Any` | Executes a registered function by id. Raises `ValueError` if the function id is not found. |
| `init_registry` | `(conn: sqlite3.Connection) -> dict` | Registers all primitives into both the DB and a fresh state dict. Creates the `_exec` closure so LOOP/WHILE/ACCUM can call back into `execute()`. Returns the populated state. |
| `load_registry` | `(conn: sqlite3.Connection) -> dict` | Rebuilds the full registry state from an existing database. Re-creates primitive callables from their factories and rebuilds composition closures for learned functions. |
| `register_learned` | `(conn, state, name, arity, composition, constants=None, const_mode="multiplicative") -> int` | Registers a new composed function. Checks for duplicate names first. Persists to DB via `db.add_learned()`, builds a composition closure, updates state metadata. Returns the new function id. |
| `_make_composed_fn` | `(state, composition, execute_fn, constants=None, const_mode="multiplicative") -> callable` | **Internal.** Builds a callable that runs a composition step-by-step. The composition is `[(child_func_id, arg_indices), ...]` where arg_indices index into an `available_values` list (starts with inputs, grows as each step appends its result). Special arg index `-1` means literal `0` (used by LOOP for MUL). Applies optional constant scaling/offset at the end. |
| `vocab_size` | `(state: dict) -> int` | Number of registered functions. Used by the model to size output heads. |
| `get_name` | `(state, func_id: int) -> str` | Get function name by id. |
| `get_names` | `(state, func_ids: list[int]) -> list[str]` | Get multiple function names. |

---

### `model.py` -- TRM (Tiny Recursive Reasoning Model)

The neural network core. Based on the paper **"Less is More: Recursive Reasoning with Tiny Networks"** (Jolicoeur-Martineau, 2025).

#### Theory: Why Recursion Instead of Depth?

Traditional transformers gain reasoning power by stacking more layers. TRM takes the opposite approach: use a **tiny** network (2 layers) but **recurse** it many times. This is analogous to how humans solve complex problems -- not by having a bigger brain, but by *thinking longer* about the same problem.

The key insight is that a small network applied recursively can simulate a much deeper network:

```
Effective depth = T * (n + 1) * n_layers
```

With the default config (T=3, n=6, n_layers=2):
- Effective depth = 3 * 7 * 2 = **42 effective layers**
- But only **2 actual layers** of parameters to train

This is memory-efficient and avoids the vanishing gradient problem of very deep networks.

#### Carry State

The model maintains two evolving tensor states across recursion steps:

- **y** (answer embedding): The model's current best guess at the answer. This is what the output heads read from.
- **z** (latent reasoning): Internal "scratch space" for intermediate reasoning. Not directly read by output heads.

Both are `(batch, seq_len, d_model)` tensors, initialized to zeros.

#### Three Levels of Recursion

**1. Latent Recursion** (`_latent_recursion`):
One full cycle of reasoning. First, `n` steps refine z (the reasoning state) by feeding it `x + y + z` through the transformer. Then 1 step refines y (the answer) by feeding it `y + z` -- crucially **without x**, so the network learns to distinguish "reasoning about the input" from "updating the answer".

**2. Deep Recursion** (`_deep_recursion`):
Runs `T` latent recursion cycles, but only the **last** one computes gradients. The first `T-1` passes are `torch.no_grad()`, meaning they improve y and z "for free" (no memory cost for backprop). This gives the effective depth multiplication.

**3. Deep Supervision** (the forward pass):
The outer training loop calls `forward()` multiple times (N_sup steps), detaching the carry between each call. This means the model must produce a useful answer at *every* supervision step, not just the final one -- it learns to iteratively refine rather than plan for a single shot.

#### Output Heads

The model produces 5 prediction heads from pooled y (mean across sequence dimension):

| Head | Output Shape | Description |
|---|---|---|
| `primary_logits` | `(batch, n_functions)` | Which function to use as the primary/outer function |
| `secondary_logits` | `(batch, n_functions)` | Which function to compose with (inner function) |
| `tertiary_logits` | `(batch, n_functions)` | Combiner function for parallel composition |
| `composition_logits` | `(batch, 4)` | How to compose: none / sequential / nested / parallel |
| `halt_logits` | `(batch, 1)` | Whether to stop reasoning (ACT-style halting) |

#### Classes and Dataclasses

| Name | Description |
|---|---|
| `Carry` | Dataclass holding `y` (answer embedding), `z` (latent reasoning), `steps` (counter), `halted` (boolean mask). |
| `_Block` | Single pre-norm transformer block. LayerNorm -> MultiheadAttention (self-attention) -> residual -> LayerNorm -> FFN (Linear -> GELU -> Dropout -> Linear -> Dropout) -> residual. |
| `TRM` | The full model. Contains input projection, positional embedding, shared transformer blocks, recursion layer norm, and 5 output heads. |

#### TRM Methods

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(*, input_dim, seq_len, d_model, n_heads=8, n_layers=2, d_ff=None, dropout=0.1, n_functions, n_recursions=6, T=3)` | Initializes all components. `d_ff` defaults to `d_model * 4`. Creates input projection, learned positional embeddings, `n_layers` transformer blocks, a layer norm for recursion input stabilization, and 5 output heads. |
| `_apply_blocks` | `(h: Tensor) -> Tensor` | Runs input through all transformer blocks sequentially. |
| `_latent_recursion` | `(x, y, z) -> (y, z)` | One full latent recursion cycle: n steps of `z = blocks(ln(x+y+z))`, then 1 step of `y = blocks(ln(y+z))`. The layer norm (`ln_recursion`) prevents NaN explosion from repeated additions. |
| `_deep_recursion` | `(x, y, z) -> (y, z)` | T-1 no-grad passes + 1 grad pass through `_latent_recursion`. Multiplies effective depth by T without proportional memory cost. |
| `forward` | `(carry: Carry, x: Tensor) -> (Carry, dict[str, Tensor])` | One deep supervision step. Projects input, runs deep recursion, produces output logits from mean-pooled y. Returns new carry (detached) and output dict. |

#### Factory/Helper Functions

| Function | Signature | Description |
|---|---|---|
| `create_model` | `(*, input_dim, seq_len, d_model, n_functions, **kwargs) -> TRM` | Factory that creates a TRM with sensible defaults. |
| `fresh_carry` | `(batch_size, seq_len, d_model) -> Carry` | Creates a zeroed carry state (all tensors are zeros). |
| `reset_carry` | `(carry: Carry) -> Carry` | Zeros out an existing carry (preserves tensor shapes/device via `zeros_like`). |
| `resize_heads` | `(model: TRM, old_n: int, new_n: int) -> None` | Expands the 3 function output heads (primary, secondary, tertiary) to accommodate a larger vocabulary. Copies existing weights and initializes new rows with small noise around the existing mean. No-op if `new_n <= old_n`. |

---

### `train.py` -- Training

Training logic for the TRM model. Uses deep supervision where each training step runs multiple supervision steps with detached carry between them.

#### Constants

| Name | Description |
|---|---|
| `COMP_TYPE_INDEX` | Maps composition type strings to integer indices: `{"none": 0, "sequential": 1, "nested": 2, "parallel": 3, "loop_direct": 0, "loop_binary": 0}`. Loop types map to 0 because they are handled separately. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `compute_loss` | `(outputs: dict[str, Tensor], target: dict, batch_size: int) -> Tensor` | Multi-head cross-entropy loss for a single target composition. Computes 5 losses: primary/secondary/tertiary function prediction (cross-entropy), composition type prediction (cross-entropy), and halt prediction (binary cross-entropy). Down-weights secondary loss to 0.1 when `secondary_id` is None, tertiary loss to 0.1 when `tertiary_id` is None. Adds entropy regularization (weight=0.05) on the 3 function heads to prevent the softmax from collapsing to always predict one function. |
| `train_on_examples` | `(model, optimizer, examples, target, *, input_dim, seq_len, num_epochs=30, n_sup=16) -> list[float]` | Trains the model to predict a specific target composition from examples. Each epoch: creates fresh carry, runs `n_sup` deep supervision steps, accumulates loss at each step with gradient clipping (max norm=1.0). Implements ACT-style early halting -- if the halt head's mean probability exceeds 0.5 after at least 1 step, stops early. Returns per-epoch loss values. |
| `train_on_replay` | `(model, optimizer, replay_buffer, *, input_dim, seq_len, epochs_per_task=2, n_sup=16) -> list[float]` | Full replay training across all known tasks. Pre-encodes all tasks, then for `n_tasks * epochs_per_task` epochs: iterates through all tasks in sequential order (curriculum-style), trains each with fresh carry and deep supervision. Includes early stopping (patience=5 epochs with no improvement). Returns per-epoch average losses. |

#### Training Theory: Deep Supervision

Standard training would run the model once and compute loss. Deep supervision instead runs the model `N_sup` times, computing and backpropagating loss at each step. The carry state (y, z) persists across steps (detached to prevent gradient flow through the entire chain), forcing the model to learn *iterative refinement* -- each pass should improve the answer.

This mimics how the model will be used at inference time: the search loop calls `forward()` repeatedly, reading predictions at each step.

---

### `search.py` -- Program Search

TRM-guided combinatorial search over function compositions. The TRM predicts likely candidates, which are then validated exactly against examples.

#### Constants

| Name | Description |
|---|---|
| `COMP_TYPES` | `["none", "sequential", "nested", "parallel"]` -- the 4 composition types the model can predict. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `guided` | `(state, model, examples, x_input, *, max_steps=10, max_depth=3, temperature_boost=0.0) -> dict \| None` | Public entry point. Delegates to `_guided_inner` and returns just the candidate (or None). |
| `_guided_inner` | `(state, model, examples, x_input, *, max_steps, max_depth, temperature_boost) -> (dict \| None, float)` | Core search loop. For each step: (1) runs the model forward on the carry, (2) extracts top-k predictions from each head with temperature scaling (starts high=2.0, decays by 0.05/step, boosted on retries), (3) generates candidate compositions from top-k combinations including NULL column variants, (4) validates each candidate against both train and holdout splits, (5) if valid candidates found, returns the simplest one (by `_complexity_score`). If no exact match, tries constant fitting on near misses. Adds small noise to carry between steps for exploration. Uses holdout validation (1/3 split) to prevent overfitting. |
| `_log_trm_step` | `(state, step, logits, comp_logits, n_functions)` | **Internal.** Logs the TRM's top-5 predictions with probabilities at each search step. Shows what the model is "thinking". |
| `_log_near_misses` | `(state, near_misses, examples, top_n=5)` | **Internal.** Scores and logs the best near-miss candidates by R^2. Helps diagnose why search failed. |
| `_describe_candidate` | `(state, cand) -> str` | **Internal.** Human-readable description of a candidate (e.g., `"ADD(MUL(...))"` or `"SUB(INC(each))"`). Includes routing and constant info. |
| `_null_column_subsets` | `(input_arity, max_null=0) -> list[list[int]]` | **Internal.** Generates all useful subsets of columns to keep (others are NULL'd out). Always includes "keep all" and single-column subsets. For 3+ columns, includes all pairs. For 4+ columns, includes all triples. This enables the model to discover that some input columns are irrelevant. |
| `_find_null_id` | `(state) -> int \| None` | **Internal.** Finds the NULL primitive's function id by scanning metadata. |
| `_generate_candidates` | `(state, tops, comp_top, max_depth, input_arity, null_subsets, null_id) -> list[dict]` | **Internal.** Builds candidate dicts from top-k TRM predictions at 3 depth levels. **Depth 1:** single functions with all NULL column/routing variants. **Depth 2:** binary compositions (sequential, nested, parallel) with NULL variants, plus LOOP candidates for unary and binary bodies. **Depth 3:** compositions using all previously-learned functions (not just TRM-predicted ones), expanding the search space for complex compositions. Filters out NULL and LOOP from normal function slots. |
| `_add_composition_candidates` | `(candidates, state, pid, sid, comp, input_arity, null_subsets)` | **Internal.** Adds sequential/nested candidates with NULL column variants. For sequential: routes kept columns into secondary, primary takes the result. For nested: applies secondary to each kept column, combines with primary. |
| `_add_parallel_candidates` | `(candidates, state, pid, sid, tid, input_arity, comp, null_subsets, max_routings=30)` | **Internal.** Adds parallel candidates: `tertiary(primary(route1), secondary(route2))`. Generates all routing combinations from kept columns with a cap of 30 to prevent combinatorial explosion. |
| `_try_fit_constants` | `(state, cand, examples, r2_threshold=0.999) -> dict \| None` | **Internal.** Tries fitting a single multiplicative or additive constant to a near-miss candidate. First tries `k * produced = expected` (least-squares scale), then `produced + k = expected` (constant offset). Returns augmented candidate if R^2 > 0.999, else None. |
| `_try_fit_any` | `(state, candidates, examples) -> dict \| None` | **Internal.** Tries constant fitting on a list of candidates. Returns the first successful fit. |
| `_fit_scale` | `(produced, expected) -> float \| None` | **Internal.** Least-squares fit for multiplicative constant: `k = sum(p*e) / sum(p*p)`. Returns None if denominator is near-zero. |
| `_fit_offset` | `(produced, expected) -> float \| None` | **Internal.** Fits an additive constant: computes mean of `expected - produced` differences. Returns the offset only if all differences are within 1e-6 of the mean (i.e., it's truly a constant offset, not noise). |
| `_r2` | `(actual, predicted) -> float` | **Internal.** Computes R-squared (coefficient of determination). Returns 1.0 if both SS_res and SS_tot are 0, `-inf` if SS_tot is 0 but SS_res isn't. |
| `_candidate` | `(primary_id, secondary_id=None, tertiary_id=None, *, comp_type="none", routing=None, null_columns=None) -> dict` | **Internal.** Helper to construct a candidate dict. |
| `_complexity_score` | `(c: dict) -> int` | **Internal.** Scores candidate complexity (lower = simpler = preferred). Composition type scores: none=0, sequential/nested=1, parallel=2, loop=3. Adds 1 for each non-None secondary/tertiary id and for constants. |
| `_cand_key` | `(c: dict) -> tuple` | **Internal.** Creates a hashable key from a candidate for deduplication. Includes all fields: primary, secondary, tertiary, comp_type, routing, null_columns. |
| `format_examples` | `(examples, *, input_dim, seq_len) -> Tensor` | Encodes examples as float tensors for the TRM. Each input value is encoded as 4 features: `val/100` (normalized), `sign` (+1/-1), `log1p(abs(val))` (magnitude), `fractional part` (decimal component). Output shape: `(batch_size, seq_len, input_dim)`. |

#### Search Theory: NULL-Based Column Elimination

Real-world data often has irrelevant columns. Instead of requiring the user to pre-select columns, the search generates candidates that "NULL out" subsets of columns. For a 3-column input `[a, b, c]`, the search tries:
- Keep all: `[a, b, c]`
- Keep one: `[a]`, `[b]`, `[c]`
- Keep pairs: `[a, b]`, `[a, c]`, `[b, c]`

For each subset, remaining columns are routed into function slots with repetition allowed (so `MUL(a, a)` for squaring is discoverable).

---

### `executor.py` -- Program Executor

Runs composed programs described by candidate dicts against inputs. Validates candidates against input/output examples.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `run` | `(state, candidate, inputs) -> Any` | Executes a candidate program on inputs. Calls `_run_base` for the core composition, then applies fitted constants if present (multiplicative or additive). |
| `_run_base` | `(state, candidate, inputs) -> Any` | **Internal.** Executes the base composition logic. Handles all 6 composition types: **none** (single function with optional routing), **sequential** (`primary(secondary(inputs))`), **nested** (`primary(secondary(x1), secondary(x2), ...)`), **parallel** (`tertiary(primary(route1), secondary(route2))`), **loop_direct** (`LOOP(body, count, init)`), **loop_binary** (`LOOP(body, count, 0, step_arg)`). Respects routing (column selection) when present. |
| `validate` | `(state, candidate, examples) -> bool` | Checks if a candidate produces the correct output for every example. Uses `math.isclose` with `rel_tol=1e-6, abs_tol=1e-9` for float comparison. Returns False on any exception. |
| `r_squared` | `(state, func_id, examples) -> float` | Computes R^2 score for a registered function against examples. Used for post-registration verification. Returns `-inf` on any failure. |

---

### `simplify.py` -- Composition Simplifier

After search finds a working composition, tries to make it shorter/simpler before registering. Three strategies tried in order.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `simplify` | `(state, composition, examples, constants=None, const_mode="multiplicative") -> (composition, effective_arity, used_cols)` | Public entry point. Tries 3 simplification strategies in order. Returns the simplified composition, the effective arity (may be less than input arity if columns were stripped), and `used_cols` (list of original column indices kept, or None if all used). |
| `_try_single_function` | `(state, examples) -> list[tuple[int, list[int]]] \| None` | **Strategy 1.** Checks if any existing non-primitive function already solves this exact problem. If so, replaces the entire composition with a single reference. Only used when no constants are involved (constants make the function unique). |
| `_try_prune` | `(state, composition, examples, constants=None, const_mode="multiplicative") -> list[tuple[int, list[int]]] \| None` | **Strategy 2.** Tries removing each non-final step from the composition. For each removal, adjusts arg indices that referenced the removed step or later steps. Validates the shortened composition against all examples. Returns the first valid shortened form, or None. |
| `_strip_unused_inputs` | `(state, composition, examples, constants=None, const_mode="multiplicative") -> (composition, effective_arity, used_cols)` | **Strategy 3.** Identifies input columns that no composition step references, removes them, and remaps arg indices. For example, if a 3-input composition only uses columns 0 and 2, it remaps to a 2-input composition using columns 0 and 1. Validates the remapped composition against remapped examples. |
| `_validate` | `(state, composition, examples, constants=None, const_mode="multiplicative") -> bool` | **Internal.** Tests a composition against all examples by running it step-by-step: builds an `available_values` list starting with inputs, appends each step's result, checks final result against expected output. Applies constants at the end. |
| `_complexity` | `(state, composition) -> int` | **Internal.** Scores composition complexity: `n_terms * 100 + max_layer * 10 + total_args`. Lower is better. Heavily penalizes more steps to prefer simpler compositions. |

---

### `main.py` -- Orchestrator / Entry Point

Single entry point that runs the full NSSR learning pipeline. Coordinates all other modules.

#### Constants

| Name | Description |
|---|---|
| `CONFIG` | Global configuration dict: `input_dim=32`, `seq_len=8`, `d_model=128`, `n_heads=8`, `n_layers=2`, `n_recursions=6`, `T=3`, `n_sup=16`, `dropout=0.1`, `lr=1e-4`, `checkpoint_dir="checkpoints"`. |
| `TARGETS` | List of `(name, examples)` tuples for the progressive learning phase: NAND and XOR, defined using bitwise operations. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `build_composition` | `(candidate, input_arity, loop_id) -> list[tuple[int, list[int]]]` | Converts a search candidate dict into a storable composition list. Handles all 6 composition types. For **none**: direct function call with routing. For **loop_direct**: `[(loop_id, [body_id, 1, 0])]`. For **loop_binary**: `[(loop_id, [body_id, 1, -1, 0])]` where -1 is literal 0. For **sequential**: two steps -- secondary on inputs, primary on result. For **nested**: one step per input column through secondary, then primary combining all. For **parallel**: primary and secondary on their routes, then tertiary combining both results. |
| `_init_replay_buffer` | `(state)` | **Internal.** Initializes the replay buffer with curriculum tasks if not already present. |
| `_is_duplicate_discovery` | `(state, candidate) -> bool` | **Internal.** Checks if a candidate is structurally identical to an existing discovery in the replay buffer. Compares all fields: primary_id, secondary_id, tertiary_id, comp_type, routing, constants. |
| `learn` | `(conn, state, model, optimizer, name, examples, *, max_search_steps=10, max_depth=3, num_epochs=30, max_retries=2) -> (bool, optimizer, float)` | **The main learning pipeline.** 6 phases: **(1)** Initialize replay buffer and pre-train if first call. **(2)** Search with retries (increasing temperature on each retry). **(3)** Check for duplicate discoveries. **(4)** Simplify the found composition (prune, strip columns). **(5)** Register the new function, resize model heads if vocab grew, rebuild optimizer. **(6)** Consolidate all knowledge via full replay training. Returns `(success, optimizer, r2_score)`. The optimizer is returned because it may be recreated if vocab changes. |
| `curriculum_tasks` | `(state) -> list[dict]` | Generates curriculum tasks: known input/output pairs for each primitive plus composition examples. Each primitive gets a clean task and a noisy variant (with junk columns and routing), teaching the TRM about NULL column elimination. Composition tasks teach sequential (e.g., `INC(MUL(a,b))`), nested (e.g., `ADD(INC(a), INC(b))`), and parallel (e.g., `ADD(INC(a), MUL(b,c))`). Uses fixed random seed (777) for reproducibility. |
| `save_checkpoint` | `(model, optimizer, state)` | Saves model state_dict, optimizer state_dict, and vocab size to `checkpoints/model.pt`. |
| `load_checkpoint` | `(model, optimizer, state) -> bool` | Loads a checkpoint. Handles vocab mismatch: if saved model has fewer functions than current registry, creates a temporary model with old vocab size, loads weights, resizes heads, and transfers. Returns False if checkpoint is stale (saved vocab > current vocab) or doesn't exist. |
| `_fmt_candidate` | `(state, c) -> str` | **Internal.** Formats a candidate as a human-readable string for logging (e.g., `"ADD(MUL(...))"`, `"LOOP(INC, count=b, init=a)"`). |
| `main` | `()` | Entry point. Initializes DB + registry, creates model + optimizer, loads checkpoint, runs curriculum pre-training (20 epochs per task), then progressive learning on TARGETS (NAND, XOR). Saves checkpoint and prints DB summary at the end. |

---

### `experiment_noise.py` -- Noise Column Experiment

End-to-end experiment testing whether NSSR can ignore irrelevant input columns. Communicates with the system via the HTTP API (requires server.py running).

#### 3-Part Experiment Design

**Part 1 -- Dummy Columns (zero signal):** Tests pure random noise columns. SQUARE(x) = x*x with 2 junk columns. FORCE(m,a) = m*a with interleaved junk columns.

**Part 2 -- Noisy/Correlated Columns:** Tests columns with misleading signal. KE(m,v) = 0.5*m*v^2 with a column correlated to m (multicollinearity trap). PE(m,h) = m*9.81*h with junk columns on both sides.

**Part 3 -- Synthesis with Noise:** TOTAL_E(m,v,h) = KE + PE with junk columns, requiring the model to compose previously-learned functions while still ignoring noise.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `api` | `(method, path, json=None) -> dict \| None` | HTTP helper. Makes a request to the server and returns parsed JSON. Prints error on 4xx/5xx. |
| `section` | `(title)` | Prints a section header with `=` bars. |
| `subsection` | `(title)` | Prints a subsection header with `---` bars. |
| `print_functions` | `(funcs)` | Prints a formatted list of functions with id, name, arity, and layer. |
| `create_dataset` | `(name, description, examples) -> dict` | Creates a dataset via POST `/datasets`. Prints first 5 examples. |
| `train_function` | `(target_name, dataset_name, max_depth=5, num_epochs=40, max_search_steps=15) -> dict` | Trains a new function via POST `/train`. Prints result status, R^2, elapsed time, and vocab size. |
| `evaluate` | `(dataset_name) -> dict` | Evaluates the model on a dataset via POST `/test/eval`. Prints exact match count, best R^2, per-function R^2 scores with ASCII bar charts, and per-example details. |
| `junk` | `() -> float` | Returns a random float in `[-100, 100]` with 2 decimal places. |
| `noisy_correlated` | `(val) -> float` | Returns `val + random(-3, 3)` -- a value loosely correlated with the input, creating a multicollinearity trap. |
| `main` | `()` | Runs the full 3-part experiment: preflight check, dummy columns (SQUARE, FORCE), noisy columns (KE, PE), synthesis (TOTAL_E). Prints a final summary with all function compositions and training history. |

---

## Interaction Diagram (Mermaid)

```mermaid
graph TB
    subgraph schema.py ["schema.py -- Database Schema"]
        SCHEMA_FUNCTIONS_TABLE["FUNCTIONS_TABLE<br/>(SQL CREATE TABLE)"]
        SCHEMA_LAYER_INDEX["LAYER_INDEX<br/>(SQL CREATE INDEX)"]
        SCHEMA_ALL_TABLES["ALL_TABLES"]
        SCHEMA_ALL_INDEXES["ALL_INDEXES"]
    end

    subgraph db.py ["db.py -- Database Operations"]
        DB_init_db["init_db()<br/>Create/open DB, apply schema"]
        DB_add_primitive["add_primitive()<br/>Insert primitive function"]
        DB_add_learned["add_learned()<br/>Insert learned function,<br/>auto-calc layer"]
        DB_get_function["get_function()<br/>Get single function row"]
        DB_get_composition["get_composition()<br/>Get composition steps"]
        DB_get_constants["get_constants()<br/>Get constants list"]
        DB_get_const_mode["get_const_mode()<br/>Get const_mode string"]
        DB_get_all_functions["get_all_functions()<br/>All functions by layer"]
        DB_get_functions_by_layer["get_functions_by_layer()<br/>Functions at specific layer"]
        DB_count_functions["count_functions()<br/>Total function count"]
        DB_max_layer["max_layer()<br/>Highest layer number"]
        DB_print_summary["print_summary()<br/>Human-readable DB dump"]
        DB__row_to_dict["_row_to_dict()<br/>Row -> dict with JSON parse"]
    end

    subgraph registry.py ["registry.py -- Function Registry"]
        REG_PRIMITIVES["PRIMITIVES<br/>(module-level dict)"]
        REG__make_primitives["_make_primitives()<br/>Build all primitive defs"]
        REG__loop_factory["_loop_factory()<br/>LOOP primitive factory"]
        REG__while_factory["_while_factory()<br/>WHILE primitive factory"]
        REG__accum_factory["_accum_factory()<br/>ACCUM primitive factory"]
        REG__empty_state["_empty_state()<br/>Blank registry state"]
        REG_execute["execute()<br/>Run function by id"]
        REG_init_registry["init_registry()<br/>Register primitives,<br/>return state"]
        REG_load_registry["load_registry()<br/>Rebuild state from DB"]
        REG_register_learned["register_learned()<br/>Register new composed fn"]
        REG__make_composed_fn["_make_composed_fn()<br/>Build composition closure"]
        REG_vocab_size["vocab_size()<br/>Count of functions"]
        REG_get_name["get_name()<br/>Function name by id"]
        REG_get_names["get_names()<br/>Multiple names by ids"]
    end

    subgraph model.py ["model.py -- TRM Neural Network"]
        MOD_Carry["Carry (dataclass)<br/>y, z, steps, halted"]
        MOD__Block["_Block (nn.Module)<br/>Pre-norm transformer block<br/>LN -> MHA -> Residual -> LN -> FFN -> Residual"]
        MOD_TRM["TRM (nn.Module)<br/>Tiny Recursive Model"]
        MOD_TRM_init["TRM.__init__()<br/>input_proj, pos_emb,<br/>blocks, heads"]
        MOD_TRM_apply_blocks["TRM._apply_blocks()<br/>Run all blocks"]
        MOD_TRM_latent_recursion["TRM._latent_recursion()<br/>n steps: z=net(x+y+z)<br/>1 step: y=net(y+z)"]
        MOD_TRM_deep_recursion["TRM._deep_recursion()<br/>T-1 no_grad + 1 grad pass"]
        MOD_TRM_forward["TRM.forward()<br/>One supervision step:<br/>embed -> deep_recurse -> heads"]
        MOD_create_model["create_model()<br/>Factory with defaults"]
        MOD_fresh_carry["fresh_carry()<br/>Zeroed carry state"]
        MOD_reset_carry["reset_carry()<br/>Zero existing carry"]
        MOD_resize_heads["resize_heads()<br/>Expand output heads<br/>for new vocab"]
    end

    subgraph train.py ["train.py -- Training"]
        TRN_COMP_TYPE_INDEX["COMP_TYPE_INDEX<br/>(string -> int map)"]
        TRN_compute_loss["compute_loss()<br/>Multi-head CE + entropy reg"]
        TRN_train_on_examples["train_on_examples()<br/>Train on single task<br/>with deep supervision"]
        TRN_train_on_replay["train_on_replay()<br/>Full replay: all tasks,<br/>early stopping"]
    end

    subgraph search.py ["search.py -- Program Search"]
        SCH_COMP_TYPES["COMP_TYPES<br/>[none, seq, nested, parallel]"]
        SCH_guided["guided()<br/>Public search entry"]
        SCH__guided_inner["_guided_inner()<br/>Core search loop:<br/>TRM predict -> generate -> validate"]
        SCH__log_trm_step["_log_trm_step()<br/>Log TRM predictions"]
        SCH__log_near_misses["_log_near_misses()<br/>Score & log best misses"]
        SCH__describe_candidate["_describe_candidate()<br/>Human-readable candidate"]
        SCH__null_column_subsets["_null_column_subsets()<br/>Generate kept-column subsets"]
        SCH__find_null_id["_find_null_id()<br/>Find NULL primitive id"]
        SCH__generate_candidates["_generate_candidates()<br/>Build candidates from<br/>top-k predictions (3 depths)"]
        SCH__add_composition_candidates["_add_composition_candidates()<br/>Seq/nested + NULL variants"]
        SCH__add_parallel_candidates["_add_parallel_candidates()<br/>Parallel + routing combos"]
        SCH__try_fit_constants["_try_fit_constants()<br/>Fit scale or offset"]
        SCH__try_fit_any["_try_fit_any()<br/>Try fitting on candidates"]
        SCH__fit_scale["_fit_scale()<br/>Least-squares scale"]
        SCH__fit_offset["_fit_offset()<br/>Constant offset"]
        SCH__r2["_r2()<br/>R-squared metric"]
        SCH__candidate["_candidate()<br/>Construct candidate dict"]
        SCH__complexity_score["_complexity_score()<br/>Simplicity preference"]
        SCH__cand_key["_cand_key()<br/>Hashable dedup key"]
        SCH_format_examples["format_examples()<br/>Encode examples as tensors"]
    end

    subgraph executor.py ["executor.py -- Program Executor"]
        EXE_run["run()<br/>Execute candidate + constants"]
        EXE__run_base["_run_base()<br/>Core composition execution<br/>(6 comp types)"]
        EXE_validate["validate()<br/>Check candidate vs examples"]
        EXE_r_squared["r_squared()<br/>R² for registered function"]
    end

    subgraph simplify.py ["simplify.py -- Composition Simplifier"]
        SIM_simplify["simplify()<br/>Public: try 3 strategies"]
        SIM__try_single_function["_try_single_function()<br/>Strategy 1: existing fn match"]
        SIM__try_prune["_try_prune()<br/>Strategy 2: remove steps"]
        SIM__strip_unused_inputs["_strip_unused_inputs()<br/>Strategy 3: drop unused cols"]
        SIM__validate["_validate()<br/>Test composition step-by-step"]
        SIM__complexity["_complexity()<br/>Complexity score"]
    end

    subgraph main.py ["main.py -- Orchestrator"]
        MAIN_CONFIG["CONFIG<br/>(global hyperparams)"]
        MAIN_TARGETS["TARGETS<br/>[NAND, XOR]"]
        MAIN_build_composition["build_composition()<br/>Candidate -> storable comp"]
        MAIN__init_replay_buffer["_init_replay_buffer()<br/>Init buffer with curriculum"]
        MAIN__is_duplicate_discovery["_is_duplicate_discovery()<br/>Check structural duplicate"]
        MAIN_learn["learn()<br/>Full pipeline: search -><br/>simplify -> register -> train"]
        MAIN_curriculum_tasks["curriculum_tasks()<br/>Generate primitive + comp<br/>curriculum with noise variants"]
        MAIN_save_checkpoint["save_checkpoint()<br/>Save model + optimizer"]
        MAIN_load_checkpoint["load_checkpoint()<br/>Load + handle vocab mismatch"]
        MAIN__fmt_candidate["_fmt_candidate()<br/>Format candidate for logging"]
        MAIN_main["main()<br/>Entry point: init -> pretrain<br/>-> learn NAND, XOR -> save"]
    end

    subgraph experiment_noise.py ["experiment_noise.py -- Noise Experiment"]
        EXP_api["api()<br/>HTTP request helper"]
        EXP_section["section()<br/>Print section header"]
        EXP_subsection["subsection()<br/>Print subsection header"]
        EXP_print_functions["print_functions()<br/>Format function list"]
        EXP_create_dataset["create_dataset()<br/>POST /datasets"]
        EXP_train_function["train_function()<br/>POST /train"]
        EXP_evaluate["evaluate()<br/>POST /test/eval"]
        EXP_junk["junk()<br/>Random noise value"]
        EXP_noisy_correlated["noisy_correlated()<br/>Correlated noise value"]
        EXP_main["main()<br/>Run 3-part experiment"]
    end

    %% === CROSS-FILE DEPENDENCIES ===

    %% schema.py -> db.py
    SCHEMA_ALL_TABLES -->|"imported"| DB_init_db
    SCHEMA_ALL_INDEXES -->|"imported"| DB_init_db

    %% db.py -> registry.py
    DB_add_primitive -->|"called by"| REG_init_registry
    DB_add_learned -->|"called by"| REG_register_learned
    DB_get_all_functions -->|"called by"| REG_load_registry
    DB_get_composition -->|"called by"| REG_load_registry
    DB_get_constants -->|"called by"| REG_load_registry
    DB_get_const_mode -->|"called by"| REG_load_registry

    %% db.py internal
    DB_get_function -->|"called by"| DB_get_composition
    DB_get_function -->|"called by"| DB_get_constants
    DB_get_function -->|"called by"| DB_get_const_mode
    DB_get_function -->|"called by"| DB_print_summary
    DB__row_to_dict -->|"called by"| DB_get_function
    DB__row_to_dict -->|"called by"| DB_get_all_functions
    DB__row_to_dict -->|"called by"| DB_get_functions_by_layer
    DB_max_layer -->|"called by"| DB_print_summary
    DB_get_functions_by_layer -->|"called by"| DB_print_summary

    %% registry.py internal
    REG__make_primitives -->|"builds"| REG_PRIMITIVES
    REG__loop_factory -->|"used by"| REG__make_primitives
    REG__while_factory -->|"used by"| REG__make_primitives
    REG__accum_factory -->|"used by"| REG__make_primitives
    REG__empty_state -->|"called by"| REG_init_registry
    REG__empty_state -->|"called by"| REG_load_registry
    REG__make_composed_fn -->|"called by"| REG_load_registry
    REG__make_composed_fn -->|"called by"| REG_register_learned
    REG_execute -->|"callback into"| REG__loop_factory
    REG_execute -->|"callback into"| REG__while_factory
    REG_execute -->|"callback into"| REG__accum_factory
    REG_execute -->|"callback into"| REG__make_composed_fn

    %% model.py internal
    MOD__Block -->|"used by"| MOD_TRM_init
    MOD_TRM_apply_blocks -->|"calls"| MOD__Block
    MOD_TRM_latent_recursion -->|"calls"| MOD_TRM_apply_blocks
    MOD_TRM_deep_recursion -->|"calls"| MOD_TRM_latent_recursion
    MOD_TRM_forward -->|"calls"| MOD_TRM_deep_recursion
    MOD_TRM_forward -->|"reads"| MOD_Carry
    MOD_create_model -->|"creates"| MOD_TRM
    MOD_fresh_carry -->|"creates"| MOD_Carry

    %% registry.py -> executor.py
    REG_execute -->|"called by"| EXE__run_base
    REG_execute -->|"called by"| EXE_r_squared

    %% executor.py internal
    EXE__run_base -->|"called by"| EXE_run
    EXE_run -->|"called by"| EXE_validate

    %% model.py -> train.py
    MOD_fresh_carry -->|"called by"| TRN_train_on_examples
    MOD_fresh_carry -->|"called by"| TRN_train_on_replay
    MOD_TRM_forward -->|"called by"| TRN_train_on_examples
    MOD_TRM_forward -->|"called by"| TRN_train_on_replay

    %% search.py -> executor.py
    EXE_validate -->|"called by"| SCH__guided_inner
    EXE_run -->|"called by"| SCH__try_fit_constants
    EXE_run -->|"called by"| SCH__log_near_misses
    EXE_run -->|"called by"| SCH__guided_inner

    %% search.py -> model.py
    MOD_fresh_carry -->|"called by"| SCH__guided_inner
    MOD_TRM_forward -->|"called by"| SCH__guided_inner

    %% search.py -> registry.py
    REG_vocab_size -->|"called by"| SCH__guided_inner

    %% search.py -> train.py
    SCH_format_examples -->|"called by"| TRN_train_on_examples

    %% search.py internal
    SCH__guided_inner -->|"called by"| SCH_guided
    SCH__generate_candidates -->|"called by"| SCH__guided_inner
    SCH__null_column_subsets -->|"called by"| SCH__guided_inner
    SCH__find_null_id -->|"called by"| SCH__guided_inner
    SCH__log_trm_step -->|"called by"| SCH__guided_inner
    SCH__log_near_misses -->|"called by"| SCH__guided_inner
    SCH__try_fit_any -->|"called by"| SCH__guided_inner
    SCH__cand_key -->|"called by"| SCH__guided_inner
    SCH__complexity_score -->|"called by"| SCH__guided_inner
    SCH__add_composition_candidates -->|"called by"| SCH__generate_candidates
    SCH__add_parallel_candidates -->|"called by"| SCH__generate_candidates
    SCH__candidate -->|"called by"| SCH__generate_candidates
    SCH__candidate -->|"called by"| SCH__add_composition_candidates
    SCH__candidate -->|"called by"| SCH__add_parallel_candidates
    SCH__try_fit_constants -->|"called by"| SCH__try_fit_any
    SCH__fit_scale -->|"called by"| SCH__try_fit_constants
    SCH__fit_offset -->|"called by"| SCH__try_fit_constants
    SCH__r2 -->|"called by"| SCH__try_fit_constants
    SCH__r2 -->|"called by"| SCH__guided_inner
    SCH__r2 -->|"called by"| SCH__log_near_misses
    SCH__describe_candidate -->|"called by"| SCH__log_near_misses

    %% simplify.py -> registry.py
    REG_execute -->|"called by"| SIM__try_single_function
    REG_execute -->|"called by"| SIM__validate

    %% simplify.py internal
    SIM__try_single_function -->|"called by"| SIM_simplify
    SIM__try_prune -->|"called by"| SIM_simplify
    SIM__strip_unused_inputs -->|"called by"| SIM_simplify
    SIM__validate -->|"called by"| SIM__try_prune
    SIM__validate -->|"called by"| SIM__strip_unused_inputs
    SIM__complexity -->|"called by"| SIM_simplify

    %% main.py -> all modules
    DB_init_db -->|"called by"| MAIN_main
    DB_print_summary -->|"called by"| MAIN_main
    REG_init_registry -->|"called by"| MAIN_main
    REG_register_learned -->|"called by"| MAIN_learn
    REG_vocab_size -->|"called by"| MAIN_main
    REG_vocab_size -->|"called by"| MAIN_learn
    REG_get_name -->|"called by"| MAIN__fmt_candidate
    REG_get_names -->|"called by"| MAIN_main
    MOD_create_model -->|"called by"| MAIN_main
    MOD_create_model -->|"called by"| MAIN_load_checkpoint
    MOD_resize_heads -->|"called by"| MAIN_learn
    MOD_resize_heads -->|"called by"| MAIN_load_checkpoint
    MOD_fresh_carry -->|"called by"| MAIN_learn
    SCH_guided -->|"called by"| MAIN_learn
    SCH_format_examples -->|"called by"| MAIN_learn
    TRN_train_on_examples -->|"called by"| MAIN_main
    TRN_train_on_replay -->|"called by"| MAIN_learn
    SIM_simplify -->|"called by"| MAIN_learn
    EXE_r_squared -->|"called by"| MAIN_learn

    %% main.py internal
    MAIN_build_composition -->|"called by"| MAIN_learn
    MAIN__init_replay_buffer -->|"called by"| MAIN_learn
    MAIN__is_duplicate_discovery -->|"called by"| MAIN_learn
    MAIN_curriculum_tasks -->|"called by"| MAIN__init_replay_buffer
    MAIN_curriculum_tasks -->|"called by"| MAIN_main
    MAIN_save_checkpoint -->|"called by"| MAIN_main
    MAIN_load_checkpoint -->|"called by"| MAIN_main
    MAIN__fmt_candidate -->|"called by"| MAIN_learn
    MAIN_learn -->|"called by"| MAIN_main

    %% experiment_noise.py internal
    EXP_api -->|"called by"| EXP_create_dataset
    EXP_api -->|"called by"| EXP_train_function
    EXP_api -->|"called by"| EXP_evaluate
    EXP_api -->|"called by"| EXP_main
    EXP_section -->|"called by"| EXP_main
    EXP_subsection -->|"called by"| EXP_main
    EXP_print_functions -->|"called by"| EXP_main
    EXP_create_dataset -->|"called by"| EXP_main
    EXP_train_function -->|"called by"| EXP_main
    EXP_evaluate -->|"called by"| EXP_main
    EXP_junk -->|"called by"| EXP_main
    EXP_noisy_correlated -->|"called by"| EXP_main

    %% experiment_noise.py -> server (external)
    EXP_api -.->|"HTTP to server.py"| SERVER["server.py<br/>(not documented)"]

    %% Styling
    classDef schemaStyle fill:#e8f5e9,stroke:#388e3c
    classDef dbStyle fill:#e3f2fd,stroke:#1976d2
    classDef regStyle fill:#fff3e0,stroke:#f57c00
    classDef modelStyle fill:#fce4ec,stroke:#c62828
    classDef trainStyle fill:#f3e5f5,stroke:#7b1fa2
    classDef searchStyle fill:#e0f7fa,stroke:#0097a7
    classDef execStyle fill:#fff8e1,stroke:#f9a825
    classDef simpStyle fill:#f1f8e9,stroke:#689f38
    classDef mainStyle fill:#e8eaf6,stroke:#3f51b5
    classDef expStyle fill:#fafafa,stroke:#616161

    class SCHEMA_FUNCTIONS_TABLE,SCHEMA_LAYER_INDEX,SCHEMA_ALL_TABLES,SCHEMA_ALL_INDEXES schemaStyle
    class DB_init_db,DB_add_primitive,DB_add_learned,DB_get_function,DB_get_composition,DB_get_constants,DB_get_const_mode,DB_get_all_functions,DB_get_functions_by_layer,DB_count_functions,DB_max_layer,DB_print_summary,DB__row_to_dict dbStyle
    class REG_PRIMITIVES,REG__make_primitives,REG__loop_factory,REG__while_factory,REG__accum_factory,REG__empty_state,REG_execute,REG_init_registry,REG_load_registry,REG_register_learned,REG__make_composed_fn,REG_vocab_size,REG_get_name,REG_get_names regStyle
    class MOD_Carry,MOD__Block,MOD_TRM,MOD_TRM_init,MOD_TRM_apply_blocks,MOD_TRM_latent_recursion,MOD_TRM_deep_recursion,MOD_TRM_forward,MOD_create_model,MOD_fresh_carry,MOD_reset_carry,MOD_resize_heads modelStyle
    class TRN_COMP_TYPE_INDEX,TRN_compute_loss,TRN_train_on_examples,TRN_train_on_replay trainStyle
    class SCH_COMP_TYPES,SCH_guided,SCH__guided_inner,SCH__log_trm_step,SCH__log_near_misses,SCH__describe_candidate,SCH__null_column_subsets,SCH__find_null_id,SCH__generate_candidates,SCH__add_composition_candidates,SCH__add_parallel_candidates,SCH__try_fit_constants,SCH__try_fit_any,SCH__fit_scale,SCH__fit_offset,SCH__r2,SCH__candidate,SCH__complexity_score,SCH__cand_key,SCH_format_examples searchStyle
    class EXE_run,EXE__run_base,EXE_validate,EXE_r_squared execStyle
    class SIM_simplify,SIM__try_single_function,SIM__try_prune,SIM__strip_unused_inputs,SIM__validate,SIM__complexity simpStyle
    class MAIN_CONFIG,MAIN_TARGETS,MAIN_build_composition,MAIN__init_replay_buffer,MAIN__is_duplicate_discovery,MAIN_learn,MAIN_curriculum_tasks,MAIN_save_checkpoint,MAIN_load_checkpoint,MAIN__fmt_candidate,MAIN_main mainStyle
    class EXP_api,EXP_section,EXP_subsection,EXP_print_functions,EXP_create_dataset,EXP_train_function,EXP_evaluate,EXP_junk,EXP_noisy_correlated,EXP_main expStyle
```
