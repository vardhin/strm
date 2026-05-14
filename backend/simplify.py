"""
Composition simplifier for NSRR.

After search finds a working composition, try to make it shorter
before registering it as a new function.

Strategies (tried in order):
  1. Replace with a single existing function
  2. Prune redundant intermediate steps
"""

import math
from typing import Any

import registry as reg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify(state: dict, composition: list[tuple[int, list[int]]],
             examples: list[tuple[list[int], Any]],
             constants: list[float] | None = None,
             const_mode: str = "multiplicative"
             ) -> tuple[list[tuple[int, list[int]]], int, list[int] | None]:
    """Return (composition, effective_arity, used_cols).

    used_cols is the list of original column indices that were kept, or None
    if all columns are used (no stripping needed).
    """
    input_arity = len(examples[0][0]) if examples else 0
    best = composition

    # Strategy 1: single existing function (only if no constants — constants
    # make the function unique even if the base composition matches)
    if not constants:
        single = _try_single_function(state, examples)
        if single is not None and _complexity(state, single) < _complexity(state, best):
            return single, input_arity, None

    # Strategy 2: prune redundant steps
    pruned = _try_prune(state, best, examples, constants, const_mode)
    if pruned is not None and _complexity(state, pruned) < _complexity(state, best):
        best = pruned

    # Strategy 3: strip unused input columns
    stripped, new_arity, used_cols = _strip_unused_inputs(state, best, examples, constants, const_mode)
    return stripped, new_arity, used_cols


# ---------------------------------------------------------------------------
# Strategy 1: single function replacement
# ---------------------------------------------------------------------------

def _try_single_function(state: dict, examples: list[tuple[list[int], Any]],
                         ) -> list[tuple[int, list[int]]] | None:
    """Check if any existing non-primitive function already solves this."""
    input_arity = len(examples[0][0])

    for fid, meta in state["metadata"].items():
        if meta["layer"] == 0:
            continue
        if meta["arity"] > input_arity:
            continue
        try:
            if all(math.isclose(reg.execute(state, fid, inp), exp, rel_tol=1e-6, abs_tol=1e-9)
                   for inp, exp in examples):
                return [(fid, list(range(input_arity)))]
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Strategy 2: prune redundant steps
# ---------------------------------------------------------------------------

def _try_prune(state: dict, composition: list[tuple[int, list[int]]],
               examples: list[tuple[list[int], Any]],
               constants: list[float] | None = None,
               const_mode: str = "multiplicative",
               ) -> list[tuple[int, list[int]]] | None:
    """Try removing each non-final step. Return first shorter form that works."""
    if len(composition) <= 1:
        return None

    input_arity = len(examples[0][0])

    for i in range(len(composition) - 1):
        removed_output_idx = input_arity + i

        # Build candidate without step i
        shortened = []
        valid = True
        for j, (fid, args) in enumerate(composition):
            if j == i:
                continue
            # Adjust arg indices that referenced the removed step or later
            new_args = []
            for idx in args:
                if idx == removed_output_idx:
                    valid = False
                    break
                elif idx > removed_output_idx:
                    new_args.append(idx - 1)
                else:
                    new_args.append(idx)
            if not valid:
                break
            shortened.append((fid, new_args))

        if not valid:
            continue

        if _validate(state, shortened, examples, constants, const_mode):
            return shortened

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(state: dict, composition: list[tuple[int, list[int]]],
              examples: list[tuple[list[int], Any]],
              constants: list[float] | None = None,
              const_mode: str = "multiplicative") -> bool:
    """Test a composition against all examples by running step-by-step."""
    for inputs, expected in examples:
        try:
            available = list(inputs)
            for child_id, args in composition:
                step_inputs = [available[i] for i in args]
                result = reg.execute(state, child_id, step_inputs)
                available.append(result)
            result = available[-1]
            if constants:
                if const_mode == "additive":
                    result = result + constants[0]
                else:
                    result = result * constants[0]
            if not math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-9):
                return False
        except Exception:
            return False
    return True


def _strip_unused_inputs(state: dict, composition: list[tuple[int, list[int]]],
                         examples: list[tuple[list[int], Any]],
                         constants: list[float] | None = None,
                         const_mode: str = "multiplicative",
                         ) -> tuple[list[tuple[int, list[int]]], int, list[int] | None]:
    """Remove input columns that no composition step references.

    If a function's composition only uses inputs at positions [0, 2] out of
    [0, 1, 2], remap to use [0, 1] and reduce arity from 3 to 2. This makes
    the function composable without needing to route junk columns.

    Returns (composition, effective_arity, used_cols).
    used_cols is None if all columns are used.
    """
    input_arity = len(examples[0][0]) if examples else 0
    if input_arity <= 1:
        return composition, input_arity, None

    # Find which original input positions are actually referenced
    used_input_cols = set()
    for _, args in composition:
        for idx in args:
            if 0 <= idx < input_arity:
                used_input_cols.add(idx)
            # -1 is literal 0 (used by LOOP), not a column reference

    if len(used_input_cols) == input_arity:
        return composition, input_arity, None  # all columns used, nothing to strip

    # Build remapping: old_col -> new_col
    used_sorted = sorted(used_input_cols)
    remap = {old: new for new, old in enumerate(used_sorted)}
    n_stripped = len(used_sorted)

    # Remap all arg indices
    new_composition = []
    for fid, args in composition:
        new_args = []
        for idx in args:
            if idx < 0:
                new_args.append(idx)  # literal (e.g. -1 for 0)
            elif idx < input_arity:
                new_args.append(remap[idx])
            else:
                # References a prior step's output — shift down
                new_args.append(idx - (input_arity - n_stripped))
        new_composition.append((fid, new_args))

    # Validate the stripped composition against remapped examples
    stripped_examples = [
        ([inp[c] for c in used_sorted], exp)
        for inp, exp in examples
    ]
    if _validate(state, new_composition, stripped_examples, constants, const_mode):
        print(f"    [simplify] stripped {input_arity} -> {n_stripped} inputs (used cols: {used_sorted})")
        return new_composition, n_stripped, used_sorted

    return composition, input_arity, None  # fallback if validation fails


def _complexity(state: dict, composition: list[tuple[int, list[int]]]) -> int:
    """Lower is better. Heavily penalizes more steps."""
    n_terms = len(composition)
    max_layer = max((state["metadata"][cid]["layer"] for cid, _ in composition), default=0)
    total_args = sum(len(args) for _, args in composition)
    return n_terms * 100 + max_layer * 10 + total_args
