"""
Composition simplifier for NSSR.

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
             const_mode: str = "multiplicative") -> list[tuple[int, list[int]]]:
    """Return the simplest equivalent composition."""
    best = composition

    # Strategy 1: single existing function (only if no constants — constants
    # make the function unique even if the base composition matches)
    if not constants:
        single = _try_single_function(state, examples)
        if single is not None and _complexity(state, single) < _complexity(state, best):
            return single

    # Strategy 2: prune redundant steps
    pruned = _try_prune(state, best, examples, constants, const_mode)
    if pruned is not None and _complexity(state, pruned) < _complexity(state, best):
        best = pruned

    return best


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
        if meta["arity"] != input_arity:
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


def _complexity(state: dict, composition: list[tuple[int, list[int]]]) -> int:
    """Lower is better. Heavily penalizes more steps."""
    n_terms = len(composition)
    max_layer = max((state["metadata"][cid]["layer"] for cid, _ in composition), default=0)
    total_args = sum(len(args) for _, args in composition)
    return n_terms * 100 + max_layer * 10 + total_args
