"""
Program search for NSSR.

Strategy: TRM-guided search with NULL-based column elimination.
The TRM predicts likely function compositions. Junk columns are handled
by generating candidates that NULL out subsets of columns — the executor
skips NULL'd columns and routes the remaining ones into functions
(with same-column repetition allowed).

Returns a candidate dict or None.
"""

import math
from itertools import product
import torch
from typing import Any

import registry as reg
import executor as exe
from model import Carry, fresh_carry


COMP_TYPES = ["none", "sequential", "nested", "parallel"]


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def guided(state: dict, model, examples: list[tuple[list[int], Any]],
           x_input: torch.Tensor, *, max_steps: int = 10,
           max_depth: int = 3, temperature_boost: float = 0.0) -> dict | None:
    """Use the TRM to predict likely compositions, then validate them."""
    result, _ = _guided_inner(state, model, examples, x_input,
                              max_steps=max_steps, max_depth=max_depth,
                              temperature_boost=temperature_boost)
    return result


def _guided_inner(state: dict, model, examples: list[tuple[list[int], Any]],
                  x_input: torch.Tensor, *, max_steps: int = 10,
                  max_depth: int = 3,
                  temperature_boost: float = 0.0) -> tuple[dict | None, float]:
    """Core search loop.

    Returns (candidate, best_near_miss_r2). candidate is None if search failed.

    Uses holdout validation and simplicity preference.
    """
    model.eval()

    batch_size, seq_len, _ = x_input.shape
    carry = fresh_carry(batch_size, seq_len, model.d_model)

    tried: set[tuple] = set()
    n_functions = reg.vocab_size(state)
    near_misses = []
    input_arity = len(examples[0][0])
    null_id = _find_null_id(state)

    # Holdout split
    import random as _rng
    shuffled = list(examples)
    _rng.seed(12345)
    _rng.shuffle(shuffled)
    if len(examples) >= 5:
        n_holdout = max(2, len(examples) // 3)
        train_examples = shuffled[:-n_holdout]
        holdout_examples = shuffled[-n_holdout:]
    else:
        train_examples = examples
        holdout_examples = examples

    # Pre-compute NULL column subsets: all ways to drop 0..N-1 columns
    null_subsets = _null_column_subsets(input_arity)

    for step in range(max_steps):
        carry, outputs = model(carry, x_input)

        temperature = max(2.0 - step * 0.05, 0.5) + temperature_boost
        logits = {
            k: outputs[k].detach().mean(dim=0)[:n_functions] / temperature
            for k in ("primary_logits", "secondary_logits", "tertiary_logits")
        }
        comp_logits = outputs["composition_logits"].detach().mean(dim=0) / temperature

        # Top-k predictions (grows with step for diversity, wider on retries)
        top_k = min(3 + step + int(temperature_boost * 2), n_functions)
        tops = {k: torch.topk(v, min(top_k, len(v))).indices for k, v in logits.items()}
        comp_top = torch.topk(comp_logits, min(len(COMP_TYPES), len(comp_logits))).indices

        # Log TRM thought process
        _log_trm_step(state, step, logits, comp_logits, n_functions)

        # Generate candidates (with NULL column variants)
        candidates = _generate_candidates(state, tops, comp_top, max_depth,
                                          input_arity, null_subsets, null_id)

        new_count = 0
        valid_candidates = []
        for cand in candidates:
            key = _cand_key(cand)
            if key in tried:
                continue
            tried.add(key)
            new_count += 1

            if exe.validate(state, cand, train_examples) and \
               exe.validate(state, cand, holdout_examples):
                valid_candidates.append(cand)
            else:
                near_misses.append(cand)

        # If we found valid candidates, return the simplest one
        if valid_candidates:
            best = min(valid_candidates, key=_complexity_score)
            return best, 1.0

        # Incremental constant fitting
        step_misses = near_misses[-new_count:] if new_count > 0 else []
        fitted = _try_fit_any(state, step_misses[:50], train_examples)
        if fitted is not None and exe.validate(state, fitted, holdout_examples):
            return fitted, 1.0

        # Small noise on carry for exploration
        with torch.no_grad():
            carry.y = carry.y + torch.randn_like(carry.y) * 0.01
            carry.z = carry.z + torch.randn_like(carry.z) * 0.01

    # Final pass: constant fitting on near misses
    fitted = _try_fit_any(state, near_misses[:200], train_examples)
    if fitted is not None and exe.validate(state, fitted, holdout_examples):
        return fitted, 1.0

    # Score best near miss
    best_r2 = -1.0
    if near_misses:
        _log_near_misses(state, near_misses, examples)
        for cand in near_misses:
            produced, expected = [], []
            for inputs, exp in examples:
                try:
                    got = exe.run(state, cand, inputs)
                    produced.append(float(got))
                    expected.append(float(exp))
                except Exception:
                    break
            if len(produced) == len(examples):
                best_r2 = max(best_r2, _r2(expected, produced))

    return None, best_r2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_trm_step(state: dict, step: int, logits: dict,
                  comp_logits: torch.Tensor, n_functions: int):
    """Log what the TRM is thinking at each search step."""
    def _top_names(key, k=5):
        vals, idxs = torch.topk(logits[key], min(k, len(logits[key])))
        probs = torch.softmax(logits[key], dim=0)
        parts = []
        for v, i in zip(vals, idxs):
            name = state["metadata"].get(i.item(), {}).get("name", f"#{i.item()}")
            p = probs[i].item()
            parts.append(f"{name}({p:.2f})")
        return ", ".join(parts)

    comp_probs = torch.softmax(comp_logits, dim=0)
    comp_ranked = torch.argsort(comp_probs, descending=True)
    comp_parts = []
    for i in comp_ranked:
        if i.item() < len(COMP_TYPES):
            comp_parts.append(f"{COMP_TYPES[i.item()]}({comp_probs[i].item():.2f})")
    comp_str = ", ".join(comp_parts)

    print(f"    [step {step}] primary:   {_top_names('primary_logits')}")
    print(f"    [step {step}] secondary: {_top_names('secondary_logits')}")
    print(f"    [step {step}] tertiary:  {_top_names('tertiary_logits')}")
    print(f"    [step {step}] comp:      {comp_str}")


def _log_near_misses(state: dict, near_misses: list[dict],
                     examples: list[tuple[list, Any]], top_n: int = 5):
    """Score and log the best near misses from the TRM search."""
    scored = []
    for cand in near_misses:
        produced = []
        expected = []
        for inputs, exp in examples:
            try:
                got = exe.run(state, cand, inputs)
                produced.append(float(got))
                expected.append(float(exp))
            except Exception:
                break
        if len(produced) == len(examples):
            r2 = _r2(expected, produced)
            scored.append((r2, cand))

    scored.sort(key=lambda x: -x[0])
    best = scored[:top_n]

    if not best:
        return

    print(f"    [near misses] {len(scored)} candidates scored, top {len(best)}:")
    for r2, cand in best:
        print(f"      R²={r2:.6f}  {_describe_candidate(state, cand)}")


def _describe_candidate(state: dict, cand: dict) -> str:
    """Human-readable description of a candidate."""
    pid = cand["primary_id"]
    sid = cand.get("secondary_id")
    tid = cand.get("tertiary_id")
    comp = cand["comp_type"]
    routing = cand.get("routing")
    null_cols = cand.get("null_columns")

    p_name = state["metadata"].get(pid, {}).get("name", f"#{pid}")
    s_name = state["metadata"].get(sid, {}).get("name", f"#{sid}") if sid is not None else None
    t_name = state["metadata"].get(tid, {}).get("name", f"#{tid}") if tid is not None else None

    routing_str = ""
    if routing:
        routing_str = f"  route={routing}"
    if null_cols:
        routing_str += f"  null={null_cols}"

    consts = cand.get("constants")
    const_str = ""
    if consts:
        mode = cand.get("const_mode", "multiplicative")
        const_str = f"  [k={consts}, {mode}]"

    if comp == "none":
        return f"{p_name}{routing_str}{const_str}"
    elif comp == "sequential":
        return f"{p_name}({s_name}(...)){routing_str}{const_str}"
    elif comp == "nested":
        return f"{p_name}({s_name}(each)){routing_str}{const_str}"
    elif comp == "parallel":
        return f"{t_name}({p_name}(...), {s_name}(...)){routing_str}{const_str}"
    else:
        return f"{comp}({p_name}, {s_name}){routing_str}{const_str}"


# ---------------------------------------------------------------------------
# NULL column subsets
# ---------------------------------------------------------------------------

def _null_column_subsets(input_arity: int, max_null: int = 0) -> list[list[int]]:
    """Generate all useful subsets of columns to keep (others get NULL'd).

    Returns list of kept-column lists. Always includes "keep all" and
    single-column subsets. For 2+ columns, includes all pairs.
    Max NULL columns defaults to input_arity - 1 (keep at least 1).
    """
    if max_null == 0:
        max_null = input_arity - 1

    all_cols = list(range(input_arity))
    subsets = [all_cols]  # keep all (no NULLs)

    if input_arity <= 1:
        return subsets

    # Single columns (NULL everything else)
    for c in range(input_arity):
        subset = [c]
        if subset not in subsets:
            subsets.append(subset)

    # Pairs
    if input_arity >= 3:
        from itertools import combinations
        for pair in combinations(range(input_arity), 2):
            subset = list(pair)
            if subset not in subsets:
                subsets.append(subset)

    # Triples (for 4+ column inputs)
    if input_arity >= 4:
        from itertools import combinations
        for triple in combinations(range(input_arity), 3):
            subset = list(triple)
            if subset not in subsets:
                subsets.append(subset)

    return subsets


def _find_null_id(state: dict) -> int | None:
    """Find the NULL primitive's function id."""
    for fid, meta in state["metadata"].items():
        if meta["name"] == "NULL":
            return fid
    return None


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(state: dict, tops: dict, comp_top: torch.Tensor,
                         max_depth: int, input_arity: int,
                         null_subsets: list[list[int]],
                         null_id: int | None) -> list[dict]:
    """Build candidate dicts from top-k TRM predictions.

    For each function/composition, tries all NULL column subsets.
    Remaining (non-NULL) columns get routed into functions with
    same-column repetition allowed.
    """
    candidates = []
    loop_id = state["loop_id"]

    primary_ids = tops["primary_logits"].tolist()
    secondary_ids = tops["secondary_logits"].tolist()
    tertiary_ids = tops["tertiary_logits"].tolist()
    comp_types_predicted = [COMP_TYPES[i] for i in comp_top.tolist() if i < len(COMP_TYPES)]

    # Filter out NULL from function predictions (it's not used as a composition function)
    if null_id is not None:
        primary_ids = [x for x in primary_ids if x != null_id]
        secondary_ids = [x for x in secondary_ids if x != null_id]
        tertiary_ids = [x for x in tertiary_ids if x != null_id]

    # Depth 1: single functions with NULL column variants
    for pid in primary_ids:
        meta = state["metadata"].get(pid)
        if meta is None:
            continue
        arity = meta["arity"]

        for kept in null_subsets:
            null_cols = [c for c in range(input_arity) if c not in kept]
            # Route kept columns into function slots (with repetition)
            if len(kept) == arity:
                candidates.append(_candidate(pid, routing=[kept],
                                             null_columns=null_cols or None))
            elif len(kept) > 0 and arity > 0:
                # Need to fill `arity` slots from `kept` columns
                for combo in product(kept, repeat=arity):
                    candidates.append(_candidate(pid, routing=[list(combo)],
                                                 null_columns=null_cols or None))

    if max_depth < 2:
        return candidates

    # Depth 2: binary compositions with NULL column variants
    for comp in comp_types_predicted:
        if comp == "none":
            continue
        for pid in primary_ids:
            if pid == loop_id:
                continue
            for sid in secondary_ids:
                if sid == loop_id:
                    continue

                if comp == "parallel":
                    for tid in tertiary_ids:
                        if tid == loop_id:
                            continue
                        _add_parallel_candidates(
                            candidates, state, pid, sid, tid,
                            input_arity, comp, null_subsets)
                else:
                    _add_composition_candidates(
                        candidates, state, pid, sid, comp,
                        input_arity, null_subsets)

    # Depth 2: LOOP candidates
    if loop_id is not None:
        for body_id in primary_ids + secondary_ids:
            if body_id == loop_id:
                continue
            meta = state["metadata"].get(body_id)
            if meta and meta["arity"] == 1:
                candidates.append(_candidate(loop_id, body_id, comp_type="loop_direct"))
            if meta and meta["arity"] == 2:
                candidates.append(_candidate(loop_id, body_id, comp_type="loop_binary"))

    if max_depth < 3:
        return candidates

    # Depth 3: use ALL learned functions (not just TRM-predicted ones,
    # since TRM may not know them well enough yet to predict them)
    learned = [fid for fid, meta in state["metadata"].items()
               if meta.get("layer", 0) > 0]

    if loop_id is not None:
        for lid in learned:
            meta = state["metadata"].get(lid)
            if meta and meta["arity"] == 1:
                candidates.append(_candidate(loop_id, lid, comp_type="loop_direct"))
            if meta and meta["arity"] == 2:
                candidates.append(_candidate(loop_id, lid, comp_type="loop_binary"))

    for comp in comp_types_predicted:
        if comp == "none":
            continue
        for lid in learned:
            if comp == "parallel":
                for pid in primary_ids:
                    if pid == loop_id:
                        continue
                    for tid in tertiary_ids:
                        _add_parallel_candidates(
                            candidates, state, lid, pid, tid,
                            input_arity, comp, null_subsets)
            else:
                for pid in primary_ids:
                    if pid == loop_id:
                        continue
                    _add_composition_candidates(
                        candidates, state, pid, lid, comp,
                        input_arity, null_subsets)

    return candidates


def _add_composition_candidates(candidates: list, state: dict,
                                pid: int, sid: int, comp: str,
                                input_arity: int,
                                null_subsets: list[list[int]]):
    """Add sequential/nested candidates with NULL column variants."""
    meta_p = state["metadata"].get(pid)
    meta_s = state["metadata"].get(sid)
    if meta_p is None or meta_s is None:
        return
    if meta_p["arity"] < 0 or meta_s["arity"] < 0:
        return

    # Default: no NULL, all columns used
    candidates.append(_candidate(pid, sid, comp_type=comp))

    if input_arity <= 1:
        return

    for kept in null_subsets:
        if len(kept) == input_arity:
            continue  # already added as default
        if len(kept) == 0:
            continue
        null_cols = [c for c in range(input_arity) if c not in kept]

        if comp == "sequential":
            # secondary takes the kept columns, primary takes the result
            arity_s = meta_s["arity"]
            if arity_s > 0:
                for combo in product(kept, repeat=arity_s):
                    candidates.append(_candidate(pid, sid, comp_type=comp,
                                                 routing=[list(combo)],
                                                 null_columns=null_cols))
        elif comp == "nested":
            # primary(secondary(x1), secondary(x2), ...) for each kept col
            candidates.append(_candidate(pid, sid, comp_type=comp,
                                         routing=[kept],
                                         null_columns=null_cols))


def _add_parallel_candidates(candidates: list, state: dict,
                             pid: int, sid: int, tid: int,
                             input_arity: int, comp: str,
                             null_subsets: list[list[int]],
                             max_routings: int = 30):
    """Add parallel candidates with NULL column variants."""
    meta_p = state["metadata"].get(pid)
    meta_s = state["metadata"].get(sid)
    if meta_p is None or meta_s is None:
        return

    arity_p = meta_p["arity"]
    arity_s = meta_s["arity"]
    if arity_p < 0 or arity_s < 0:
        return
    seen = set()

    for kept in null_subsets:
        if len(kept) == 0:
            continue
        null_cols = [c for c in range(input_arity) if c not in kept]

        for combo_p in product(kept, repeat=arity_p):
            for combo_s in product(kept, repeat=arity_s):
                key = (combo_p, combo_s, tuple(null_cols) if null_cols else ())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_candidate(
                    pid, sid, tid, comp_type=comp,
                    routing=[list(combo_p), list(combo_s)],
                    null_columns=null_cols or None))
                if len(seen) >= max_routings:
                    return


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def _try_fit_constants(state: dict, cand: dict,
                       examples: list[tuple[list, Any]],
                       r2_threshold: float = 0.999) -> dict | None:
    """Try fitting a single multiplicative or additive constant."""
    produced = []
    expected = []
    for inputs, exp in examples:
        try:
            got = exe.run(state, cand, inputs)
            produced.append(float(got))
            expected.append(float(exp))
        except Exception:
            return None

    if not produced:
        return None

    k = _fit_scale(produced, expected)
    if k is not None:
        r2 = _r2(expected, [k * p for p in produced])
        if r2 > r2_threshold:
            return {**cand, "constants": [k]}

    k = _fit_offset(produced, expected)
    if k is not None:
        r2 = _r2(expected, [p + k for p in produced])
        if r2 > r2_threshold:
            return {**cand, "constants": [k], "const_mode": "additive"}

    return None


def _try_fit_any(state: dict, candidates: list[dict],
                  examples: list[tuple[list, Any]]) -> dict | None:
    """Try constant-fitting on a list of candidates."""
    for cand in candidates:
        fitted = _try_fit_constants(state, cand, examples)
        if fitted is not None:
            return fitted
    return None


def _fit_scale(produced: list[float], expected: list[float]) -> float | None:
    num = sum(p * e for p, e in zip(produced, expected))
    den = sum(p * p for p in produced)
    if abs(den) < 1e-15:
        return None
    return num / den


def _fit_offset(produced: list[float], expected: list[float]) -> float | None:
    diffs = [e - p for e, p in zip(produced, expected)]
    mean_diff = sum(diffs) / len(diffs)
    if all(abs(d - mean_diff) < 1e-6 for d in diffs):
        return mean_diff
    return None


def _r2(actual: list[float], predicted: list[float]) -> float:
    mean_a = sum(actual) / len(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -float("inf")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(primary_id: int, secondary_id: int | None = None,
               tertiary_id: int | None = None, *,
               comp_type: str = "none",
               routing: list[list[int]] | None = None,
               null_columns: list[int] | None = None) -> dict:
    return {
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "tertiary_id": tertiary_id,
        "comp_type": comp_type,
        "routing": routing,
        "null_columns": null_columns,
    }


def _complexity_score(c: dict) -> int:
    """Lower = simpler = preferred."""
    comp_scores = {"none": 0, "sequential": 1, "nested": 1,
                   "parallel": 2, "loop_direct": 3, "loop_binary": 3}
    score = comp_scores.get(c["comp_type"], 2)
    if c.get("secondary_id") is not None:
        score += 1
    if c.get("tertiary_id") is not None:
        score += 1
    if c.get("constants"):
        score += 1
    return score


def _cand_key(c: dict) -> tuple:
    routing = c.get("routing")
    routing_key = tuple(tuple(r) for r in routing) if routing else None
    null_key = tuple(c.get("null_columns") or [])
    return (c["primary_id"], c["secondary_id"], c.get("tertiary_id"),
            c["comp_type"], routing_key, null_key)


def format_examples(examples: list[tuple[list, Any]], *,
                    input_dim: int, seq_len: int) -> torch.Tensor:
    """Encode examples as float vectors for the TRM."""
    batch_size = len(examples)
    data = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32)

    for b, (inputs, _) in enumerate(examples):
        for pos in range(min(len(inputs), seq_len)):
            val = float(inputs[pos])
            data[b, pos, 0] = val / 100.0
            data[b, pos, 1] = 1.0 if val >= 0 else -1.0
            data[b, pos, 2] = torch.log1p(torch.tensor(abs(val))).item()
            data[b, pos, 3] = val - int(val) if val >= 0 else -(abs(val) - int(abs(val)))

    return data
