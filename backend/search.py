"""
Program search for NSSR.

Strategy: TRM-guided search with correlation-based column routing.
The model predicts likely function compositions. Correlation heuristics
identify which input columns actually matter. Constant fitting runs
incrementally on promising near-misses.

Returns a candidate dict or None.
"""

import math
from itertools import combinations, combinations_with_replacement, permutations, product
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
           max_depth: int = 3) -> dict | None:
    """Use the TRM to predict likely compositions, then validate them.

    First prunes junk columns (low correlation with output), then runs
    TRM-guided search on the cleaned inputs. Constant fitting is attempted
    incrementally on near-misses.
    """
    model.eval()
    input_arity = len(examples[0][0])

    # Prune junk columns: only keep columns with meaningful correlation
    kept_cols = _prune_columns(examples, input_arity)
    if len(kept_cols) < input_arity:
        print(f"    [routing] pruned {input_arity} -> {len(kept_cols)} cols: {kept_cols}")
        pruned_examples = [
            ([inp[c] for c in kept_cols], out)
            for inp, out in examples
        ]
        # Re-encode for the TRM
        pruned_x = format_examples(pruned_examples,
                                   input_dim=x_input.shape[-1],
                                   seq_len=x_input.shape[1])
        # Search on pruned inputs
        result = _guided_inner(state, model, pruned_examples, pruned_x,
                               max_steps=max_steps, max_depth=max_depth)
        if result is not None:
            # Remap routing back to original column indices
            return _remap_candidate(result, kept_cols)

        # If pruned search failed, fall through to full search
        print(f"    [routing] pruned search failed, trying full input")

    return _guided_inner(state, model, examples, x_input,
                         max_steps=max_steps, max_depth=max_depth)


def _guided_inner(state: dict, model, examples: list[tuple[list[int], Any]],
                  x_input: torch.Tensor, *, max_steps: int = 10,
                  max_depth: int = 3) -> dict | None:
    """Core search loop on (possibly pruned) examples."""
    model.eval()

    batch_size, seq_len, _ = x_input.shape
    carry = fresh_carry(batch_size, seq_len, model.d_model)

    tried: set[tuple] = set()
    n_functions = reg.vocab_size(state)
    near_misses = []
    input_arity = len(examples[0][0])

    # Pre-compute correlation subsets once (cheap, data-driven)
    corr_subsets = _correlation_subsets(examples, input_arity)

    for step in range(max_steps):
        carry, outputs = model(carry, x_input)

        # Average logits across batch, mask to valid vocab
        logits = {
            k: outputs[k].detach().mean(dim=0)[:n_functions]
            for k in ("primary_logits", "secondary_logits", "tertiary_logits")
        }
        comp_logits = outputs["composition_logits"].detach().mean(dim=0)

        # Routing: merge model predictions with correlation heuristic
        routing_logits = outputs["routing_logits"].detach().mean(dim=0)
        model_subsets = _routing_subsets_from_scores(
            routing_logits[:input_arity], input_arity)
        routing_subsets = model_subsets[:]
        for s in corr_subsets:
            if s not in routing_subsets:
                routing_subsets.append(s)
        routing_subsets = routing_subsets[:8]

        # Top-k predictions (grows with step for diversity)
        top_k = min(3 + step, n_functions)
        tops = {k: torch.topk(v, min(top_k, len(v))).indices for k, v in logits.items()}
        comp_top = torch.topk(comp_logits, min(len(COMP_TYPES), len(comp_logits))).indices

        # Generate and deduplicate candidates
        candidates = _generate_candidates(state, tops, comp_top, max_depth,
                                          input_arity,
                                          routing_subsets=routing_subsets)

        new_count = 0
        for cand in candidates:
            key = _cand_key(cand)
            if key in tried:
                continue
            tried.add(key)
            new_count += 1

            if exe.validate(state, cand, examples):
                return cand
            near_misses.append(cand)

        # Incremental constant fitting on this step's candidates (capped)
        step_misses = near_misses[-new_count:] if new_count > 0 else []
        fitted = _try_fit_any(state, step_misses[:50], examples)
        if fitted is not None:
            return fitted

        # Small noise on carry for exploration
        with torch.no_grad():
            carry.y = carry.y + torch.randn_like(carry.y) * 0.01
            carry.z = carry.z + torch.randn_like(carry.z) * 0.01

    # Targeted pass: try all learned functions with correlation-guided routing.
    # This catches compositions the TRM hasn't learned to predict yet.
    fitted = _targeted_learned_search(state, examples, input_arity, corr_subsets)
    if fitted is not None:
        return fitted

    # Final pass: constant fitting on near misses (capped)
    return _try_fit_any(state, near_misses[:200], examples)


# ---------------------------------------------------------------------------
# Targeted search over learned functions
# ---------------------------------------------------------------------------

def _targeted_learned_search(state: dict, examples: list[tuple[list, Any]],
                              input_arity: int,
                              corr_subsets: list[list[int]]) -> dict | None:
    """Try compositions using learned + key primitive functions with routing.

    This is a small focused search — NOT exhaustive over all functions.
    Only tries learned functions and a few key primitives (MUL, ADD, CONST,
    DIV) in parallel/sequential/nested compositions with correlation-guided
    routing. Tries constant fitting on each candidate immediately.
    """
    loop_id = state["loop_id"]

    # Collect learned functions and key primitives
    learned = []
    key_primitives = []
    for fid, meta in state["metadata"].items():
        if fid == loop_id or meta["arity"] < 1:
            continue
        if meta["layer"] > 0:
            learned.append(fid)
        elif meta["name"] in ("MUL", "ADD", "SUB", "CONST", "DIV"):
            key_primitives.append(fid)

    pool = learned + key_primitives
    if not pool:
        return None

    near_misses = []
    tried = set()

    def _try(cand):
        key = _cand_key(cand)
        if key in tried:
            return None
        tried.add(key)
        if exe.validate(state, cand, examples):
            return cand
        near_misses.append(cand)
        return None

    # 1. Single learned functions with routing + constants
    for fid in pool:
        meta = state["metadata"].get(fid)
        if meta is None:
            continue
        arity = meta["arity"]
        # Try with all columns
        result = _try(_candidate(fid))
        if result:
            return result
        # Try with routed subsets (exact size match)
        for subset in corr_subsets:
            if len(subset) == arity and len(subset) != input_arity:
                result = _try(_candidate(fid, comp_type="none", routing=[subset]))
                if result:
                    return result
        # Try permutation-based routing from top correlated columns
        if arity < input_arity and arity <= 4:
            top_cols = set()
            for s in corr_subsets[:3]:
                top_cols.update(s)
            top_cols = sorted(top_cols)[:min(len(top_cols), 5)]
            perm_count = 0
            for perm in product(top_cols, repeat=arity):
                route = list(perm)
                result = _try(_candidate(fid, comp_type="none", routing=[route]))
                if result:
                    return result
                perm_count += 1
                if perm_count >= 50:
                    break

    # Constant fitting on single-function candidates
    fitted = _try_fit_any(state, near_misses, examples)
    if fitted:
        return fitted
    near_misses.clear()

    # 2. Sequential compositions: primary(secondary(routed_inputs))
    for pid in pool:
        for sid in pool:
            meta_s = state["metadata"].get(sid)
            if meta_s is None:
                continue
            # Full inputs
            result = _try(_candidate(pid, sid, comp_type="sequential"))
            if result:
                return result
            # Routed
            for subset in corr_subsets:
                if len(subset) == meta_s["arity"] and len(subset) != input_arity:
                    result = _try(_candidate(pid, sid, comp_type="sequential",
                                             routing=[subset]))
                    if result:
                        return result

    fitted = _try_fit_any(state, near_misses[:100], examples)
    if fitted:
        return fitted
    near_misses.clear()

    # 3. Parallel: tid(pid(routed), sid(routed))
    combiners = [fid for fid in pool
                 if state["metadata"].get(fid, {}).get("arity") == 2]
    par_misses = []
    par_total = 0
    for pid in pool:
        meta_p = state["metadata"].get(pid)
        if meta_p is None:
            continue
        arity_p = meta_p["arity"]
        if arity_p < 1 or arity_p > input_arity:
            continue
        for sid in pool:
            meta_s = state["metadata"].get(sid)
            if meta_s is None:
                continue
            arity_s = meta_s["arity"]
            if arity_s < 1 or arity_s > input_arity:
                continue
            for tid in combiners:
                batch = []
                _add_parallel_routed_from_subsets(
                    batch, pid, sid, tid, "parallel",
                    arity_p, arity_s, input_arity, corr_subsets,
                    max_routings=100)
                for cand in batch:
                    result = _try(cand)
                    if result:
                        return result
                    par_misses.append(cand)
                par_total += len(batch)

    print(f"    [targeted] parallel: {par_total} candidates, {len(par_misses)} misses, fitting...")
    fitted = _try_fit_any(state, par_misses[:200], examples)
    if fitted:
        return fitted

    return None


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(state: dict, tops: dict, comp_top: torch.Tensor,
                         max_depth: int, input_arity: int = 2,
                         routing_subsets: list[list[int]] | None = None
                         ) -> list[dict]:
    """Build candidate dicts from top-k predictions at each depth.

    routing_subsets: model-predicted column subsets to try (from head_routing).
    Each subset is a list of column indices, e.g. [0, 2] means "use cols 0 and 2".
    The "all columns" subset is always included.
    """
    candidates = []
    loop_id = state["loop_id"]

    primary_ids = tops["primary_logits"].tolist()
    secondary_ids = tops["secondary_logits"].tolist()
    tertiary_ids = tops["tertiary_logits"].tolist()
    comp_types_predicted = [COMP_TYPES[i] for i in comp_top.tolist() if i < len(COMP_TYPES)]

    # Depth 1: single functions (with routing)
    for pid in primary_ids:
        candidates.append(_candidate(pid))
        _add_routed_variants(candidates, state, pid, None, None,
                             "none", input_arity, routing_subsets)

    if max_depth < 2:
        return candidates

    # Depth 2: binary compositions
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
                            input_arity, comp,
                            routing_subsets=routing_subsets)
                else:
                    candidates.append(_candidate(pid, sid, comp_type=comp))
                    _add_routed_variants(candidates, state, pid, sid, None,
                                         comp, input_arity, routing_subsets)

    # Depth 2: LOOP candidates (unary + binary body)
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

    # Depth 3: use learned functions if the TRM suggests them
    learned = [fid for fid in primary_ids + secondary_ids
               if state["metadata"].get(fid, {}).get("layer", 0) > 0]

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
                            input_arity, comp,
                            routing_subsets=routing_subsets)
            else:
                for pid in primary_ids:
                    if pid == loop_id:
                        continue
                    candidates.append(_candidate(pid, lid, comp_type=comp))
                    _add_routed_variants(candidates, state, pid, lid, None,
                                         comp, input_arity, routing_subsets)

    return candidates


def _add_routed_variants(candidates: list, state: dict,
                         pid: int, sid: int | None, tid: int | None,
                         comp: str, input_arity: int,
                         routing_subsets: list[list[int]] | None):
    """Add routed variants of a candidate for non-parallel comp types.

    For each model-suggested column subset, check if the function arities
    match and create a routed candidate. Skips the "all columns" subset
    since that's already the default candidate.
    """
    if routing_subsets is None or input_arity <= 1:
        return
    if comp in ("parallel", "loop_direct", "loop_binary"):
        return

    for subset in routing_subsets:
        if len(subset) == input_arity:
            continue

        subset_arity = len(subset)

        if comp == "none":
            meta = state["metadata"].get(pid)
            if meta and meta["arity"] == subset_arity:
                candidates.append(_candidate(pid, comp_type="none",
                                             routing=[subset]))

        elif comp == "sequential":
            meta_s = state["metadata"].get(sid)
            if meta_s and meta_s["arity"] == subset_arity:
                candidates.append(_candidate(pid, sid, comp_type="sequential",
                                             routing=[subset]))

        elif comp == "nested":
            meta_p = state["metadata"].get(pid)
            if meta_p and meta_p["arity"] == subset_arity:
                candidates.append(_candidate(pid, sid, comp_type="nested",
                                             routing=[subset]))


def _add_parallel_candidates(candidates: list, state: dict,
                             pid: int, sid: int, tid: int,
                             input_arity: int, comp: str,
                             routing_subsets: list[list[int]] | None = None):
    """Add parallel candidates with routing guided by routing_subsets."""
    meta_p = state["metadata"].get(pid)
    meta_s = state["metadata"].get(sid)
    if meta_p is None or meta_s is None:
        return

    arity_p = meta_p["arity"]
    arity_s = meta_s["arity"]

    if arity_p == input_arity and arity_s == input_arity:
        candidates.append(_candidate(pid, sid, tid, comp_type=comp))
        return

    if 0 < arity_p <= input_arity and 0 < arity_s <= input_arity:
        if routing_subsets is not None:
            _add_parallel_routed_from_subsets(
                candidates, pid, sid, tid, comp,
                arity_p, arity_s, input_arity, routing_subsets)
        else:
            # No guidance — try a limited set of routings
            seen = set()
            for route_p, route_s in _generate_routings(input_arity, arity_p, arity_s):
                key = (tuple(route_p), tuple(route_s))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_candidate(pid, sid, tid, comp_type=comp,
                                                 routing=[route_p, route_s]))
                if len(seen) >= 20:
                    break


def _add_parallel_routed_from_subsets(candidates: list,
                                       pid: int, sid: int, tid: int,
                                       comp: str,
                                       arity_p: int, arity_s: int,
                                       input_arity: int,
                                       routing_subsets: list[list[int]],
                                       max_routings: int = 25):
    """Generate parallel routing candidates guided by model-predicted subsets.

    Caps total routings per function triple to keep search fast.
    """
    seen = set()
    # Sort subsets smallest-first so focused routing gets tried before
    # the combinatorial explosion of larger pools burns the cap.
    sorted_subsets = sorted(routing_subsets, key=len)
    for subset in sorted_subsets:
        if len(subset) < max(arity_p, arity_s):
            continue  # pool too small for either branch, skip
        # Use product (with replacement + ordering) to allow same-column
        # routing (e.g. MUL(x[2], x[2]) for squaring) and correct
        # positional ordering for composed functions.
        for combo_p in product(subset, repeat=arity_p):
            for combo_s in product(subset, repeat=arity_s):
                key = (combo_p, combo_s)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_candidate(pid, sid, tid, comp_type=comp,
                                             routing=[list(combo_p), list(combo_s)]))
                if len(seen) >= max_routings:
                    return


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def _try_fit_constants(state: dict, cand: dict,
                       examples: list[tuple[list, Any]],
                       r2_threshold: float = 0.999) -> dict | None:
    """Try fitting a single multiplicative constant into a candidate."""
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
    """Try constant-fitting on a list of candidates. Return best match or None."""
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
# Column pruning
# ---------------------------------------------------------------------------

def _prune_columns(examples: list[tuple[list, Any]], input_arity: int,
                   tolerance: float = 0.1) -> list[int]:
    """Return indices of columns with abs(correlation) >= tolerance.

    Columns below the tolerance are junk — they contribute nothing to
    the output. Pruning them shrinks the search space dramatically.
    Always keeps at least 1 column (the highest-correlated).
    """
    if input_arity <= 1 or len(examples) < 3:
        return list(range(input_arity))

    n = len(examples)
    outputs = [float(examples[i][1]) for i in range(n)]
    out_mean = sum(outputs) / n
    out_std = max(1e-12, (sum((y - out_mean)**2 for y in outputs) / n) ** 0.5)

    correlations = []
    for c in range(input_arity):
        vals = [float(examples[i][0][c]) for i in range(n)]
        c_mean = sum(vals) / n
        c_std = max(1e-12, (sum((x - c_mean)**2 for x in vals) / n) ** 0.5)
        cov = sum((vals[i] - c_mean) * (outputs[i] - out_mean)
                  for i in range(n)) / n
        correlations.append(abs(cov / (c_std * out_std)))

    kept = [c for c in range(input_arity) if correlations[c] >= tolerance]

    # Always keep at least the top column
    if not kept:
        best = max(range(input_arity), key=lambda c: correlations[c])
        kept = [best]

    return kept


def _remap_candidate(candidate: dict, kept_cols: list[int]) -> dict:
    """Remap a candidate's routing from pruned indices back to original indices.

    If the candidate has routing like [[0, 1], [0, 2]] and kept_cols is
    [0, 2, 4], the remapped routing is [[0, 2], [0, 4]].

    If no routing, add one that maps kept_cols to the function inputs.
    """
    remapped = dict(candidate)
    routing = candidate.get("routing")
    if routing:
        remapped["routing"] = [
            [kept_cols[i] for i in route]
            for route in routing
        ]
    else:
        # No routing = function uses all pruned columns in order.
        # Remap to original indices so executor picks the right columns.
        comp = candidate["comp_type"]
        if comp == "none":
            remapped["routing"] = [kept_cols]
        elif comp == "sequential":
            remapped["routing"] = [kept_cols]
        elif comp == "parallel":
            remapped["routing"] = [kept_cols, kept_cols]
    return remapped


# ---------------------------------------------------------------------------
# Input routing
# ---------------------------------------------------------------------------

def _routing_subsets_from_scores(scores: torch.Tensor, input_arity: int,
                                 threshold: float = 0.5,
                                 max_subsets: int = 6) -> list[list[int]]:
    """Convert per-column relevance scores into a small set of column subsets."""
    probs = torch.sigmoid(scores)
    all_cols = list(range(input_arity))

    subsets = [all_cols]

    if input_arity <= 1:
        return subsets

    relevant = [i for i in all_cols if probs[i].item() > threshold]
    if 0 < len(relevant) < input_arity:
        subsets.append(relevant)

    ranked = torch.argsort(probs).tolist()
    for drop_idx in ranked:
        subset = [i for i in all_cols if i != drop_idx]
        if subset not in subsets:
            subsets.append(subset)
        if len(subsets) >= max_subsets:
            break

    if input_arity >= 3:
        top2 = torch.topk(probs, min(2, input_arity)).indices.tolist()
        top2.sort()
        if top2 not in subsets:
            subsets.append(top2)

    return subsets[:max_subsets]


def _correlation_subsets(examples: list[tuple[list, Any]], input_arity: int,
                         max_subsets: int = 6) -> list[list[int]]:
    """Generate column subsets by ranking columns on correlation with output.

    Fast statistical heuristic — computes abs(correlation) between each
    input column and the output, then generates top-k subsets.
    """
    if input_arity <= 1 or len(examples) < 3:
        return [list(range(input_arity))]

    n = len(examples)
    cols = []
    for c in range(input_arity):
        vals = [float(examples[i][0][c]) for i in range(n)]
        cols.append(vals)
    outputs = [float(examples[i][1]) for i in range(n)]

    out_mean = sum(outputs) / n
    out_std = max(1e-12, (sum((y - out_mean)**2 for y in outputs) / n) ** 0.5)

    correlations = []
    for c in range(input_arity):
        c_mean = sum(cols[c]) / n
        c_std = max(1e-12, (sum((x - c_mean)**2 for x in cols[c]) / n) ** 0.5)
        cov = sum((cols[c][i] - c_mean) * (outputs[i] - out_mean)
                  for i in range(n)) / n
        correlations.append(abs(cov / (c_std * out_std)))

    ranked = sorted(range(input_arity), key=lambda i: -correlations[i])

    all_cols = list(range(input_arity))
    subsets = [all_cols]

    # Top-k subsets for k = 1, 2, ..., input_arity-1
    for k in range(1, input_arity):
        subset = sorted(ranked[:k])
        if subset not in subsets:
            subsets.append(subset)
        if len(subsets) >= max_subsets:
            break

    # Also generate all pairs from top-3 correlated columns
    if input_arity >= 3:
        top3 = ranked[:min(3, input_arity)]
        for pair in combinations(top3, 2):
            subset = sorted(pair)
            if subset not in subsets:
                subsets.append(subset)
            if len(subsets) >= max_subsets:
                break

    return subsets[:max_subsets]


def _generate_routings(input_arity: int, arity_p: int, arity_s: int
                       ) -> list[tuple[list[int], list[int]]]:
    """Generate ways to route inputs to two sub-functions.

    Uses product to allow same-column routing and correct positional
    ordering (e.g. MUL(x[i], x[i]) for squaring). Capped to avoid
    combinatorial explosion.
    """
    all_indices = list(range(input_arity))
    routings = []
    for combo_p in product(all_indices, repeat=arity_p):
        for combo_s in product(all_indices, repeat=arity_s):
            routings.append((list(combo_p), list(combo_s)))
            if len(routings) >= 200:
                return routings
    return routings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(primary_id: int, secondary_id: int | None = None,
               tertiary_id: int | None = None, *,
               comp_type: str = "none",
               routing: list[list[int]] | None = None) -> dict:
    return {
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "tertiary_id": tertiary_id,
        "comp_type": comp_type,
        "routing": routing,
    }


def _cand_key(c: dict) -> tuple:
    routing = c.get("routing")
    routing_key = tuple(tuple(r) for r in routing) if routing else None
    return (c["primary_id"], c["secondary_id"], c.get("tertiary_id"),
            c["comp_type"], routing_key)


def format_examples(examples: list[tuple[list, Any]], *,
                    input_dim: int, seq_len: int) -> torch.Tensor:
    """Encode examples as float vectors for the TRM.

    Each input value is placed at position 0 of its input_dim-sized slot,
    with a magnitude encoding spread across a few additional dimensions
    for richer signal.

    Returns: (batch_size, seq_len, input_dim) float tensor.
    """
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
