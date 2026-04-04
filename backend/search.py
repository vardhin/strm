"""
Program search for NSSR.

Two strategies:
  - exhaustive: brute-force over all function combinations (for shallow depths)
  - guided:     TRM model predicts likely compositions, tested iteratively

Both return a candidate dict or None.
"""

import math
import torch
from typing import Any

import registry as reg
import executor as exe
from model import Carry, fresh_carry


COMP_TYPES = ["none", "sequential", "nested", "parallel"]


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------

def exhaustive(state: dict, examples: list[tuple[list[int], Any]],
               max_depth: int = 3) -> dict | None:
    """Brute-force search over all function combinations up to max_depth."""
    loop_id = state["loop_id"]
    all_ids = list(state["functions"].keys())
    input_arity = len(examples[0][0])
    near_misses = []  # candidates to try constant-fitting on

    # Depth 1: single functions
    for fid in all_ids:
        if fid == loop_id:
            continue
        cand = _candidate(fid)
        if exe.validate(state, cand, examples):
            return cand
        near_misses.append(cand)

    if max_depth < 2:
        return _try_fit_any(state, near_misses, examples)

    # Depth 2: two-function compositions
    for pid in all_ids:
        if pid == loop_id:
            continue
        for sid in all_ids:
            if sid == loop_id:
                continue
            for comp in ("sequential", "nested"):
                cand = _candidate(pid, sid, comp_type=comp)
                if exe.validate(state, cand, examples):
                    return cand
                near_misses.append(cand)

    # Depth 2: LOOP with unary body — LOOP(body_fn, count=b, init=a)
    if loop_id is not None:
        for body_id in all_ids:
            meta = state["metadata"].get(body_id)
            if meta is None or body_id == loop_id:
                continue
            if meta["arity"] == 1:
                cand = _candidate(loop_id, body_id, comp_type="loop_direct")
                if exe.validate(state, cand, examples):
                    return cand

    # Depth 2: LOOP with binary body — LOOP(body_fn, count=b, init=0, step=a)
    if loop_id is not None:
        for body_id in all_ids:
            meta = state["metadata"].get(body_id)
            if meta is None or body_id == loop_id:
                continue
            if meta["arity"] == 2:
                cand = _candidate(loop_id, body_id, comp_type="loop_binary")
                if exe.validate(state, cand, examples):
                    return cand

    if max_depth < 3:
        return _try_fit_any(state, near_misses, examples)

    # Depth 3: parallel with tertiary combiner
    if input_arity >= 2:
        for pid in all_ids:
            if pid == loop_id:
                continue
            for sid in all_ids:
                if sid == loop_id:
                    continue
                for tid in all_ids:
                    if tid == loop_id:
                        continue
                    cand = _candidate(pid, sid, tid, comp_type="parallel")
                    if exe.validate(state, cand, examples):
                        return cand
                    near_misses.append(cand)

    return _try_fit_any(state, near_misses, examples)


# ---------------------------------------------------------------------------
# TRM-guided search
# ---------------------------------------------------------------------------

def guided(state: dict, model, examples: list[tuple[list[int], Any]],
           x_input: torch.Tensor, *, max_steps: int = 10,
           max_depth: int = 3) -> dict | None:
    """Use the TRM to predict likely compositions, then validate them."""
    model.eval()

    batch_size, seq_len, _ = x_input.shape
    carry = fresh_carry(batch_size, seq_len, model.d_model)

    tried: set[tuple] = set()
    n_functions = reg.vocab_size(state)
    near_misses = []

    for step in range(max_steps):
        carry, outputs = model(carry, x_input)

        # Average logits across batch, mask to valid vocab
        logits = {
            k: outputs[k].detach().mean(dim=0)[:n_functions]
            for k in ("primary_logits", "secondary_logits", "tertiary_logits")
        }
        comp_logits = outputs["composition_logits"].detach().mean(dim=0)

        # Top-k predictions (grows with step for diversity)
        top_k = min(3 + step, n_functions)
        tops = {k: torch.topk(v, min(top_k, len(v))).indices for k, v in logits.items()}
        comp_top = torch.topk(comp_logits, min(len(COMP_TYPES), len(comp_logits))).indices

        # Generate and deduplicate candidates
        candidates = _generate_candidates(state, tops, comp_top, max_depth)

        for cand in candidates:
            key = _cand_key(cand)
            if key in tried:
                continue
            tried.add(key)

            if exe.validate(state, cand, examples):
                return cand
            near_misses.append(cand)

        # Small noise on carry for exploration
        with torch.no_grad():
            carry.y = carry.y + torch.randn_like(carry.y) * 0.01
            carry.z = carry.z + torch.randn_like(carry.z) * 0.01

    # No exact match found — try fitting constants on all near misses
    return _try_fit_any(state, near_misses, examples)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(state: dict, tops: dict, comp_top: torch.Tensor,
                         max_depth: int) -> list[dict]:
    """Build candidate dicts from top-k predictions at each depth."""
    candidates = []
    loop_id = state["loop_id"]

    primary_ids = tops["primary_logits"].tolist()
    secondary_ids = tops["secondary_logits"].tolist()
    tertiary_ids = tops["tertiary_logits"].tolist()
    comp_types_predicted = [COMP_TYPES[i] for i in comp_top.tolist() if i < len(COMP_TYPES)]

    # Depth 1: single functions
    for pid in primary_ids:
        candidates.append(_candidate(pid))

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
                        candidates.append(_candidate(pid, sid, tid, comp_type=comp))
                else:
                    candidates.append(_candidate(pid, sid, comp_type=comp))

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
                        candidates.append(_candidate(lid, pid, tid, comp_type=comp))
            else:
                for pid in primary_ids:
                    if pid == loop_id:
                        continue
                    candidates.append(_candidate(pid, lid, comp_type=comp))

    return candidates


# ---------------------------------------------------------------------------
# Constant fitting
# ---------------------------------------------------------------------------

def _try_fit_constants(state: dict, cand: dict,
                       examples: list[tuple[list, Any]],
                       r2_threshold: float = 0.999) -> dict | None:
    """Try fitting a single multiplicative constant into a candidate.

    For each composition type, we compute what the candidate produces WITHOUT
    a constant, then solve for k such that k * produced = expected.
    Returns a new candidate dict with 'constants' field if successful.
    """
    comp = cand["comp_type"]
    pid = cand["primary_id"]

    # Collect (produced, expected) pairs by running the candidate as-is
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

    # Try fitting k: expected = k * produced
    k = _fit_scale(produced, expected)
    if k is not None:
        r2 = _r2(expected, [k * p for p in produced])
        if r2 > r2_threshold:
            return {**cand, "constants": [k]}

    # Try fitting additive: expected = produced + k
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
    """Solve for k in expected = k * produced (least squares)."""
    num = sum(p * e for p, e in zip(produced, expected))
    den = sum(p * p for p in produced)
    if abs(den) < 1e-15:
        return None
    return num / den


def _fit_offset(produced: list[float], expected: list[float]) -> float | None:
    """Solve for k in expected = produced + k."""
    diffs = [e - p for e, p in zip(produced, expected)]
    mean_diff = sum(diffs) / len(diffs)
    # Check consistency: all diffs should be ~equal
    if all(abs(d - mean_diff) < 1e-6 for d in diffs):
        return mean_diff
    return None


def _r2(actual: list[float], predicted: list[float]) -> float:
    """Compute R² score."""
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
               comp_type: str = "none") -> dict:
    return {
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "tertiary_id": tertiary_id,
        "comp_type": comp_type,
    }


def _cand_key(c: dict) -> tuple:
    return (c["primary_id"], c["secondary_id"], c.get("tertiary_id"), c["comp_type"])


def format_examples(examples: list[tuple[list, Any]], *,
                    input_dim: int, seq_len: int) -> torch.Tensor:
    """Encode examples as float vectors for the TRM.

    Each input value is placed at position 0 of its input_dim-sized slot,
    with a magnitude encoding spread across a few additional dimensions
    for richer signal. This replaces the old bit-vector encoding and
    supports arbitrary float inputs.

    Returns: (batch_size, seq_len, input_dim) float tensor.
    """
    batch_size = len(examples)
    data = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32)

    for b, (inputs, _) in enumerate(examples):
        for pos in range(min(len(inputs), seq_len)):
            val = float(inputs[pos])
            # Channel 0: raw value (normalized loosely — values typically < 1000)
            data[b, pos, 0] = val / 100.0
            # Channel 1: sign
            data[b, pos, 1] = 1.0 if val >= 0 else -1.0
            # Channel 2: log-magnitude (for scale awareness)
            data[b, pos, 2] = torch.log1p(torch.tensor(abs(val))).item()
            # Channel 3: fractional part (distinguishes 2.5 from 2.0)
            data[b, pos, 3] = val - int(val) if val >= 0 else -(abs(val) - int(abs(val)))
            # Channels 4+: leave as zero padding

    return data
