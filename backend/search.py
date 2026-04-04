"""
Program search for NSSR.

Two strategies:
  - exhaustive: brute-force over all function combinations (for shallow depths)
  - guided:     TRM model predicts likely compositions, tested iteratively

Both return a candidate dict or None.
"""

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

    # Depth 1: single functions
    for fid in all_ids:
        if fid == loop_id:
            continue
        cand = _candidate(fid)
        if exe.validate(state, cand, examples):
            return cand

    if max_depth < 2:
        return None

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
        return None

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

    return None


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

        # Small noise on carry for exploration
        with torch.no_grad():
            carry.y = carry.y + torch.randn_like(carry.y) * 0.01
            carry.z = carry.z + torch.randn_like(carry.z) * 0.01

    return None


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


def format_examples(examples: list[tuple[list[int], Any]], *,
                    input_dim: int, seq_len: int) -> torch.Tensor:
    """Encode examples as bit vectors for the TRM.

    Returns: (batch_size, seq_len, input_dim) float tensor.
    """
    batch_size = len(examples)
    data = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32)

    for b, (inputs, _) in enumerate(examples):
        for pos in range(min(len(inputs), seq_len)):
            val = inputs[pos]
            if val < 0:
                val = (1 << input_dim) + val
            for bit in range(input_dim):
                data[b, pos, bit] = float((val >> bit) & 1)

    return data
