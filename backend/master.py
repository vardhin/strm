#MODULES
import math
from typing import Any
from itertools import product
import torch
import torch.nn.functional as F
import os
import torch
import sqlite3
import random

import executor as exe
import registry as reg
import db

"""
TRM (Tiny Recursive reasoning Model) for NSSR.

Based on "Less is More: Recursive Reasoning with Tiny Networks"
(Jolicoeur-Martineau, 2025).

Architecture:
    - Single tiny transformer (2 layers) recursed deeply
    - Two carry states: y (answer embedding) and z (latent reasoning)
    - Latent recursion: n steps of z = net(x+y+z), then y = net(y+z)
    - Deep recursion: T-1 no-grad passes + 1 grad pass
    - Deep supervision: N_sup improvement steps with detach between each

    Effective depth per supervision step = T * (n+1) * n_layers
    With n=6, T=3, 2 layers → 3 * 7 * 2 = 42 effective layers

Outputs per forward pass:
    - primary_logits:     which function to use first
    - secondary_logits:   which function to compose with
    - tertiary_logits:    combiner function (for parallel composition)
    - composition_logits: how to compose (none / sequential / nested / parallel)
    - halt_logits:        whether to stop reasoning

NOTE: nn.Module is used here because PyTorch requires it for parameter
management (state_dict, optimizer, device transfer, etc).
"""

import torch
import torch.nn as nn

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Carry state
# ---------------------------------------------------------------------------

@dataclass
class Carry:
    y: torch.Tensor        # (batch, seq_len, d_model) — answer embedding
    z: torch.Tensor        # (batch, seq_len, d_model) — latent reasoning
    steps: torch.Tensor    # (batch,) — step counter
    halted: torch.Tensor   # (batch,) — boolean halt mask


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TRM(nn.Module):
    def __init__(self, *, input_dim: int, seq_len: int, d_model: int,
                 n_heads: int = 8, n_layers: int = 2, d_ff: int | None = None,
                 dropout: float = 0.1, n_functions: int,
                 n_recursions: int = 6, T: int = 3):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.n_recursions = n_recursions
        self.T = T

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)

        # Single transformer — reused for all recursion steps
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # Normalization for recursion inputs to prevent NaN explosion
        self.ln_recursion = nn.LayerNorm(d_model)

        # Output heads (read from y, the answer embedding)
        self.head_primary = nn.Linear(d_model, n_functions)
        self.head_secondary = nn.Linear(d_model, n_functions)
        self.head_tertiary = nn.Linear(d_model, n_functions)
        self.head_composition = nn.Linear(d_model, 4)   # none, seq, nested, parallel
        self.head_halt = nn.Linear(d_model, 1)

    def _apply_blocks(self, h: torch.Tensor) -> torch.Tensor:
        """Run input through all transformer blocks."""
        for block in self.blocks:
            h = block(h)
        return h

    def _latent_recursion(self, x: torch.Tensor, y: torch.Tensor,
                          z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One full latent recursion: n steps refining z, then 1 step refining y.

        The SAME network is used for both — the task is distinguished by
        whether x is in the input (reasoning) or not (answer refinement).
        """
        # n steps: refine z given (x, y, z)
        for _ in range(self.n_recursions):
            z = self._apply_blocks(self.ln_recursion(x + y + z))

        # 1 step: refine y given (y, z) — NO x, so the network knows
        # this is answer-refinement, not reasoning
        y = self._apply_blocks(self.ln_recursion(y + z))

        return y, z

    def _deep_recursion(self, x: torch.Tensor, y: torch.Tensor,
                        z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """T-1 no-grad passes + 1 grad pass through latent_recursion.

        This gives effective depth = T * (n+1) * n_layers without needing
        to backprop through all of it.
        """
        # T-1 passes without gradients (improve y,z cheaply)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self._latent_recursion(x, y, z)

        # 1 pass with gradients (the one we backprop through)
        y, z = self._latent_recursion(x, y, z)

        return y, z

    def forward(self, carry: Carry, x: torch.Tensor) -> tuple[Carry, dict[str, torch.Tensor]]:
        """
        One deep supervision step.

        Args:
            carry: previous (y, z) reasoning state
            x: (batch, seq_len, input_dim) — bit-encoded examples

        Returns:
            (new_carry, outputs_dict)
        """
        # Embed input (fixed across all recursion steps)
        x_emb = self.input_proj(x) + self.pos_emb[:x.shape[1]].unsqueeze(0)

        # Deep recursion: T*(n+1) applications of the transformer
        new_y, new_z = self._deep_recursion(x_emb, carry.y, carry.z)

        new_carry = Carry(
            y=new_y.detach(),
            z=new_z.detach(),
            steps=carry.steps + 1,
            halted=carry.halted,
        )

        # Output heads read from y (the answer), pooled across sequence
        pooled = new_y.mean(dim=1)

        outputs = {
            "primary_logits":     self.head_primary(pooled),
            "secondary_logits":   self.head_secondary(pooled),
            "tertiary_logits":    self.head_tertiary(pooled),
            "composition_logits": self.head_composition(pooled),
            "halt_logits":        self.head_halt(pooled).squeeze(-1),
        }

        return new_carry, outputs


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_model(*, input_dim: int, seq_len: int, d_model: int,
                 n_functions: int, **kwargs) -> TRM:
    """Create a TRM with sensible defaults."""
    return TRM(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        n_functions=n_functions,
        **kwargs,
    )


def fresh_carry(batch_size: int, seq_len: int, d_model: int) -> Carry:
    """Create a zeroed carry state."""
    return Carry(
        y=torch.zeros(batch_size, seq_len, d_model),
        z=torch.zeros(batch_size, seq_len, d_model),
        steps=torch.zeros(batch_size, dtype=torch.int32),
        halted=torch.zeros(batch_size, dtype=torch.bool),
    )


def reset_carry(carry: Carry) -> Carry:
    """Zero out the carry."""
    return Carry(
        y=torch.zeros_like(carry.y),
        z=torch.zeros_like(carry.z),
        steps=torch.zeros_like(carry.steps),
        halted=torch.zeros_like(carry.halted),
    )


def resize_heads(model: TRM, old_n: int, new_n: int) -> None:
    """Expand the function output heads to accommodate new vocabulary.

    Copies existing weights and initializes new rows with small noise
    around the existing mean.
    """
    if new_n <= old_n:
        return

    for attr in ("head_primary", "head_secondary", "head_tertiary"):
        old_head = getattr(model, attr)
        new_head = nn.Linear(model.d_model, new_n, bias=old_head.bias is not None)

        with torch.no_grad():
            new_head.weight[:old_n] = old_head.weight
            avg = old_head.weight.mean(dim=0)
            new_head.weight[old_n:] = avg.unsqueeze(0) + torch.randn(new_n - old_n, model.d_model) * 0.01

            if old_head.bias is not None:
                new_head.bias[:old_n] = old_head.bias
                new_head.bias[old_n:] = 0.0

        setattr(model, attr, new_head)


"""
Composition simplifier for NSSR.

After search finds a working composition, try to make it shorter
before registering it as a new function.

Strategies (tried in order):
  1. Replace with a single existing function
  2. Prune redundant intermediate steps
"""

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


"""
Program search for NSSR.

Strategy: TRM-guided search with NULL-based column elimination.
The TRM predicts likely function compositions. Junk columns are handled
by generating candidates that NULL out subsets of columns — the executor
skips NULL'd columns and routes the remaining ones into functions
(with same-column repetition allowed).

Returns a candidate dict or None.
"""


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
        fitted = _try_fit_any(state, step_misses, train_examples)
        if fitted is not None and exe.validate(state, fitted, holdout_examples):
            return fitted, 1.0

        # Small noise on carry for exploration
        with torch.no_grad():
            carry.y = carry.y + torch.randn_like(carry.y) * 0.01
            carry.z = carry.z + torch.randn_like(carry.z) * 0.01

    # Final pass: constant fitting on near misses
    fitted = _try_fit_any(state, near_misses, train_examples)
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

"""
Training for the NSSR TRM model.

Uses deep supervision: each training step runs N_sup supervision steps,
where the carry (y, z) is detached between steps. This lets the model
iteratively refine its answer across multiple passes, emulating very
deep networks without backpropping through all of them.

A target is a candidate dict (same shape as search returns):
    {
        "primary_id":   int,
        "secondary_id": int | None,
        "tertiary_id":  int | None,
        "comp_type":    "none" | "sequential" | "nested" | "parallel" | "loop_direct",
    }
"""


COMP_TYPE_INDEX = {"none": 0, "sequential": 1, "nested": 2, "parallel": 3, "loop_direct": 0, "loop_binary": 0}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(outputs: dict[str, torch.Tensor], target: dict,
                 batch_size: int) -> torch.Tensor:
    """Multi-head cross-entropy loss for a single target composition."""

    primary_target = torch.full((batch_size,), target["primary_id"], dtype=torch.long)
    secondary_val = target.get("secondary_id") or 0
    secondary_target = torch.full((batch_size,), secondary_val, dtype=torch.long)
    tertiary_val = target.get("tertiary_id") or 0
    tertiary_target = torch.full((batch_size,), tertiary_val, dtype=torch.long)
    comp_target = torch.full((batch_size,), COMP_TYPE_INDEX[target["comp_type"]], dtype=torch.long)

    loss_p = F.cross_entropy(outputs["primary_logits"], primary_target)
    loss_s = F.cross_entropy(outputs["secondary_logits"], secondary_target)
    loss_t = F.cross_entropy(outputs["tertiary_logits"], tertiary_target)
    loss_c = F.cross_entropy(outputs["composition_logits"], comp_target)
    loss_h = F.binary_cross_entropy_with_logits(
        outputs["halt_logits"], torch.ones(batch_size)
    )

    # Down-weight tertiary and halt when they don't matter
    t_weight = 1.0 if target.get("tertiary_id") is not None else 0.1
    s_weight = 1.0 if target.get("secondary_id") is not None else 0.1

    total = loss_p + s_weight * loss_s + t_weight * loss_t + loss_c + 0.1 * loss_h

    # Entropy regularization: prevent softmax from collapsing to a single function.
    entropy_weight = 0.05
    for key in ("primary_logits", "secondary_logits", "tertiary_logits"):
        probs = F.softmax(outputs[key], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        total = total - entropy_weight * entropy

    return total


# ---------------------------------------------------------------------------
# Training loop with deep supervision
# ---------------------------------------------------------------------------

def train_on_examples(model: TRM, optimizer: torch.optim.Optimizer,
                      examples: list, target: dict, *,
                      input_dim: int, seq_len: int,
                      num_epochs: int = 30, n_sup: int = 16) -> list[float]:
    """Train the model to predict `target` given `examples`.

    Uses deep supervision: each epoch runs n_sup supervision steps,
    accumulating loss at each step. The carry (y, z) persists across
    supervision steps (detached) so the model learns to iteratively
    refine its answer.

    Returns list of loss values per epoch.
    """
    x_input = format_examples(examples, input_dim=input_dim, seq_len=seq_len)
    batch_size = x_input.shape[0]
    losses = []

    model.train()
    for epoch in range(num_epochs):
        # Fresh carry at the start of each epoch
        carry = fresh_carry(batch_size, seq_len, model.d_model)
        epoch_loss = 0.0

        for sup_step in range(n_sup):
            # Forward: one deep supervision step
            # (internally does T*(n+1) transformer applications)
            carry, outputs = model(carry, x_input)

            loss = compute_loss(outputs, target, batch_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # ACT: check if model wants to halt (simple version —
            # halt if prediction is confident enough)
            with torch.no_grad():
                halt_prob = torch.sigmoid(outputs["halt_logits"]).mean()
                if halt_prob > 0.5 and sup_step >= 1:
                    break

        avg_loss = epoch_loss / (sup_step + 1)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}: loss = {avg_loss:.4f} ({sup_step + 1} sup steps)")

    return losses


def train_on_replay(model: TRM, optimizer: torch.optim.Optimizer,
                    replay_buffer: list[dict], *,
                    input_dim: int, seq_len: int,
                    epochs_per_task: int = 2, n_sup: int = 16) -> list[float]:
    """Full replay training: shuffle all tasks each epoch, train on everything.

    Each entry in replay_buffer is:
        {"examples": [...], "target": {...}}

    Epochs = len(replay_buffer) * epochs_per_task.
    Each epoch shuffles the buffer and does one pass through all tasks.

    Returns list of average loss per epoch.
    """
    n_tasks = len(replay_buffer)
    if n_tasks == 0:
        return []

    num_epochs = n_tasks * epochs_per_task

    # Pre-encode all tasks
    encoded = []
    for entry in replay_buffer:
        examples = entry["examples"]
        target = entry["target"]
        x_input = format_examples(examples, input_dim=input_dim, seq_len=seq_len)
        batch_size = x_input.shape[0]
        encoded.append({
            "x_input": x_input,
            "target": target,
            "batch_size": batch_size,
        })

    losses = []
    model.train()
    best_loss = float("inf")
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        # Sequential order: learning builds on prior knowledge (curriculum first)
        order = list(range(n_tasks))

        epoch_loss = 0.0
        task_count = 0

        for idx in order:
            enc = encoded[idx]
            x_input = enc["x_input"]
            target = enc["target"]
            batch_size = enc["batch_size"]

            # Fresh carry per task
            carry = fresh_carry(batch_size, seq_len, model.d_model)

            for sup_step in range(n_sup):
                carry, outputs = model(carry, x_input)
                loss = compute_loss(outputs, target, batch_size)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # ACT halt
                with torch.no_grad():
                    halt_prob = torch.sigmoid(outputs["halt_logits"]).mean()
                    if halt_prob > 0.5 and sup_step >= 1:
                        break

            epoch_loss += loss.item()
            task_count += 1

        avg_loss = epoch_loss / max(task_count, 1)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"    replay epoch {epoch}/{num_epochs}: avg_loss = {avg_loss:.4f} ({n_tasks} tasks)")

        # Early stopping: if no improvement for `patience` epochs, stop
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop at epoch {epoch} (no improvement for {patience} epochs, loss={avg_loss:.4f})")
                break

    return losses

# ==========================================
# --- helpers ---
# ==========================================

def junk():
    """Random junk value that has no relationship to the output."""
    return round(random.uniform(-100, 100), 2)

def noisy_correlated(val):
    """A value loosely correlated with `val` — multicollinearity trap."""
    return round(val + random.uniform(-3, 3), 2)


# ==========================================
# --- main.py (Orchestrator) ---
# ==========================================

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "input_dim": 32,
    "seq_len": 8,
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
    """Convert a search candidate to a storable composition list."""
    comp_type = candidate["comp_type"]
    pid = candidate["primary_id"]
    sid = candidate.get("secondary_id")
    tid = candidate.get("tertiary_id")
    routing = candidate.get("routing")

    if comp_type == "none":
        cols = routing[0] if routing else list(range(input_arity))
        return [(pid, cols)]

    if comp_type == "loop_direct":
        return [(pid, [sid, 1, 0])]

    if comp_type == "loop_binary":
        return [(pid, [sid, 1, -1, 0])]

    if comp_type == "sequential":
        cols = routing[0] if routing else list(range(input_arity))
        return [(sid, cols), (pid, [input_arity])]

    if comp_type == "nested":
        cols = routing[0] if routing else list(range(input_arity))
        steps = [(sid, [c]) for c in cols]
        combiner_args = [input_arity + i for i in range(len(cols))]
        steps.append((pid, combiner_args))
        return steps

    if comp_type == "parallel":
        if routing:
            comp = [(pid, routing[0]), (sid, routing[1])]
        else:
            comp = [(pid, list(range(input_arity))), (sid, list(range(input_arity)))]
        if tid is not None:
            comp.append((tid, [input_arity, input_arity + 1]))
        return comp

    raise ValueError(f"Unknown composition type: {comp_type}")


# ---------------------------------------------------------------------------
# Learning pipeline
# ---------------------------------------------------------------------------

def _init_replay_buffer(state: dict):
    if "replay_buffer" not in state:
        state["replay_buffer"] = list(curriculum_tasks(state))
        print(f"  replay buffer initialized with {len(state['replay_buffer'])} curriculum tasks")


def _is_duplicate_discovery(state: dict, candidate: dict) -> bool:
    for entry in state.get("replay_buffer", []):
        target = entry["target"]
        if (target.get("primary_id") == candidate.get("primary_id") and
            target.get("secondary_id") == candidate.get("secondary_id") and
            target.get("tertiary_id") == candidate.get("tertiary_id") and
            target.get("comp_type") == candidate.get("comp_type") and
            target.get("routing") == candidate.get("routing") and
            target.get("constants") == candidate.get("constants")):
            return True
    return False


def learn(conn: sqlite3.Connection, state: dict, model,
          optimizer: torch.optim.Optimizer, name: str,
          examples: list[tuple[list, float | int]], *,
          max_search_steps: int = 10, max_depth: int = 3,
          num_epochs: int = 30, max_retries: int = 2):
    
    input_dim = CONFIG["input_dim"]
    seq_len = CONFIG["seq_len"]

    fresh = "replay_buffer" not in state
    _init_replay_buffer(state)
    if fresh:
        print("  loading curriculum knowledge...")
        train_on_replay(model, optimizer, state["replay_buffer"],
                        input_dim=input_dim, seq_len=seq_len,
                        epochs_per_task=2, n_sup=CONFIG["n_sup"])

    print(f"\n{'=' * 50}")
    print(f"Learning: {name}")
    print(f"{'=' * 50}")

    x_input = format_examples(examples, input_dim=input_dim, seq_len=seq_len)
    candidate = None
    for attempt in range(1 + max_retries):
        candidate = guided(state, model, examples, x_input,
                           max_steps=max_search_steps, max_depth=max_depth,
                           temperature_boost=attempt * 1.0)
        if candidate is not None:
            break
        if attempt < max_retries:
            print(f"  search failed (attempt {attempt + 1}), retrying with higher temperature...")

    if candidate is None:
        print(f"  could not find composition for {name}")
        return False, optimizer, 0.0

    constants = candidate.get("constants")
    const_mode = candidate.get("const_mode", "multiplicative")
    if constants:
        print(f"  found: {_fmt_candidate(state, candidate)}  (k={constants[0]}, {const_mode})")
    else:
        print(f"  found: {_fmt_candidate(state, candidate)}")

    if _is_duplicate_discovery(state, candidate):
        print(f"  duplicate of existing knowledge — skipping")
        fid = None
        for f, m in state["metadata"].items():
            if m["name"] == name:
                fid = f
                break
        if fid is not None:
            r2 = exe.r_squared(state, fid, examples)
            print(f"  verification: R²={r2:.6f}")
            return r2 > 0.999, optimizer, r2
        return True, optimizer, 1.0

    input_arity = len(examples[0][0])
    composition = build_composition(candidate, input_arity, state.get("loop_id"))
    composition, effective_arity, used_cols = simplify(
        state, composition, examples, constants, const_mode)

    old_vocab = reg.vocab_size(state)
    fid = reg.register_learned(conn, state, name, effective_arity, composition,
                               constants, const_mode)
    if used_cols is not None:
        state["metadata"][fid]["used_cols"] = used_cols
    new_vocab = reg.vocab_size(state)
    print(f"  registered {name} as id={fid}")

    if new_vocab > old_vocab:
        resize_heads(model, old_vocab, new_vocab)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    state["replay_buffer"].append({"examples": examples, "target": dict(candidate)})
    
    if used_cols is not None and len(used_cols) < input_arity:
        clean_examples = [([inp[c] for c in used_cols], out) for inp, out in examples]
        clean_target = dict(candidate)
        clean_target.pop("routing", None)
        clean_target.pop("null_columns", None)
        state["replay_buffer"].append({"examples": clean_examples, "target": clean_target})
    print(f"  replay buffer: {len(state['replay_buffer'])} tasks")

    print(f"  consolidating knowledge ({len(state['replay_buffer'])} known facts)...")
    train_on_replay(model, optimizer, state["replay_buffer"],
                    input_dim=input_dim, seq_len=seq_len,
                    epochs_per_task=2, n_sup=CONFIG["n_sup"])

    if used_cols is not None:
        verify_examples = [([inp[c] for c in used_cols], exp) for inp, exp in examples]
    else:
        verify_examples = examples
    r2 = exe.r_squared(state, fid, verify_examples)
    ok = r2 > 0.999
    print(f"  verification: R²={r2:.6f} ({'passed' if ok else 'FAILED'})")
    return ok, optimizer, r2


# ---------------------------------------------------------------------------
# Curriculum & Targets (Abridged data loops)
# ---------------------------------------------------------------------------
def curriculum_tasks(state: dict) -> list[dict]:
    import random as _rng
    _rng_state = _rng.getstate()
    _rng.seed(777)

    def _junk(): return round(_rng.uniform(-100, 100), 2)
    tasks = []

    def _find(name):
        for fid, m in state["metadata"].items():
            if m["name"] == name: return fid
        return None

    def _add_task(fid, examples):
        target = {"primary_id": fid, "secondary_id": None, "tertiary_id": None, "comp_type": "none"}
        tasks.append({"target": dict(target), "examples": examples})
        arity = len(examples[0][0])
        if arity == 1:
            noisy_examples = [([_junk(), inp[0], _junk()], out) for inp, out in examples]
            noisy_target = dict(target)
            noisy_target["routing"], noisy_target["null_columns"] = [[1]], [0, 2]
            tasks.append({"target": noisy_target, "examples": noisy_examples})
        elif arity == 2:
            noisy_examples = [([inp[0], _junk(), inp[1], _junk()], out) for inp, out in examples]
            noisy_target = dict(target)
            noisy_target["routing"], noisy_target["null_columns"] = [[0, 2]], [1, 3]
            tasks.append({"target": noisy_target, "examples": noisy_examples})

    or_id = _find("OR")
    and_id = _find("AND")
    not_id = _find("NOT")
    add_id = _find("ADD")
    sub_id = _find("SUB")
    mul_id = _find("MUL")
    inc_id = _find("INC")
    dec_id = _find("DEC")
    square_id = _find("SQUARE")
    inverse_id = _find("MULTIPLICATIVE_INV")
    abs_id = _find("ABS")
    div_id = _find("DIV")  # <-- Added DIV 

    if or_id is not None: _add_task(or_id, [([2, 3], 2|3), ([1, 4], 1|4), ([0, 7], 0|7)])
    if and_id is not None: _add_task(and_id, [([2, 3], 2&3), ([1, 4], 1&4), ([7, 3], 7&3)])
    if not_id is not None: _add_task(not_id, [([0], ~0), ([1], ~1), ([5], ~5)])
    if add_id is not None: _add_task(add_id, [([0, 0], 0), ([1, 2], 3), ([5, 3], 8)])
    if sub_id is not None: _add_task(sub_id, [([5, 2], 3), ([10, 3], 7), ([7, 7], 0)])
    if mul_id is not None: _add_task(mul_id, [([0, 5], 0), ([1, 5], 5), ([2, 3], 6)])
    if inc_id is not None: _add_task(inc_id, [([0], 1), ([1], 2), ([5], 6)])
    if dec_id is not None: _add_task(dec_id, [([1], 0), ([2], 1), ([6], 5)])
    
    # FIX: Use higher numbers so SQUARE distinctively stands out from INC/MUL
    if square_id is not None: _add_task(square_id, [([2], 4), ([3], 9), ([5], 25)])
    
    if inverse_id is not None: _add_task(inverse_id, [([1], 1.0), ([-2], -0.5), ([4], 0.25)])
    
    if abs_id is not None: _add_task(abs_id, [([-3], 3), ([0], 0), ([5], 5)])
    
    # ADDED: Network needs to know what DIV is so it can divide by 2 for Kinetic Energy (0.5)
    if div_id is not None: _add_task(div_id, [([10, 2], 5.0), ([9, 3], 3.0), ([5, 2], 2.5)])

    # --- Compositional Base ---
    if inc_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": inc_id, "secondary_id": mul_id, "tertiary_id": None, "comp_type": "sequential"},
            "examples": [([2, 3], 7), ([1, 5], 6), ([4, 2], 9)],
        })
    if add_id is not None and inc_id is not None:
        tasks.append({
            "target": {"primary_id": add_id, "secondary_id": inc_id, "tertiary_id": None, "comp_type": "nested"},
            "examples": [([1, 2], 5), ([3, 4], 9), ([0, 0], 2)],
        })
    if add_id is not None and inc_id is not None and mul_id is not None:
        tasks.append({
            "target": {"primary_id": inc_id, "secondary_id": mul_id, "tertiary_id": add_id, "comp_type": "parallel", "routing": [[0], [1, 2]]},
            "examples": [([1, 2, 3], 8), ([0, 3, 4], 13), ([2, 1, 5], 8)],
        })
        
    _rng.setstate(_rng_state)
    return tasks

TARGETS = [
    ("NAND", [([0, 0], ~(0&0)), ([0, 1], ~(0&1)), ([1, 0], ~(1&0)), ([1, 1], ~(1&1))]),
    ("XOR",  [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]),
]


# ---------------------------------------------------------------------------
# Checkpointing & Logging
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer: torch.optim.Optimizer, state: dict):
    path = os.path.join(CONFIG["checkpoint_dir"], "model.pt")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "n_functions": reg.vocab_size(state),
    }, path)
    print(f"  checkpoint saved to {path}")


def load_checkpoint(model, optimizer: torch.optim.Optimizer, state: dict):
    path = os.path.join(CONFIG["checkpoint_dir"], "model.pt")
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, weights_only=False)
        old_n = ckpt["n_functions"]
        new_n = reg.vocab_size(state)
        if old_n == new_n:
            model.load_state_dict(ckpt["model_state"], strict=False)
            try: optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception: pass
        elif old_n < new_n:
            tmp = create_model(input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                               d_model=CONFIG["d_model"], n_functions=old_n,
                               n_layers=CONFIG["n_layers"], n_recursions=CONFIG["n_recursions"], T=CONFIG["T"])
            tmp.load_state_dict(ckpt["model_state"], strict=False)
            resize_heads(tmp, old_n, new_n)
            model.load_state_dict(tmp.state_dict(), strict=False)
        else:
            print(f"  Stale checkpoint ({old_n} > {new_n} funcs), starting fresh.")
            return False
        print(f"  checkpoint loaded ({old_n} -> {new_n} functions)")
        return True
    except Exception as e:
        print(f"  Warning: could not load checkpoint: {e}")
        return False

def _fmt_candidate(state: dict, c: dict) -> str:
    p = reg.get_name(state, c["primary_id"])
    s = reg.get_name(state, c["secondary_id"]) if c.get("secondary_id") is not None else None
    t = reg.get_name(state, c["tertiary_id"]) if c.get("tertiary_id") is not None else None

    if c["comp_type"] == "none": return p
    if c["comp_type"] == "loop_direct": return f"LOOP({s}, count=b, init=a)"
    if c["comp_type"] == "loop_binary": return f"LOOP({s}, count=b, init=0, step=a)"
    if c["comp_type"] == "sequential": return f"{p}({s}(...))"
    if c["comp_type"] == "nested": return f"{p}({s}(x), {s}(y))"
    if c["comp_type"] == "parallel": return f"{t}({p}(...), {s}(...))" if t else f"parallel({p}, {s})"
    return str(c)


# ---------------------------------------------------------------------------
# Execution Entry
# ---------------------------------------------------------------------------

def main():
    print("NSSR — Neuro-Symbolic Recursive Regression\n")

    conn = db.init_db(os.path.join(CONFIG["checkpoint_dir"], "symbolic.db"))
    state = reg.init_registry(conn)
    print(f"primitives: {list(reg.get_names(state, list(state['metadata'].keys())))}\n")

    n_funcs = reg.vocab_size(state)
    model = create_model(input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                         d_model=CONFIG["d_model"], n_functions=n_funcs,
                         n_layers=CONFIG["n_layers"], n_recursions=CONFIG["n_recursions"], T=CONFIG["T"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    load_checkpoint(model, optimizer, state)

    print("\n--- Curriculum pre-training ---")
    for task in curriculum_tasks(state):
        train_on_examples(model, optimizer, task["examples"], task["target"],
                          input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                          num_epochs=20, n_sup=CONFIG["n_sup"])

    print("\n--- PART 2: NOISY / CORRELATED COLUMNS ---")
    
    # Step 4: KE with a correlated noise column
    print("\n[Step 4] KE(m,v) = 0.5*m*v² with correlated noise column [m, noise_correlated, v]")
    ke_train = [([float(m), noisy_correlated(m), float(v)], 0.5 * m * v * v) 
                for m in [1, 2, 3, 4, 5] for v in [1, 2, 3, 4, 5]]
    _, optimizer, _ = learn(conn, state, model, optimizer, "KE_N", ke_train[:25], 
                            max_depth=5, num_epochs=50, max_search_steps=100)
    
    """
    # Step 5: PE with junk columns on both sides
    print("\n[Step 5] PE(m,h) = m*9.81*h with junk on both sides [junk, m, junk, h]")
    g = 9.81
    pe_train = [([junk(), float(m), junk(), float(h)], m * g * h) 
                for m in [1, 2, 3, 4, 5] for h in [1, 2, 3, 4, 5, 6]]
    _, optimizer, _ = learn(conn, state, model, optimizer, "PE_N", pe_train[:30], 
                            max_depth=5, num_epochs=50, max_search_steps=100)


    print("\n--- PART 3: SYNTHESIS WITH NOISE ---")

    # Step 6: TOTAL_E with noise
    print("\n[Step 6] E(m,v,h) = KE + PE with junk columns [m, junk, v, junk, h]")
    energy_train = []
    for m in [1, 2, 3, 4]:
        for v in [1, 2, 3, 4]:
            for h in [1, 2, 3]:
                ke = 0.5 * m * v * v
                pe = m * g * h
                energy_train.append(([float(m), junk(), float(v), junk(), float(h)], ke + pe))
                
    _, optimizer, _ = learn(conn, state, model, optimizer, "TOTAL_E_N", energy_train[:48], 
                            max_depth=5, num_epochs=60, max_search_steps=100)
    """
    print("\n--- Experiment Complete ---")
    save_checkpoint(model, optimizer, state)
    db.print_summary(conn)
    conn.close()

if __name__ == "__main__":
    random.seed(42)
    main()