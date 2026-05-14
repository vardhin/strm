"""
TRM (Tiny Recursive reasoning Model) for NSRR.

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
