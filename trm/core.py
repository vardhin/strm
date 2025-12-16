import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass
from layers import CastedLinear, trunc_normal_init_


@dataclass
class SymbolicTRMCarry:
    """Carry state for the TRM."""
    z_H: torch.Tensor  # High-level reasoning state
    z_L: torch.Tensor  # Low-level computation state
    steps: torch.Tensor  # Step counter
    halted: torch.Tensor  # Halting mask


class TRMBlock(nn.Module):
    """Single transformer block for TRM."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            CastedLinear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            CastedLinear(d_ff, d_model, bias=True),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


class SymbolicTRMCore(nn.Module):
    """Core TRM model for symbolic reasoning."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Set dtype FIRST before using it
        self.forward_dtype = torch.float32
        
        d_model = config['hidden_size']
        n_heads = config.get('n_heads', 8)
        d_ff = config.get('d_ff', d_model * 4)
        dropout = config.get('dropout', 0.1)
        
        # Input projection
        self.input_proj = CastedLinear(config['input_dim'], d_model, bias=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TRMBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(config.get('n_layers', 6))
        ])
        
        # Layer norms
        self.ln_H = nn.LayerNorm(d_model)
        self.ln_L = nn.LayerNorm(d_model)
        
        # Positional embeddings
        max_seq_len = config.get('seq_len', 128)
        self.pos_emb = nn.Parameter(
            trunc_normal_init_(torch.empty(max_seq_len, d_model, dtype=self.forward_dtype), std=1.0)
        )
        
        # Mixing coefficients for carry update
        self.alpha_H = nn.Parameter(torch.tensor(0.5))
        self.alpha_L = nn.Parameter(torch.tensor(0.5))
        
        # Output heads
        self.head_primary_op = CastedLinear(d_model, config['max_functions'], bias=False)
        self.head_secondary_op = CastedLinear(d_model, config['max_functions'], bias=False)
        self.head_tertiary_op = CastedLinear(d_model, config['max_functions'], bias=False)
        self.head_composition = CastedLinear(d_model, 4, bias=False)  # none, seq, nested, parallel
        self.head_halt = CastedLinear(d_model, 1, bias=False)
        
        self.d_model = d_model
    
    def reset_carry(self, halted: torch.Tensor, carry: SymbolicTRMCarry) -> SymbolicTRMCarry:
        """Reset carry state for halted sequences."""
        batch_size = halted.shape[0]
        
        # Only reset if halted
        if halted.all():
            return SymbolicTRMCarry(
                z_H=torch.zeros_like(carry.z_H),
                z_L=torch.zeros_like(carry.z_L),
                steps=torch.zeros_like(carry.steps),
                halted=torch.ones_like(carry.halted)
            )
        
        return carry
    
    def forward(self, carry: SymbolicTRMCarry, x: torch.Tensor) -> Tuple[SymbolicTRMCarry, Dict[str, torch.Tensor]]:
        """Forward pass through TRM."""
        
        batch_size, seq_len, _ = x.shape
        
        # Project input and add positional embeddings
        x_proj = self.input_proj(x)
        x_proj = x_proj + self.pos_emb[:seq_len].unsqueeze(0)
        
        # Combine with carry state
        z_H = self.ln_H(carry.z_H + x_proj)
        z_L = self.ln_L(carry.z_L + x_proj)
        
        # Process through transformer blocks
        for block in self.blocks:
            z_H = block(z_H)
        
        # Update carry with mixing
        alpha_H = torch.sigmoid(self.alpha_H)
        alpha_L = torch.sigmoid(self.alpha_L)
        
        new_z_H = alpha_H * z_H + (1 - alpha_H) * carry.z_H
        new_z_L = alpha_L * z_L + (1 - alpha_L) * carry.z_L
        
        new_carry = SymbolicTRMCarry(
            z_H=new_z_H,
            z_L=new_z_L,
            steps=carry.steps + 1,
            halted=carry.halted
        )
        
        # Pool sequence for predictions (mean pooling)
        z_H_pooled = z_H.mean(dim=1)
        
        # Prediction heads
        primary_logits = self.head_primary_op(z_H_pooled)
        secondary_logits = self.head_secondary_op(z_H_pooled)
        tertiary_logits = self.head_tertiary_op(z_H_pooled)
        composition_logits = self.head_composition(z_H_pooled)
        halt_logits = self.head_halt(z_H_pooled)
        
        outputs = {
            'primary_logits': primary_logits,
            'secondary_logits': secondary_logits,
            'tertiary_logits': tertiary_logits,
            'composition_logits': composition_logits,
            'halt_logits': halt_logits
        }
        
        return new_carry, outputs