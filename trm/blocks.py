import torch
import torch.nn as nn
from typing import List
from layers import rms_norm, SwiGLU, Attention, CosSin

class SymbolicTRMBlock(nn.Module):
    """Single reasoning block following original TRM design."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = Attention(
            hidden_size=config['hidden_size'],
            head_dim=config['hidden_size'] // config['num_heads'],
            num_heads=config['num_heads'],
            num_key_value_heads=config['num_heads'],
            causal=False
        )
        
        self.mlp = SwiGLU(
            hidden_size=config['hidden_size'],
            expansion=config['expansion'],
        )
        
        self.norm_eps = config['rms_norm_eps']

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm architecture like original
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        
        return hidden_states


class SymbolicTRMReasoningModule(nn.Module):
    """Reasoning module with input injection like original."""
    
    def __init__(self, layers: List[SymbolicTRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Key: Addition-based injection, not concatenation
        hidden_states = hidden_states + input_injection
        
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        
        return hidden_states