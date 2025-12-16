import torch
import torch.nn as nn
import math
from typing import Tuple, Dict
from common import trunc_normal_init_
from layers import RotaryEmbedding, CastedLinear
from .carry import SymbolicTRMCarry
from .blocks import SymbolicTRMBlock, SymbolicTRMReasoningModule

class SymbolicTRMCore(nn.Module):
    """Core TRM following the original nested H/L cycle structure."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config['forward_dtype'])
        
        # Embeddings
        self.embed_scale = math.sqrt(config['hidden_size'])
        
        # Input/Output heads for symbolic reasoning
        self.input_proj = CastedLinear(config['input_dim'], config['hidden_size'], bias=False)
        
        # Function prediction heads
        self.head_primary_op = CastedLinear(config['hidden_size'], config['max_functions'], bias=False)
        self.head_secondary_op = CastedLinear(config['hidden_size'], config['max_functions'], bias=False)
        self.head_composition_type = CastedLinear(config['hidden_size'], 4, bias=False)
        
        # Halting head (like Q-head in original)
        self.halt_head = CastedLinear(config['hidden_size'], 1, bias=True)
        
        # Position encodings
        if config['pos_encodings'] == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config['hidden_size'] // config['num_heads'],
                max_position_embeddings=config['seq_len'],
                base=config.get('rope_theta', 10000.0)
            )
        
        # Reasoning layers
        self.L_level = SymbolicTRMReasoningModule(
            layers=[SymbolicTRMBlock(config) for _ in range(config['L_layers'])]
        )
        
        # Initial states
        self.H_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config['hidden_size'], dtype=self.forward_dtype), std=1.0)
        )
        self.L_init = nn.Parameter(
            trunc_normal_init_(torch.empty(config['hidden_size'], dtype=self.forward_dtype), std=1.0)
        )
        
        # Initialize halt head to start halted
        with torch.no_grad():
            self.halt_head.weight.zero_()
            self.halt_head.bias.fill_(-5.0)

    def _input_embeddings(self, input_tensor: torch.Tensor):
        """Project input examples to hidden size."""
        embedding = self.input_proj(input_tensor)
        return self.embed_scale * embedding

    def reset_carry(self, reset_flag: torch.Tensor, carry: SymbolicTRMCarry):
        """Reset carry for new sequences."""
        batch_size = reset_flag.shape[0]
        
        return SymbolicTRMCarry(
            z_H=torch.where(
                reset_flag.view(-1, 1, 1),
                self.H_init.expand(batch_size, self.config['seq_len'], -1),
                carry.z_H
            ),
            z_L=torch.where(
                reset_flag.view(-1, 1, 1),
                self.L_init.expand(batch_size, self.config['seq_len'], -1),
                carry.z_L
            ),
            steps=torch.where(reset_flag, torch.zeros_like(carry.steps), carry.steps),
            halted=reset_flag | carry.halted
        )

    def forward(self, carry: SymbolicTRMCarry, input_batch: torch.Tensor) -> Tuple[SymbolicTRMCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass following original TRM structure:
        - H_cycles-1 iterations without gradient
        - 1 iteration with gradient
        - Nested L_cycles for each H iteration
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # Input encoding
        input_embeddings = self._input_embeddings(input_batch)
        
        # Expand to sequence length if needed
        if input_embeddings.shape[1] == 1:
            input_embeddings = input_embeddings.expand(-1, self.config['seq_len'], -1)
        
        z_H, z_L = carry.z_H, carry.z_L
        
        # H_cycles-1 without gradient (efficiency optimization from original)
        with torch.no_grad():
            for _h_step in range(self.config['H_cycles'] - 1):
                # L_cycles for low-level reasoning
                for _l_step in range(self.config['L_cycles']):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                # Update H from L
                z_H = self.L_level(z_H, z_L, **seq_info)
        
        # Final H_cycle with gradient
        for _l_step in range(self.config['L_cycles']):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        
        # Output predictions (use first position for global prediction)
        primary_logits = self.head_primary_op(z_H[:, 0])
        secondary_logits = self.head_secondary_op(z_H[:, 0])
        composition_logits = self.head_composition_type(z_H[:, 0])
        halt_logits = self.halt_head(z_H[:, 0]).squeeze(-1)
        
        # New carry (detached for next iteration)
        new_carry = SymbolicTRMCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=carry.steps + 1,
            halted=carry.halted
        )
        
        # Update halting (during training)
        if self.training:
            with torch.no_grad():
                is_last_step = new_carry.steps >= self.config['halt_max_steps']
                halted = is_last_step | (halt_logits > 0)
                new_carry = SymbolicTRMCarry(
                    z_H=new_carry.z_H,
                    z_L=new_carry.z_L,
                    steps=new_carry.steps,
                    halted=halted
                )
        
        outputs = {
            'primary_logits': primary_logits,
            'secondary_logits': secondary_logits,
            'composition_logits': composition_logits,
            'halt_logits': halt_logits,
            'confidence': torch.sigmoid(halt_logits)
        }
        
        return new_carry, outputs