"""
TRM that searches through primitive combinations in a scratchpad
Handles multi-input, multi-output programs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class NeurosymbolicTRM(nn.Module):
    """
    TRM with better exploration and gradient stability
    """
    def __init__(self, 
                 primitives_manager,
                 embed_dim: int = 128,
                 num_layers: int = 2,
                 n_recursions: int = 6,
                 T_deep: int = 3,
                 max_program_len: int = 20):
        super().__init__()
        
        self.prims = primitives_manager
        self.embed_dim = embed_dim
        self.vocab_size = primitives_manager.get_vocab_size()
        self.max_program_len = max_program_len
        self.n_recursions = n_recursions
        self.T_deep = T_deep
        
        # Input embedding (problem = [arg_0, arg_1, ..., target_0, ...])
        self.problem_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Token embeddings for program
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_program_len, embed_dim) * 0.02)
        
        # Transformer for reasoning
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim) for _ in range(num_layers)
        ])
        
        # Scratchpad initializer
        self.scratchpad_init = nn.Linear(embed_dim, embed_dim)
        
        # Program generator (with exploration bonus)
        self.program_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.vocab_size)
        )
        
        # Satisfaction and halt predictors
        self.satisfaction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.halt_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature for exploration (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent gradient collapse"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def latent_recursion(self, x_problem, y_program, z_scratchpad):
        """
        One cycle of latent refinement (n recursions)
        Returns updated scratchpad
        """
        # Combine problem, current program, and scratchpad
        combined = z_scratchpad + x_problem.mean(dim=1, keepdim=True) + y_program.mean(dim=1, keepdim=True)
        
        # Apply transformer layers n times
        for _ in range(self.n_recursions):
            for layer in self.transformer:
                combined = layer(combined)
        
        return combined
    
    def deep_recursion(self, x_problem, y_program, z_scratchpad):
        """
        T-1 recursions without grad, then 1 with grad
        Returns updated (program, scratchpad)
        """
        # T-1 cycles without gradients
        with torch.no_grad():
            for _ in range(self.T_deep - 1):
                z_scratchpad = self.latent_recursion(x_problem, y_program, z_scratchpad)
        
        # Detach to cut gradient flow
        z_scratchpad = z_scratchpad.detach()
        
        # Final cycle with gradients
        z_scratchpad = self.latent_recursion(x_problem, y_program, z_scratchpad)
        
        return z_scratchpad
    
    def forward(self, problem_data, current_program_ids=None, z_scratchpad=None):
        """
        One refinement step
        
        Args:
            problem_data: [batch, problem_len] tensor of floats
            current_program_ids: [batch, max_program_len] tensor of token IDs (optional)
            z_scratchpad: [batch, max_program_len, embed_dim] (optional)
        
        Returns:
            program_logits: [batch, max_program_len, vocab_size]
            z_scratchpad: [batch, max_program_len, embed_dim]
            satisfaction: [batch, 1] probability that program is correct
            q_halt: [batch, 1] probability to stop refining
        """
        batch_size = problem_data.size(0)
        
        # Encode problem (each value separately)
        # problem_data shape: [batch, problem_len]
        x_problem = self.problem_encoder(problem_data.unsqueeze(-1))  # [batch, problem_len, embed_dim]
        
        # Initialize program representation if needed
        if current_program_ids is None:
            # Start with random program
            y_program = self.token_embedding.weight[0:1].expand(batch_size, self.max_program_len, -1)
            program_len = self.max_program_len
        else:
            # Use current program - pad or truncate to max_program_len
            program_len = current_program_ids.size(1)
            
            # Ensure current_program_ids matches max_program_len
            if program_len < self.max_program_len:
                # Pad with PAD tokens
                pad_id = self.prims.token_to_id.get("PAD", 0)
                padding = torch.full((batch_size, self.max_program_len - program_len), 
                                    pad_id, 
                                    dtype=current_program_ids.dtype, 
                                    device=current_program_ids.device)
                current_program_ids = torch.cat([current_program_ids, padding], dim=1)
            elif program_len > self.max_program_len:
                # Truncate
                current_program_ids = current_program_ids[:, :self.max_program_len]
            
            y_program = self.token_embedding(current_program_ids)  # [batch, max_program_len, embed_dim]
        
        # Add positional encoding - now sizes match
        y_program = y_program + self.pos_encoding
        
        # Initialize scratchpad if needed
        if z_scratchpad is None:
            z_scratchpad = self.scratchpad_init(x_problem.mean(dim=1, keepdim=True)).expand(-1, self.max_program_len, -1)
        
        # Deep recursion to refine scratchpad
        z_scratchpad = self.deep_recursion(x_problem, y_program, z_scratchpad)
        
        # Generate program tokens from scratchpad
        # Add entropy bonus for exploration
        program_logits = self.program_head(z_scratchpad)  # [batch, max_program_len, vocab_size]
        
        # Clamp temperature to prevent it from going to 0
        temp = torch.clamp(self.temperature, min=0.5, max=5.0)
        program_logits = program_logits / temp
        
        # Compute satisfaction (how good is this program?)
        # Pool over sequence dimension, then predict
        pooled = z_scratchpad.mean(dim=1)  # [batch, embed_dim]
        satisfaction = self.satisfaction_head(pooled)  # [batch, 1]
        
        # Compute halt probability
        q_halt = self.halt_head(pooled)  # [batch, 1]
        
        return program_logits, z_scratchpad, satisfaction, q_halt


class TransformerBlock(nn.Module):
    """Minimal transformer block with better gradient flow"""
    def __init__(self, embed_dim, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x