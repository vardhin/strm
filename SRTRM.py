import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Callable, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from common import trunc_normal_init_
from layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, 
    CosSin, CastedEmbedding, CastedLinear
)
from sparse_embedding import CastedSparseEmbedding

# ==========================================
# 1. Symbolic Abstraction Layer (The "Memory")
# ==========================================

class SymbolicRegistry:
    """
    Manages the available functions (primitives + learned abstractions).
    Acts as the persistent storage for abstraction layers.
    """
    def __init__(self):
        self.functions: Dict[int, Callable] = {}
        self.metadata: Dict[int, Dict] = {}
        self.next_id = 0
        
        # Initialize Layer 0: Logical Primitives
        self.register("OR", lambda x, y: x | y, arity=2, layer=0)
        self.register("AND", lambda x, y: x & y, arity=2, layer=0)
        self.register("NOT", lambda x: ~x & 1, arity=1, layer=0)

    def register(self, name: str, func: Callable, arity: int, layer: int = -1) -> int:
        """Saves a new function to the registry."""
        fid = self.next_id
        self.functions[fid] = func
        if layer == -1:
            layer = self._compute_layer(func)
        self.metadata[fid] = {"name": name, "arity": arity, "layer": layer}
        self.next_id += 1
        return fid

    def _compute_layer(self, func: Callable) -> int:
        """Compute abstraction layer based on function composition."""
        if not self.metadata:
            return 0
        return max(m['layer'] for m in self.metadata.values()) + 1

    def get_vocab_size(self):
        return self.next_id

    def execute_function(self, func_id: int, args: List[Any]) -> Any:
        """Execute a single function by ID."""
        func = self.functions[func_id]
        return func(*args)

# ==========================================
# 2. TRM-Based Architecture (Following Original)
# ==========================================

@dataclass
class SymbolicTRMCarry:
    z_H: torch.Tensor  # High-level reasoning state
    z_L: torch.Tensor  # Low-level reasoning state
    steps: torch.Tensor
    halted: torch.Tensor

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

class SymbolicTRMCore(nn.Module):
    """Core TRM following the original nested H/L cycle structure."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config['forward_dtype'])
        
        # Embeddings
        self.embed_scale = math.sqrt(config['hidden_size'])
        embed_init_std = 1.0 / self.embed_scale
        
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

# ==========================================
# 3. The Symbolic Reasoning Agent
# ==========================================

class SymbolicAgent:
    def __init__(self, registry: SymbolicRegistry, d_model=128, max_recursion=10):
        self.registry = registry
        self.d_model = d_model
        self.max_recursion = max_recursion
        
        # Configuration matching original TRM structure
        self.config = {
            'input_dim': 12,  # 4 examples * 3 values
            'seq_len': 4,     # Number of I/O examples
            'hidden_size': d_model,
            'expansion': 2.0,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'rms_norm_eps': 1e-5,
            'rope_theta': 10000.0,
            'forward_dtype': 'float32',
            'max_functions': registry.get_vocab_size(),
            'H_cycles': 2,    # High-level reasoning cycles
            'L_cycles': 2,    # Low-level reasoning cycles
            'L_layers': 2,    # Number of transformer layers
            'halt_max_steps': max_recursion,
        }
        
        self.model = SymbolicTRMCore(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.training_history = []

    def initial_carry(self, batch_size: int) -> SymbolicTRMCarry:
        """Initialize carry state."""
        return SymbolicTRMCarry(
            z_H=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            z_L=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool)
        )

    def resize_model_for_new_abstraction(self):
        """Dynamically resize output heads when new function is learned."""
        new_vocab_size = self.registry.get_vocab_size()
        self.config['max_functions'] = new_vocab_size
        
        for head_name in ['head_primary_op', 'head_secondary_op']:
            old_head = getattr(self.model, head_name)
            old_vocab_size = old_head.weight.shape[0]  # Get size from weight shape
            
            new_head = CastedLinear(self.d_model, new_vocab_size, bias=False)
            
            with torch.no_grad():
                # Copy existing weights
                new_head.weight[:old_vocab_size] = old_head.weight
                
                # Initialize new function with semantic prior
                avg_weight = old_head.weight.mean(dim=0)
                new_head.weight[old_vocab_size:] = avg_weight.unsqueeze(0) + torch.randn(
                    new_vocab_size - old_vocab_size, self.d_model
                ) * 0.01
            
            setattr(self.model, head_name, new_head)
        
        print(f"  [Resize] Model vocabulary expanded to {new_vocab_size} functions")

    def format_examples(self, examples: List[Tuple[List[int], Any]]) -> torch.Tensor:
        """Convert I/O examples to model input format."""
        data = []
        for inputs, output in examples:
            data.extend(inputs)
            if isinstance(output, tuple):
                data.extend(output)
            else:
                data.append(output)
        
        # Pad or truncate
        if len(data) < self.config['input_dim']:
            data.extend([0] * (self.config['input_dim'] - len(data)))
        data = data[:self.config['input_dim']]
        
        return torch.tensor([data], dtype=torch.float32)

    def execute_program(self, primary_id: int, secondary_id: Optional[int], 
                       comp_type: str, inputs: List[int]) -> Any:
        """Execute a symbolic program on given inputs."""
        primary_meta = self.registry.metadata[primary_id]
        
        if comp_type == 'none':
            # Handle arity correctly
            if primary_meta['arity'] == 1:
                return self.registry.execute_function(primary_id, [inputs[0]])
            else:
                return self.registry.execute_function(primary_id, inputs)
                
        elif comp_type == 'sequential':
            # f2(f1(x, y), ...)
            result1 = self.registry.execute_function(primary_id, inputs)
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                return self.registry.execute_function(secondary_id, [result1])
            else:
                return self.registry.execute_function(secondary_id, [result1] + inputs[1:])
                
        elif comp_type == 'nested':
            # f1(x, f2(y, z)) or f1(x, f2(x))
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                # For NOT: f1(x, f2(x))
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            elif secondary_meta['arity'] == 2 and len(inputs) >= 2:
                # For binary ops: f1(x, f2(x, y))
                result2 = self.registry.execute_function(secondary_id, inputs)
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            else:
                return None
                
        elif comp_type == 'parallel':
            result1 = self.registry.execute_function(primary_id, inputs)
            result2 = self.registry.execute_function(secondary_id, inputs)
            return (result1, result2)
            
        return None

    def validate_program(self, primary_id: int, secondary_id: Optional[int],
                        comp_type: str, examples: List[Tuple[List[int], Any]]) -> bool:
        """Validate a symbolic program against examples."""
        for inputs, expected_output in examples:
            try:
                result = self.execute_program(primary_id, secondary_id, comp_type, inputs)
                if result != expected_output:
                    return False
            except (IndexError, TypeError, KeyError):
                return False
        return True

    def exhaustive_search(self, examples: List[Tuple[List[int], Any]]) -> Optional[Dict]:
        """Exhaustive search over all possible programs."""
        vocab_size = self.registry.get_vocab_size()
        
        # Try single operations first
        for primary_id in range(vocab_size):
            if self.validate_program(primary_id, 0, 'none', examples):
                return {
                    'primary_id': primary_id,
                    'secondary_id': 0,
                    'comp_type': 'none',
                    'score': 1.0,
                    'step': 0
                }
        
        # Try all compositions
        for primary_id in range(vocab_size):
            for secondary_id in range(vocab_size):
                for comp_type in ['sequential', 'nested', 'parallel']:
                    if self.validate_program(primary_id, secondary_id, comp_type, examples):
                        return {
                            'primary_id': primary_id,
                            'secondary_id': secondary_id,
                            'comp_type': comp_type,
                            'score': 1.0,
                            'step': 0
                        }
        
        return None

    def search_with_recursive_trm(self, examples: List[Tuple[List[int], Any]], 
                                  task_name: str = "Unknown") -> Optional[Dict]:
        """
        Use TRM's recursive reasoning to find valid program.
        Falls back to exhaustive search if needed.
        """
        print(f"\n--- Searching for: {task_name} ---")
        
        # First try exhaustive search (for bootstrapping)
        program = self.exhaustive_search(examples)
        if program is not None:
            print(f"  [Success] Found via exhaustive search")
            print(f"    Primary: {self.registry.metadata[program['primary_id']]['name']}")
            if program['comp_type'] != 'none':
                print(f"    Secondary: {self.registry.metadata[program['secondary_id']]['name']}")
            print(f"    Composition: {program['comp_type']}")
            return program
        
        # Then try TRM-based search
        x_input = self.format_examples(examples)
        carry = self.initial_carry(batch_size=1)
        carry = self.model.reset_carry(carry.halted, carry)
        
        self.model.eval()
        candidates = []
        
        with torch.no_grad():
            for step in range(self.config['halt_max_steps']):
                carry, outputs = self.model(carry, x_input)
                
                # Extract predictions
                primary_probs = F.softmax(outputs['primary_logits'], dim=-1)
                secondary_probs = F.softmax(outputs['secondary_logits'], dim=-1)
                comp_probs = F.softmax(outputs['composition_logits'], dim=-1)
                
                # Get top candidates
                top_k = 3
                primary_top = torch.topk(primary_probs[0], min(top_k, primary_probs.shape[-1]))
                secondary_top = torch.topk(secondary_probs[0], min(top_k, secondary_probs.shape[-1]))
                comp_top = torch.topk(comp_probs[0], min(4, comp_probs.shape[-1]))
                
                for p_idx, p_prob in zip(primary_top.indices, primary_top.values):
                    for s_idx, s_prob in zip(secondary_top.indices, secondary_top.values):
                        for c_idx, c_prob in zip(comp_top.indices, comp_top.values):
                            comp_types = ['none', 'sequential', 'nested', 'parallel']
                            comp_type = comp_types[c_idx.item()]
                            
                            score = (p_prob * s_prob * c_prob * outputs['confidence'][0]).item()
                            
                            candidates.append({
                                'primary_id': p_idx.item(),
                                'secondary_id': s_idx.item(),
                                'comp_type': comp_type,
                                'score': score,
                                'step': step
                            })
                
                # Early halt if confident
                if outputs['confidence'][0] > 0.9:
                    break
        
        # Sort and validate
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        for candidate in candidates[:15]:
            if self.validate_program(
                candidate['primary_id'],
                candidate['secondary_id'],
                candidate['comp_type'],
                examples
            ):
                print(f"  [Success] Found via TRM at step {candidate['step']}")
                print(f"    Primary: {self.registry.metadata[candidate['primary_id']]['name']}")
                print(f"    Secondary: {self.registry.metadata[candidate['secondary_id']]['name']}")
                print(f"    Composition: {candidate['comp_type']}")
                return candidate
        
        print(f"  [Failed] No valid program found")
        return None

    def train_step(self, examples: List[Tuple[List[int], Any]], 
                   target_program: Dict) -> float:
        """Train with deep supervision across recursive steps."""
        self.model.train()
        x_input = self.format_examples(examples)
        carry = self.initial_carry(batch_size=1)
        carry = self.model.reset_carry(carry.halted, carry)
        
        total_loss = 0.0
        
        for step in range(min(3, self.config['halt_max_steps'])):  # Train on first few steps
            carry, outputs = self.model(carry, x_input)
            
            # Supervised losses
            primary_loss = F.cross_entropy(
                outputs['primary_logits'],
                torch.tensor([target_program['primary_id']])
            )
            secondary_loss = F.cross_entropy(
                outputs['secondary_logits'],
                torch.tensor([target_program['secondary_id']])
            )
            
            comp_types = ['none', 'sequential', 'nested', 'parallel']
            comp_target = comp_types.index(target_program['comp_type'])
            comp_loss = F.cross_entropy(
                outputs['composition_logits'],
                torch.tensor([comp_target])
            )
            
            # Halt loss (should halt after finding solution)
            halt_loss = F.binary_cross_entropy_with_logits(
                outputs['halt_logits'],
                torch.ones_like(outputs['halt_logits'])
            )
            
            step_loss = primary_loss + secondary_loss + comp_loss + 0.1 * halt_loss
            total_loss += step_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def learn_abstraction(self, name: str, examples: List[Tuple[List[int], Any]], 
                         num_epochs: int = 50) -> bool:
        """Full pipeline: Search -> Train -> Save -> Resize"""
        print(f"\n{'='*60}")
        print(f"Learning Abstraction: {name}")
        print(f"{'='*60}")
        
        # Search
        program = self.search_with_recursive_trm(examples, task_name=name)
        
        if program is None:
            return False
        
        # Train
        print(f"\n  [Training] Refining weights...")
        for epoch in range(num_epochs):
            loss = self.train_step(examples, program)
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss = {loss:.4f}")
        
        # Save
        def learned_function(*args):
            return self.execute_program(
                program['primary_id'],
                program['secondary_id'],
                program['comp_type'],
                list(args)
            )
        
        arity = len(examples[0][0])
        new_id = self.registry.register(name, learned_function, arity=arity)
        print(f"\n  [Memory] Saved '{name}' as Function ID {new_id}")
        
        # Resize
        self.resize_model_for_new_abstraction()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.training_history.append({
            'name': name,
            'id': new_id,
            'program': program,
            'examples': examples
        })
        
        return True

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    print("="*60)
    print("Symbolic Regression with TRM Architecture")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8)
    
    print(f"\nInitial: {[m['name'] for m in registry.metadata.values()]}")
    
    # Learn NAND first (simpler)
    print("\n" + "="*60)
    print("Testing exhaustive search manually")
    print("="*60)
    nand_examples = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    
    # Test all compositions
    for p_id in range(registry.get_vocab_size()):
        for s_id in range(registry.get_vocab_size()):
            for comp in ['none', 'sequential', 'nested']:
                if agent.validate_program(p_id, s_id, comp, nand_examples):
                    print(f"Found NAND: primary={registry.metadata[p_id]['name']}, "
                          f"secondary={registry.metadata[s_id]['name']}, comp={comp}")
    
    # Now learn NAND
    success = agent.learn_abstraction("NAND", nand_examples, num_epochs=30)
    
    if success:
        # Test NAND
        print("\n" + "="*60)
        print("Testing NAND")
        print("="*60)
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        for inputs, expected in nand_examples:
            result = registry.execute_function(nand_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} NAND{tuple(inputs)} = {result}")
    
    # XOR requires more complex composition - skip for now
    print("\n" + "="*60)
    print("Note: XOR requires 3+ function composition")
    print("XOR = (A OR B) AND NOT(A AND B)")
    print("="*60)

if __name__ == "__main__":
    main()