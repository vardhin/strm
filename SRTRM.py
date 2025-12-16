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
        """
        Convert I/O examples to structured format with explicit I/O separation.
        Shape: [batch, num_examples, 3] where each row is [input1, input2, output]
        This lets the model attend to output patterns (e.g., XOR=[0,1,1,0] vs NAND=[1,1,1,0])
        
        For unary functions, we pad the second input with 0.
        """
        data = []
        for inputs, output in examples:
            # Pad unary inputs to binary (2 inputs)
            if len(inputs) == 1:
                padded_inputs = inputs + [0]  # NOT(x) -> [x, 0]
            else:
                padded_inputs = inputs[:2]  # Take first 2 for binary ops
            
            # Each example is a row: [input1, input2, output]
            data.append(padded_inputs + [output])
        
        # Pad to fixed size (4 examples)
        while len(data) < self.config['seq_len']:
            data.append([0, 0, 0])
        
        # Flatten for model input: [batch, seq_len * 3]
        flat_data = [val for row in data for val in row]
        return torch.tensor([flat_data], dtype=torch.float32)

    def format_examples_with_attention(self, examples):
        """Encode examples with explicit I/O separation."""
        # Shape: [batch, num_examples, 3] where 3 = [input1, input2, output]
        data = []
        for inputs, output in examples:
            data.append(inputs + [output])
        
        # Pad to fixed size
        while len(data) < self.config['seq_len']:
            data.append([0, 0, 0])
        
        return torch.tensor([data], dtype=torch.float32)

    def execute_program(self, primary_id: int, secondary_id: Optional[int], 
                       comp_type: str, inputs: List[int], tertiary_id: Optional[int] = None) -> Any:
        """Execute a symbolic program on given inputs (now supports 3-function composition)."""
        primary_meta = self.registry.metadata[primary_id]
        
        if comp_type == 'none':
            # Single function
            if primary_meta['arity'] == 1:
                return self.registry.execute_function(primary_id, [inputs[0]])
            else:
                return self.registry.execute_function(primary_id, inputs)
                
        elif comp_type == 'sequential':
            # f2(f1(x, y))
            result1 = self.registry.execute_function(primary_id, inputs)
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                return self.registry.execute_function(secondary_id, [result1])
            else:
                return self.registry.execute_function(secondary_id, [result1] + inputs[1:])
                
        elif comp_type == 'nested':
            # f1(x, f2(x, y))
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            elif secondary_meta['arity'] == 2 and len(inputs) >= 2:
                result2 = self.registry.execute_function(secondary_id, inputs)
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            else:
                return None
                
        elif comp_type == 'parallel':
            # f_tertiary(f_primary(x, y), f_secondary(x, y))
            # This enables: XOR = AND(OR(x,y), NAND(x,y))
            
            # Execute primary and secondary - handle arity properly
            primary_meta = self.registry.metadata[primary_id]
            secondary_meta = self.registry.metadata[secondary_id]
            
            # Get correct number of arguments for each function
            if primary_meta['arity'] == 1:
                result1 = self.registry.execute_function(primary_id, [inputs[0]])
            else:
                result1 = self.registry.execute_function(primary_id, inputs[:primary_meta['arity']])
                
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
            else:
                result2 = self.registry.execute_function(secondary_id, inputs[:secondary_meta['arity']])
            
            # If tertiary_id specified, use it as combiner
            if tertiary_id is not None:
                tertiary_meta = self.registry.metadata[tertiary_id]
                if tertiary_meta['arity'] == 2:
                    return self.registry.execute_function(tertiary_id, [result1, result2])
                elif tertiary_meta['arity'] == 1:
                    # Unary combiner - just return first result
                    return self.registry.execute_function(tertiary_id, [result1])
            
            # Default: try AND as combiner for backward compatibility
            if isinstance(result1, int) and isinstance(result2, int):
                return result1 & result2
            else:
                return (result1, result2)
            
        return None

    def validate_program(self, primary_id: int, secondary_id: Optional[int],
                        comp_type: str, examples: List[Tuple[List[int], Any]], 
                        tertiary_id: Optional[int] = None) -> bool:
        """Validate a symbolic program against examples."""
        for inputs, expected_output in examples:
            try:
                result = self.execute_program(primary_id, secondary_id, comp_type, inputs, tertiary_id)
                if result != expected_output:
                    return False
            except (IndexError, TypeError, KeyError, Exception):
                return False
        return True

    def exhaustive_search(self, examples: List[Tuple[List[int], Any]]) -> Optional[Dict]:
        """Exhaustive search with 3-function compositions."""
        vocab_size = self.registry.get_vocab_size()
        
        # Try single operations
        for primary_id in range(vocab_size):
            if self.validate_program(primary_id, 0, 'none', examples):
                return {
                    'primary_id': primary_id,
                    'secondary_id': 0,
                    'tertiary_id': None,
                    'comp_type': 'none',
                    'score': 1.0,
                    'step': 0
                }
        
        # Try 2-function compositions
        for primary_id in range(vocab_size):
            for secondary_id in range(vocab_size):
                for comp_type in ['sequential', 'nested']:
                    if self.validate_program(primary_id, secondary_id, comp_type, examples):
                        return {
                            'primary_id': primary_id,
                            'secondary_id': secondary_id,
                            'tertiary_id': None,
                            'comp_type': comp_type,
                            'score': 1.0,
                            'step': 0
                        }
        
        # Try 3-function parallel composition: tertiary(primary(x,y), secondary(x,y))
        for primary_id in range(vocab_size):
            for secondary_id in range(vocab_size):
                for tertiary_id in range(vocab_size):
                    if self.validate_program(primary_id, secondary_id, 'parallel', examples, tertiary_id):
                        return {
                            'primary_id': primary_id,
                            'secondary_id': secondary_id,
                            'tertiary_id': tertiary_id,
                            'comp_type': 'parallel',
                            'score': 1.0,
                            'step': 0
                        }
        
        return None

    def generate_curriculum_tasks(self) -> List[Tuple[str, List[Tuple[List[int], Any]], Dict]]:
        """Generate synthetic tasks to teach compositional search."""
        tasks = []
        
        # Task 1: Teach basic single-function usage
        tasks.append((
            "OR_identity",
            [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
            {'primary_id': 0, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        tasks.append((
            "AND_identity", 
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 1, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        tasks.append((
            "NOT_identity",
            [([0], 1), ([1], 0)],  # Unary function examples
            {'primary_id': 2, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        # Task 2: Teach sequential composition (builds on primitives)
        tasks.append((
            "NOT_OR",  # NOT(OR(x,y)) = NOR
            [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)],
            {'primary_id': 0, 'secondary_id': 2, 'tertiary_id': None, 'comp_type': 'sequential', 'step': 0}
        ))
        
        tasks.append((
            "NOT_AND",  # NOT(AND(x,y)) = NAND
            [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
            {'primary_id': 1, 'secondary_id': 2, 'tertiary_id': None, 'comp_type': 'sequential', 'step': 0}
        ))
        
        # Task 3: Teach parallel composition MORE EXPLICITLY
        # Use a simpler example: AND(OR(x,y), AND(x,y)) = AND (identity check)
        tasks.append((
            "parallel_identity",  # AND(OR(x,y), AND(x,y))
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 0, 'secondary_id': 1, 'tertiary_id': 1, 'comp_type': 'parallel', 'step': 0}
        ))
        
        # Task 4: OR as combiner (critical for XOR!)
        # OR(AND(x,y), AND(x,y)) - another identity to teach OR combiner
        tasks.append((
            "parallel_or_combiner",
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 1, 'secondary_id': 1, 'tertiary_id': 0, 'comp_type': 'parallel', 'step': 0}
        ))
        
        return tasks

    def search_with_recursive_trm(self, examples: List[Tuple[List[int], Any]], 
                                  task_name: str = "Unknown",
                                  exploration_bonus: float = 0.5) -> Optional[Dict]:  # Increase from 0.3
        """
        Use TRM's recursive reasoning with:
        1. Learned heuristics
        2. STRONGER exploration bonuses (penalize recent predictions)
        3. Top-down layer prioritization
        """
        print(f"\n--- Searching for: {task_name} ---")
        
        # Only use exhaustive search for very early bootstrapping (just primitives)
        if self.registry.get_vocab_size() <= 3:
            print(f"  [Bootstrap] Using exhaustive search (vocab size = {self.registry.get_vocab_size()})")
            program = self.exhaustive_search(examples)
            if program is not None:
                print(f"  [Success] Found via exhaustive search")
                print(f"    Primary: {self.registry.metadata[program['primary_id']]['name']}")
                if program['comp_type'] != 'none':
                    print(f"    Secondary: {self.registry.metadata[program['secondary_id']]['name']}")
                if program['tertiary_id'] is not None:
                    print(f"    Tertiary: {self.registry.metadata[program['tertiary_id']]['name']}")
                print(f"    Composition: {program['comp_type']}")
                return program
        
        # TRM-based learned search with semantic priors
        print(f"  [TRM Search] Using learned heuristics (vocab size = {self.registry.get_vocab_size()})...")
        x_input = self.format_examples(examples)
        carry = self.initial_carry(batch_size=1)
        carry = self.model.reset_carry(carry.halted, carry)
        
        self.model.eval()
        candidates = []
        
        # Build layer-based priors (top-down preference)
        layer_functions = {}
        for fid, meta in self.registry.metadata.items():
            layer = meta['layer']
            if layer not in layer_functions:
                layer_functions[layer] = []
            layer_functions[layer].append(fid)
        
        # Penalize recently used functions MORE AGRESSIVELY
        recent_usage = {}
        for i, history_item in enumerate(self.training_history[-5:]):  # Look at last 5 tasks (was 3)
            prog = history_item['program']
            # Decay penalty: most recent gets highest penalty
            decay = (len(self.training_history[-5:]) - i) / len(self.training_history[-5:])
            recent_usage[prog['primary_id']] = recent_usage.get(prog['primary_id'], 0) + decay
            recent_usage[prog['secondary_id']] = recent_usage.get(prog['secondary_id'], 0) + decay
            if prog.get('tertiary_id') is not None:
                recent_usage[prog['tertiary_id']] = recent_usage.get(prog['tertiary_id'], 0) + decay
        
        print(f"    Available functions by layer:")
        for layer in sorted(layer_functions.keys(), reverse=True):  # Show top-down
            func_names = [self.registry.metadata[fid]['name'] for fid in layer_functions[layer]]
            print(f"      Layer {layer}: {func_names}")
        
        if recent_usage:
            print(f"    Recent usage penalties (decay-weighted):")
            for fid, count in sorted(recent_usage.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      {self.registry.metadata[fid]['name']}: -{exploration_bonus * count:.3f}")
        
        with torch.no_grad():
            for step in range(self.config['halt_max_steps']):
                carry, outputs = self.model(carry, x_input)
                
                primary_probs = F.softmax(outputs['primary_logits'], dim=-1)
                secondary_probs = F.softmax(outputs['secondary_logits'], dim=-1)
                comp_probs = F.softmax(outputs['composition_logits'], dim=-1)
                
                # Apply STRONGER exploration bonus (penalize recent predictions)
                primary_probs_adjusted = primary_probs.clone()
                secondary_probs_adjusted = secondary_probs.clone()
                
                for fid, count in recent_usage.items():
                    penalty = exploration_bonus * count
                    # Cap penalty at 0.9 to avoid complete suppression
                    penalty = min(penalty, 0.9)
                    primary_probs_adjusted[0, fid] *= (1 - penalty)
                    secondary_probs_adjusted[0, fid] *= (1 - penalty)
                
                # Renormalize after penalty
                primary_probs_adjusted = primary_probs_adjusted / primary_probs_adjusted.sum()
                secondary_probs_adjusted = secondary_probs_adjusted / secondary_probs_adjusted.sum()
                
                print(f"\n    [Step {step}] TRM Predictions:")
                print(f"      Confidence: {outputs['confidence'][0].item():.4f}")
                
                # Show top-3 for each prediction (after exploration bonus)
                vocab_size = self.registry.get_vocab_size()
                print(f"      Top-3 Primary (after exploration bonus):")
                top_primary = torch.topk(primary_probs_adjusted[0], min(3, vocab_size))
                for i in range(len(top_primary.indices)):
                    idx = top_primary.indices[i].item()
                    prob = top_primary.values[i].item()
                    original_prob = primary_probs[0, idx].item()
                    penalty_str = f" (was {original_prob:.4f})" if idx in recent_usage else ""
                    print(f"        {i+1}. {self.registry.metadata[idx]['name']}: {prob:.4f}{penalty_str}")
                
                print(f"      Top-3 Secondary (after exploration bonus):")
                top_secondary = torch.topk(secondary_probs_adjusted[0], min(3, vocab_size))
                for i in range(len(top_secondary.indices)):
                    idx = top_secondary.indices[i].item()
                    prob = top_secondary.values[i].item()
                    original_prob = secondary_probs[0, idx].item()
                    penalty_str = f" (was {original_prob:.4f})" if idx in recent_usage else ""
                    print(f"        {i+1}. {self.registry.metadata[idx]['name']}: {prob:.4f}{penalty_str}")
                
                print(f"      Composition probabilities:")
                comp_types = ['none', 'sequential', 'nested', 'parallel']
                for comp_type, prob in zip(comp_types, comp_probs[0]):
                    print(f"        {comp_type}: {prob.item():.4f}")
                
                # Generate candidates with layer-based prioritization
                top_k = min(vocab_size, 5)
                
                primary_top = torch.topk(primary_probs_adjusted[0], top_k)
                secondary_top = torch.topk(secondary_probs_adjusted[0], top_k)
                comp_top = torch.topk(comp_probs[0], min(4, comp_probs.shape[-1]))
                
                # Generate 3-function candidates with parallel composition
                for p_idx, p_prob in zip(primary_top.indices, primary_top.values):
                    for s_idx, s_prob in zip(secondary_top.indices, secondary_top.values):
                        # Try tertiary combiners for parallel composition
                        for t_idx in range(vocab_size):
                            t_prob = primary_probs_adjusted[0, t_idx]
                            
                            # Layer-based bonus: prefer higher-layer functions for primary/secondary
                            # but also allow primitives (like + in exp(sin(x)+cos(x)))
                            p_layer = self.registry.metadata[p_idx.item()]['layer']
                            s_layer = self.registry.metadata[s_idx.item()]['layer']
                            t_layer = self.registry.metadata[t_idx]['layer']
                            
                            # STRONGER diversity bonus for mixing abstraction levels
                            diversity_bonus = 1.0
                            if len(set([p_layer, s_layer, t_layer])) > 1:
                                diversity_bonus = 1.5  # Was 1.2, increase to 1.5
                            
                            score = (p_prob * s_prob * t_prob * outputs['confidence'][0] * diversity_bonus).item()
                            
                            candidates.append({
                                'primary_id': p_idx.item(),
                                'secondary_id': s_idx.item(),
                                'tertiary_id': t_idx,
                                'comp_type': 'parallel',
                                'score': score,
                                'step': step
                            })
                        
                        # Also try 2-function compositions
                        for c_idx, c_prob in zip(comp_top.indices, comp_top.values):
                            comp_types = ['none', 'sequential', 'nested', 'parallel']
                            comp_type = comp_types[c_idx.item()]
                            
                            if comp_type != 'parallel':
                                # Same diversity bonus
                                p_layer = self.registry.metadata[p_idx.item()]['layer']
                                s_layer = self.registry.metadata[s_idx.item()]['layer']
                                
                                diversity_bonus = 1.0
                                if comp_type != 'none' and p_layer != s_layer:
                                    diversity_bonus = 1.5  # Was 1.2
                                
                                score = (p_prob * s_prob * c_prob * outputs['confidence'][0] * diversity_bonus).item()
                                
                                candidates.append({
                                    'primary_id': p_idx.item(),
                                    'secondary_id': s_idx.item(),
                                    'tertiary_id': None,
                                    'comp_type': comp_type,
                                    'score': score,
                                    'step': step
                                })
                
                if outputs['confidence'][0] > 0.8 or step >= 5:  # Increase max steps from 3 to 5
                    print(f"\n    [Halting] Confidence: {outputs['confidence'][0].item():.4f} (threshold: 0.8, step: {step+1}/5)")
                    break
        
        # Sort and validate
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n    [Validation] Testing top {min(len(candidates), 100)} candidates...")  # Increase from 50 to 100
        print(f"      Top-5 candidates by score:")
        for i, candidate in enumerate(candidates[:5]):
            p_name = self.registry.metadata[candidate['primary_id']]['name']
            s_name = self.registry.metadata[candidate['secondary_id']]['name']
            t_name = self.registry.metadata[candidate['tertiary_id']]['name'] if candidate.get('tertiary_id') else 'N/A'
            print(f"        {i+1}. {p_name} + {s_name} + {t_name} ({candidate['comp_type']}): {candidate['score']:.6f}")
        
        for i, candidate in enumerate(candidates[:100]):  # Test top 100 instead of 50
            p_name = self.registry.metadata[candidate['primary_id']]['name']
            s_name = self.registry.metadata[candidate['secondary_id']]['name']
            t_name = self.registry.metadata[candidate['tertiary_id']]['name'] if candidate.get('tertiary_id') else 'N/A'
            
            if i < 10:
                print(f"\n      Testing Rank {i+1}: {p_name} + {s_name} + {t_name} ({candidate['comp_type']})")
                print(f"        Executing on examples:")
                all_match = True
                for inp, expected in examples:
                    try:
                        result = self.execute_program(
                            candidate['primary_id'],
                            candidate['secondary_id'],
                            candidate['comp_type'],
                            inp,
                            candidate.get('tertiary_id')
                        )
                        match = result == expected
                        all_match = all_match and match
                        status = "✓" if match else "✗"
                        print(f"          {status} f{tuple(inp)} = {result} (expected {expected})")
                    except Exception as e:
                        print(f"          ✗ f{tuple(inp)} = ERROR: {str(e)[:50]}")
                        all_match = False
                
                if all_match:
                    print(f"\n  [Success] Found via TRM at step {candidate['step']} (rank {i+1}/100)")
                    print(f"    Primary: {p_name}")
                    print(f"    Secondary: {s_name}")
                    if candidate.get('tertiary_id') is not None:
                        print(f"    Tertiary: {t_name}")
                    print(f"    Composition: {candidate['comp_type']}")
                    print(f"    Score: {candidate['score']:.6f}")
                    return candidate
            else:
                if self.validate_program(
                    candidate['primary_id'],
                    candidate['secondary_id'],
                    candidate['comp_type'],
                    examples,
                    candidate.get('tertiary_id')
                ):
                    print(f"\n  [Success] Found via TRM at step {candidate['step']} (rank {i+1}/100)")
                    print(f"    Primary: {p_name}")
                    print(f"    Secondary: {s_name}")
                    if candidate.get('tertiary_id') is not None:
                        print(f"    Tertiary: {t_name}")
                    print(f"    Composition: {candidate['comp_type']}")
                    print(f"    Score: {candidate['score']:.6f}")
                    return candidate
        
        print(f"\n  [Failed] No valid program found in top 100 candidates")
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
        
        # Train (skip for exhaustive finds, focus on refining TRM predictions)
        if program['step'] > 0:
            print(f"\n  [Training] Refining TRM search...")
            for epoch in range(num_epochs):
                loss = self.train_step(examples, program)
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss:.4f}")
        else:
            print(f"\n  [Training] Found via exhaustive search, training TRM to predict it...")
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
                list(args),
                program.get('tertiary_id')
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

    def train_on_curriculum(self, num_epochs: int = 20):
        """Pre-train on curriculum tasks to teach compositional patterns."""
        print("\n" + "="*60)
        print("Training on Curriculum Tasks")
        print("="*60)
        
        tasks = self.generate_curriculum_tasks()
        
        for task_name, examples, target_program in tasks:
            print(f"\n  [Curriculum] Learning: {task_name}")
            
            # Print what we're teaching
            p_name = self.registry.metadata[target_program['primary_id']]['name']
            if target_program['comp_type'] == 'none':
                print(f"    Target: {p_name}")
            elif target_program['comp_type'] == 'parallel' and target_program.get('tertiary_id') is not None:
                s_name = self.registry.metadata[target_program['secondary_id']]['name']
                t_name = self.registry.metadata[target_program['tertiary_id']]['name']
                print(f"    Target: {p_name} + {s_name} ({target_program['comp_type']} with {t_name} combiner)")
            else:
                s_name = self.registry.metadata[target_program['secondary_id']]['name']
                print(f"    Target: {p_name} + {s_name} ({target_program['comp_type']})")
            
            # Train on this task
            for epoch in range(num_epochs):
                loss = self.train_step(examples, target_program)
                if epoch % 10 == 0:
                    print(f"      Epoch {epoch}: Loss = {loss:.4f}")
            
            # Store in history for exploration bonus
            self.training_history.append({
                'name': task_name,
                'id': None,  # Curriculum tasks don't create new functions
                'program': target_program,
                'examples': examples
            })
        
        print(f"\n  [Curriculum] Pre-training complete!")

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
    
    # ==========================================
    # Step 0: Pre-train on curriculum
    # ==========================================
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Layer 1: Learn NAND
    # ==========================================
    nand_examples = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    success = agent.learn_abstraction("NAND", nand_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing NAND")
        print("="*60)
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        for inputs, expected in nand_examples:
            result = registry.execute_function(nand_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} NAND{tuple(inputs)} = {result}")
    
    # ==========================================
    # Layer 2: Learn XOR (using NAND)
    # ==========================================
    xor_examples = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing XOR")
        print("="*60)
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        for inputs, expected in xor_examples:
            result = registry.execute_function(xor_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} XOR{tuple(inputs)} = {result}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    for fid, meta in registry.metadata.items():
        print(f"  [{fid}] {meta['name']} (arity={meta['arity']}, layer={meta['layer']})")
    
    print("\n" + "="*60)
    print("Abstraction Hierarchy Learned!")
    print("="*60)

if __name__ == "__main__":
    main()