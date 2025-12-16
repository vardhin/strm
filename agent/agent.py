import torch
import os
from typing import List, Tuple, Any, Optional, Dict
from symbolic import SymbolicRegistry, ProgramExecutor, CurriculumGenerator
from trm import SymbolicTRMCore, SymbolicTRMCarry
from layers import CastedLinear
from .search import ProgramSearcher
from .training import TRMTrainer

class SymbolicAgent:
    """Main orchestrator for symbolic reasoning with TRM."""
    
    def __init__(self, registry: SymbolicRegistry, d_model: int = 128, 
                 max_recursion: int = 8, input_dim: int = 32, max_composition_depth: int = 3):
        """Initialize agent with TRM model."""
        self.registry = registry
        self.executor = ProgramExecutor(registry)
        self.curriculum_gen = CurriculumGenerator(registry, input_dim=input_dim)
        self.d_model = d_model
        self.max_recursion = max_recursion
        self.input_dim = input_dim  # Store this!
        
        # Configuration
        self.config = {
            'input_dim': input_dim,
            'seq_len': 4,
            'hidden_size': d_model,
            'expansion': 2.0,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'rms_norm_eps': 1e-5,
            'rope_theta': 10000.0,
            'forward_dtype': 'float32',
            'max_functions': registry.get_vocab_size(),
            'H_cycles': 2,
            'L_cycles': 2,
            'L_layers': 2,
            'halt_max_steps': max_recursion,
            'max_composition_depth': max_composition_depth,
        }
        
        self.model = SymbolicTRMCore(self.config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.training_history = []
        
        # Initialize components
        self.searcher = ProgramSearcher(
            registry, self.executor, self.model, 
            self.config, self.training_history
        )
        self.trainer = TRMTrainer(
            self.model, self.optimizer, self.config, registry
        )

    def train_on_curriculum(self, num_epochs: int = 20):
        """Pre-train on curriculum tasks."""
        print("\n" + "="*60)
        print("Training on Curriculum Tasks")
        print("="*60)
        
        tasks = self.curriculum_gen.generate_curriculum()
        
        for task in tasks:
            print(f"\n  [Curriculum] Learning: {task['name']}")
            print(f"    Target: {self.registry.metadata[task['target']['primary_id']]['name']}")
            
            # Train on this task
            self._train_on_composition(
                task['target'], 
                task['examples'], 
                num_epochs=num_epochs
            )
        
        print("\n  [Curriculum] Pre-training complete!\n")

    def learn_abstraction(self, name: str, examples: List[Tuple[List[int], int]], 
                         num_epochs: int = 30, exploration_bonus: float = 0.5,
                         max_terms: int = 2) -> bool:
        """Learn a new abstraction from examples using pure TRM learning.
        
        Args:
            name: Name of the abstraction
            examples: Training examples
            num_epochs: Number of training epochs
            exploration_bonus: Exploration bonus for sampling
            max_terms: Maximum composition depth (2=binary, 3=ternary, etc.)
        """
        print("\n" + "="*60)
        print(f"Learning Abstraction: {name}")
        print(f"  Max composition depth: {max_terms}")
        print("="*60)
        
        # Search using TRM with specified depth
        comp_data = self.search_with_trm(examples, exploration_bonus=exploration_bonus, max_terms=max_terms)
        
        if comp_data is None:
            print(f"\n  [Failed] Could not find program for {name}")
            print(f"  Suggestion: Try more epochs, increase exploration_bonus, or increase max_terms")
            return False
        
        # Found a solution! Train TRM to predict it
        print(f"\n  [Training] Training TRM to predict this composition...")
        self._train_on_composition(comp_data, examples, num_epochs=num_epochs)
        print(f"  [Training] Complete!")
        
        # Register the new function
        fid = self._register_abstraction(name, comp_data, examples)
        
        # Resize model to accommodate new function
        self.resize_model_for_new_abstraction()
        
        return True
    
    def _train_on_composition(self, comp_data: Dict, examples: List[Tuple[List[int], int]], 
                             num_epochs: int = 30):
        """Train TRM to predict a composition."""
        x_input = self.format_examples(examples)
        
        # Use trainer - it will handle carry initialization
        self.trainer.train_on_examples(
            x_input, comp_data, examples, num_epochs
        )

    def search_with_trm(self, examples: List[Tuple[List[int], int]], 
                       max_steps: int = 10,
                       confidence_threshold: float = 0.8,
                       exploration_bonus: float = 0.5,
                       max_terms: int = 2) -> Optional[Dict]:
        """Search for composition using TRM predictions with feedback."""
        
        print(f"\n  [TRM Search] Pure learning with feedback")
        print(f"    Vocab size: {len(self.registry.functions)}")
        print(f"    Max steps: {max_steps}")
        print(f"    Max composition depth: {max_terms}")
        
        x_input = self.format_examples(examples)
        batch_size = x_input.shape[0]
        seq_len = self.config['seq_len']
        d_model = self.config['hidden_size']
        
        carry = SymbolicTRMCarry(
            z_H=torch.zeros(batch_size, seq_len, d_model),
            z_L=torch.zeros(batch_size, seq_len, d_model),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool)
        )
        carry = self.model.reset_carry(carry.halted, carry)
        
        tried_candidates = set()
        
        for step in range(max_steps):
            print(f"\n  [TRM Search] Step {step+1}/{max_steps}...")
            
            carry, output = self.model(carry, x_input)
            
            # Extract predictions
            primary_logits = output['primary_logits']
            if primary_logits.dim() == 3:
                primary_logits = primary_logits.mean(dim=1).mean(dim=0)
                secondary_logits = output['secondary_logits'].mean(dim=1).mean(dim=0)
                tertiary_logits = output['tertiary_logits'].mean(dim=1).mean(dim=0)
                composition_logits = output['composition_logits'].mean(dim=1).mean(dim=0)
            else:
                primary_logits = primary_logits.mean(dim=0)
                secondary_logits = output['secondary_logits'].mean(dim=0)
                tertiary_logits = output['tertiary_logits'].mean(dim=0)
                composition_logits = output['composition_logits'].mean(dim=0)
            
            top_k = min(3 + step, len(primary_logits))
            comp_types = ['none', 'sequential', 'nested', 'parallel']
            
            _, primary_top = torch.topk(primary_logits, min(top_k, len(primary_logits)))
            _, secondary_top = torch.topk(secondary_logits, min(top_k, len(secondary_logits)))
            _, tertiary_top = torch.topk(tertiary_logits, min(top_k, len(tertiary_logits)))
            _, comp_type_top = torch.topk(composition_logits, min(len(comp_types), len(composition_logits)))
            
            print(f"    Top {top_k} predictions:")
            print(f"      Primary: {[self.registry.metadata[idx.item()]['name'] for idx in primary_top[:5]]}")
            print(f"      Secondary: {[self.registry.metadata[idx.item()]['name'] for idx in secondary_top[:5]]}")
            print(f"      Tertiary: {[self.registry.metadata[idx.item()]['name'] for idx in tertiary_top[:5]]}")
            print(f"      Comp types: {[comp_types[idx.item()] for idx in comp_type_top]}")
            
            # Generate candidates with tertiary support
            candidates = self._generate_deep_candidates(
                primary_top, secondary_top, tertiary_top, comp_type_top, comp_types, max_terms
            )
            
            # Filter duplicates
            new_candidates = []
            for cand in candidates:
                key = (cand['primary_id'], cand['secondary_id'], cand.get('tertiary_id'), 
                       cand['comp_type'], cand.get('loop_count', 1))
                if key not in tried_candidates:
                    tried_candidates.add(key)
                    new_candidates.append(cand)
            
            print(f"    Testing {len(new_candidates)} new candidates (total tried: {len(tried_candidates)})")
            
            for rank, candidate in enumerate(new_candidates, 1):
                if self._test_composition(candidate, examples):
                    self._print_solution(candidate)
                    return candidate
            
            if len(new_candidates) == 0:
                print(f"    All candidates exhausted at this depth")
            
            # Add noise for exploration
            with torch.no_grad():
                carry.z_H += torch.randn_like(carry.z_H) * 0.01
                carry.z_L += torch.randn_like(carry.z_L) * 0.01
        
        print(f"\n  ✗ No solution found after {max_steps} steps")
        print(f"  Tried {len(tried_candidates)} unique candidates total")
        return None
    
    def _generate_deep_candidates(self, primary_top, secondary_top, tertiary_top, 
                                 comp_type_top, comp_types, max_terms: int) -> List[Dict]:
        """Generate candidates with tertiary combiner support."""
        candidates = []
        
        # Depth 1: Single functions
        if max_terms >= 1:
            for prim_idx in primary_top:
                candidates.append({
                    'primary_id': prim_idx.item(),
                    'secondary_id': None,
                    'tertiary_id': None,
                    'comp_type': 'none',
                    'loop_count': 1
                })
        
        # Depth 2: Binary compositions
        if max_terms >= 2:
            for comp_idx in comp_type_top:
                comp_type = comp_types[comp_idx.item()]
                if comp_type == 'none':
                    continue
                
                prim_list = primary_top
                sec_list = secondary_top
                
                # Expand search if space is small
                if len(primary_top) * len(secondary_top) < 20:
                    all_fns = list(self.registry.functions.keys())
                    prim_list = torch.tensor([f for f in all_fns if f != self.registry.loop_id])
                    sec_list = torch.tensor(all_fns)
                
                for prim_idx in prim_list:
                    for sec_idx in sec_list:
                        prim_id = prim_idx if isinstance(prim_idx, int) else prim_idx.item()
                        sec_id = sec_idx if isinstance(sec_idx, int) else sec_idx.item()
                        
                        loop_count = 1
                        if sec_id == self.registry.loop_id:
                            prim_meta = self.registry.metadata.get(prim_id)
                            if prim_meta and prim_meta.get('arity') == 1:
                                loop_count = -1
                        
                        # For parallel composition, try with combiners
                        if comp_type == 'parallel':
                            for tert_idx in tertiary_top:
                                tert_id = tert_idx if isinstance(tert_idx, int) else tert_idx.item()
                                candidates.append({
                                    'primary_id': prim_id,
                                    'secondary_id': sec_id,
                                    'tertiary_id': tert_id,
                                    'comp_type': comp_type,
                                    'loop_count': loop_count
                                })
                        else:
                            candidates.append({
                                'primary_id': prim_id,
                                'secondary_id': sec_id,
                                'tertiary_id': None,
                                'comp_type': comp_type,
                                'loop_count': loop_count
                            })
        
        # Depth 3+: Use learned functions
        if max_terms >= 3:
            learned_fns = [fid for fid, meta in self.registry.metadata.items() 
                          if meta.get('layer', 0) > 0]
            
            if learned_fns:
                print(f"      Trying {len(learned_fns)} learned functions...")
                
                for comp_idx in comp_type_top:
                    comp_type = comp_types[comp_idx.item()]
                    if comp_type == 'none':
                        continue
                    
                    # Parallel with learned functions - KEY FOR XOR!
                    if comp_type == 'parallel':
                        # Try: combiner(LEARNED(x,y), PRIMITIVE(x,y))
                        for learned_id in learned_fns:
                            for prim_idx in primary_top:
                                for tert_idx in tertiary_top:
                                    candidates.append({
                                        'primary_id': learned_id,
                                        'secondary_id': prim_idx.item(),
                                        'tertiary_id': tert_idx.item(),
                                        'comp_type': 'parallel',
                                        'loop_count': 1
                                    })
                    
                    # Sequential/nested with learned functions
                    else:
                        for prim_idx in primary_top:
                            for learned_id in learned_fns:
                                candidates.append({
                                    'primary_id': prim_idx.item(),
                                    'secondary_id': learned_id,
                                    'tertiary_id': None,
                                    'comp_type': comp_type,
                                    'loop_count': 1
                                })
        
        return candidates
    
    def _print_solution(self, candidate: Dict):
        """Pretty print the found solution."""
        comp_type = candidate['comp_type']
        p_name = self.registry.metadata[candidate['primary_id']]['name']
        
        if comp_type == 'none':
            print(f"\n  ✓ Found: {p_name}")
        else:
            s_name = self.registry.metadata[candidate['secondary_id']]['name']
            loop_info = f" (dynamic loop)" if candidate.get('loop_count') == -1 else ""
            
            if comp_type == 'sequential':
                print(f"\n  ✓ Found: {s_name}({p_name}(x)){loop_info}")
            elif comp_type == 'nested':
                print(f"\n  ✓ Found: {p_name}({s_name}(x), {s_name}(y)){loop_info}")
            elif comp_type == 'parallel':
                print(f"\n  ✓ Found: parallel({p_name}, {s_name}){loop_info}")
    
    def _test_composition(self, comp_data: Dict, examples: List[Tuple[List[int], int]]) -> bool:
        """Test if composition matches all examples."""
        try:
            for inputs, expected in examples:
                result = self.executor.execute_program(
                    comp_data['primary_id'],
                    comp_data['secondary_id'],
                    comp_data['comp_type'],
                    inputs,
                    tertiary_id=comp_data.get('tertiary_id'),
                    loop_count=comp_data.get('loop_count', 1)
                )
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def _train_on_composition(self, comp_data: Dict, examples: List[Tuple[List[int], int]], 
                             num_epochs: int = 30):
        """Train TRM to predict a composition."""
        x_input = self.format_examples(examples)
        
        # Use trainer - it will handle carry initialization
        self.trainer.train_on_examples(
            x_input, comp_data, examples, num_epochs
        )

    def _register_abstraction(self, name: str, comp_data: Dict, examples: List) -> int:
        """Register new abstraction in registry."""
        def learned_fn(inputs):
            return self.executor.execute_program(
                comp_data['primary_id'],
                comp_data['secondary_id'],
                comp_data['comp_type'],
                inputs,
                tertiary_id=comp_data.get('tertiary_id'),
                loop_count=comp_data.get('loop_count', 1)
            )
        
        input_arity = len(examples[0][0])
        fid = self.registry.register(name, learned_fn, arity=input_arity)
        
        print(f"\n  [Memory] Saved '{name}' as Function ID {fid}")
        return fid

    def format_examples(self, examples: List[Tuple[List[int], Any]]) -> torch.Tensor:
        """Format examples into tensor for TRM input."""
        input_dim = self.config['input_dim']
        seq_len = self.config['seq_len']
        batch_size = len(examples)
        
        batch_data = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32)
        
        for batch_idx, (inputs, _) in enumerate(examples):
            num_inputs = min(len(inputs), seq_len)
            
            for pos in range(num_inputs):
                val = inputs[pos]
                
                if val >= 0:
                    bits = [(val >> i) & 1 for i in range(input_dim)]
                else:
                    unsigned = (1 << input_dim) + val
                    bits = [(unsigned >> i) & 1 for i in range(input_dim)]
                
                batch_data[batch_idx, pos] = torch.tensor(bits, dtype=torch.float32)
        
        return batch_data

    def resize_model_for_new_abstraction(self):
        """Dynamically resize output heads when new function is learned."""
        new_vocab_size = self.registry.get_vocab_size()
        self.config['max_functions'] = new_vocab_size
        
        # Resize all function prediction heads
        heads_to_resize = ['head_primary_op', 'head_secondary_op', 'head_tertiary_op']
        
        for head_name in heads_to_resize:
            if not hasattr(self.model, head_name):
                continue
                
            old_head = getattr(self.model, head_name)
            old_vocab_size = old_head.weight.shape[0]
            
            if old_vocab_size >= new_vocab_size:
                continue  # Already big enough
            
            new_head = CastedLinear(self.d_model, new_vocab_size, bias=False)
            
            with torch.no_grad():
                # Copy old weights
                new_head.weight[:old_vocab_size] = old_head.weight
                # Initialize new weights with small noise around mean
                avg_weight = old_head.weight.mean(dim=0)
                new_head.weight[old_vocab_size:] = avg_weight.unsqueeze(0) + torch.randn(
                    new_vocab_size - old_vocab_size, self.d_model
                ) * 0.01
            
            setattr(self.model, head_name, new_head)
        
        print(f"  [Resize] Model vocabulary expanded to {new_vocab_size} functions")

    def save_checkpoint(self, path: str = "checkpoints/model.pt", registry_path: str = "checkpoints/registry.pkl"):
        """Save model and registry."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        self.registry.save(registry_path)
        print(f"  [Agent] Model saved to {path}")
        print(f"  [Registry] Saved to {registry_path}")

    def load(self, path: str = "checkpoints/model.pt", registry_path: str = "checkpoints/registry.pkl"):
        """Load model and registry."""
        self.registry.load(registry_path)
        
        checkpoint = torch.load(path, weights_only=False)
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', [])
        
        # Reinitialize model with loaded config
        self.model = SymbolicTRMCore(self.config)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"  [Agent] Model loaded from {path}")
        print(f"  [Agent] Vocabulary size: {self.registry.get_vocab_size()}")
        print(f"  [Agent] Training history: {len(self.training_history)} episodes")