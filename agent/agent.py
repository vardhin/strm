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
        self.curriculum_gen = CurriculumGenerator(registry, input_dim=input_dim)  # Pass input_dim!
        self.d_model = d_model
        self.max_recursion = max_recursion
        
        # Configuration - allow custom input_dim
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
            'max_composition_depth': max_composition_depth,  # NEW: control composition depth
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

    def initial_carry(self, batch_size: int) -> SymbolicTRMCarry:
        """Initialize carry state."""
        return SymbolicTRMCarry(
            z_H=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            z_L=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool)
        )

    def format_examples(self, examples: List[Tuple[List[int], Any]]) -> torch.Tensor:
        """Format examples into tensor for TRM input.
        
        Each input value occupies one position in the sequence.
        We encode each integer using input_dim features.
        Pad to seq_len positions with zeros.
        
        Returns: [batch_size, seq_len, input_dim]
        """
        import torch
        
        input_dim = self.config['input_dim']  # bits per integer
        seq_len = self.config['seq_len']      # max sequence positions
        batch_size = len(examples)
        
        # Create tensor: [batch_size, seq_len, input_dim]
        batch_data = torch.zeros(batch_size, seq_len, input_dim, dtype=torch.float32)
        
        for batch_idx, (inputs, _) in enumerate(examples):
            num_inputs = min(len(inputs), seq_len)
            
            for pos in range(num_inputs):
                val = inputs[pos]
                
                # Convert integer to binary representation
                if val >= 0:
                    bits = [(val >> i) & 1 for i in range(input_dim)]
                else:
                    # Two's complement for negative
                    unsigned = (1 << input_dim) + val
                    bits = [(unsigned >> i) & 1 for i in range(input_dim)]
                
                batch_data[batch_idx, pos] = torch.tensor(bits, dtype=torch.float32)
        
        return batch_data

    def resize_model_for_new_abstraction(self):
        """Dynamically resize output heads when new function is learned."""
        new_vocab_size = self.registry.get_vocab_size()
        self.config['max_functions'] = new_vocab_size
        
        for head_name in ['head_primary_op', 'head_secondary_op']:
            old_head = getattr(self.model, head_name)
            old_vocab_size = old_head.weight.shape[0]
            
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

    def search_with_recursive_trm(self, examples: List[Tuple[List[int], Any]], 
                                  task_name: str = "Unknown",
                                  exploration_bonus: float = 0.5) -> Optional[dict]:
        """Search for program using TRM heuristics with fallback to exhaustive."""
        x_input = self.format_examples(examples)
        carry = self.initial_carry(batch_size=1)
        
        # Try TRM search first (only if we have enough functions)
        if self.registry.get_vocab_size() > 3:
            result = self.searcher.trm_search(examples, task_name, x_input, carry, exploration_bonus)
            if result:
                self._print_found_program(result, exhaustive=False)
                return result
        
        # Fallback to exhaustive
        print(f"  [Bootstrap] Using exhaustive search (vocab size = {self.registry.get_vocab_size()})")
        result = self.searcher.exhaustive_search(examples)
        if result:
            print(f"  [Success] Found via exhaustive search")
            self._print_found_program(result, exhaustive=True)
        return result
    
    def _print_found_program(self, program: dict, exhaustive: bool = False):
        """Print discovered program."""
        method = "exhaustive search" if exhaustive else "TRM"
        print(f"  [Success] Found via {method}")
        print(f"    Primary: {self.registry.metadata[program['primary_id']]['name']}")
        if program['comp_type'] != 'none':
            print(f"    Secondary: {self.registry.metadata[program['secondary_id']]['name']}")
        if program['tertiary_id'] is not None:
            print(f"    Tertiary: {self.registry.metadata[program['tertiary_id']]['name']}")
        print(f"    Composition: {program['comp_type']}")

    def learn_abstraction(self, name: str, examples: List[Tuple[List[int], Any]], 
                         num_epochs: int = 30, exploration_bonus: float = 0.5) -> bool:
        """Learn a new abstraction from examples."""
        print(f"\n{'='*60}")
        print(f"Learning Abstraction: {name}")
        print(f"{'='*60}\n")
        
        # Search for program
        program = self.search_with_recursive_trm(examples, name, exploration_bonus)
        
        if not program:
            print(f"\n  [Failed] Could not find program for {name}")
            return False
        
        # Detect arity from examples
        input_arity = len(examples[0][0])
        
        # Train TRM to predict this program
        print(f"\n  [Training] Found via {'TRM' if 'TRM' in str(program.get('source', 'exhaustive')) else 'exhaustive search'}, training TRM to predict it...")
        self.train_on_program(program, examples, num_epochs)
        
        # Save as new abstraction with correct arity
        new_id = self.save_abstraction(name, program, input_arity)
        print(f"\n  [Memory] Saved '{name}' as Function ID {new_id}")
        print(f"  [Resize] Model vocabulary expanded to {self.registry.get_vocab_size()} functions")
        
        return True
    
    def save_abstraction(self, name: str, program: dict, arity: int) -> int:
        """Save a learned program as a new named function."""
        # Extract program structure
        primary_id = program['primary_id']
        secondary_id = program['secondary_id']
        tertiary_id = program.get('tertiary_id')
        comp_type = program['comp_type']
        loop_count = program.get('loop_count', 1)
        
        # Validate loop_count is properly captured
        if secondary_id == self.registry.loop_id and loop_count == 1:
            print(f"  [Warning] LOOP detected but loop_count=1, may be unintended")
        
        # Debug: print what we're saving
        print(f"  [Debug] Saving abstraction with loop_count={loop_count}")
        
        # Create executable function with correct loop_count captured in closure
        def abstraction_fn(inputs):  # Changed: don't use *inputs, just inputs
            return self.executor.execute_program(
                primary_id, 
                secondary_id, 
                comp_type, 
                inputs,  # Changed: removed list() wrapper since inputs is already a list
                tertiary_id,
                loop_count  # This captures the value in closure
            )
        
        # Register as new function with explicit arity
        # Note: layer will be auto-computed by registry._compute_layer()
        layer = max(m['layer'] for m in self.registry.metadata.values()) + 1
        new_id = self.registry.register(name, abstraction_fn, arity=arity, layer=layer)
        
        # IMPORTANT: Save composition metadata for reconstruction
        self.registry.compositions[new_id] = {
            'composition': comp_type,
            'primary_id': primary_id,
            'secondary_id': secondary_id,
            'tertiary_id': tertiary_id,
            'loop_count': loop_count
        }
        
        # Update model to handle new vocabulary size
        self.resize_model(self.registry.get_vocab_size())
        
        return new_id
    
    def register_abstraction(self, name: str, primary_id: int, secondary_id: Optional[int], 
                            tertiary_id: Optional[int], composition: str, 
                            loop_count: int = 1) -> int:
        """Register a new learned abstraction in the registry."""
        print(f"  [Debug] Saving abstraction with loop_count={loop_count}")
        
        # Build the composed function
        if composition == 'sequential':
            if secondary_id == self.registry.loop_id:
                # LOOP composition
                composed_fn = self.registry._create_loop_function(primary_id, loop_count)
                arity = self.registry.metadata[primary_id]['arity']
            else:
                # Sequential composition: f2(f1(x))
                composed_fn = lambda inputs, p=primary_id, s=secondary_id: \
                    self.registry.execute_function(s, [self.registry.execute_function(p, inputs)])
                arity = self.registry.metadata[primary_id]['arity']
        
        elif composition == 'nested':
            # Nested composition: f1(f2(x[0]), f2(x[1]), ...)
            composed_fn = lambda inputs, p=primary_id, s=secondary_id: \
                self.registry.execute_function(p, [self.registry.execute_function(s, [x]) for x in inputs])
            arity = self.registry.metadata[secondary_id]['arity']
        
        elif composition == 'parallel':
            # Parallel composition: tertiary(primary(inputs), secondary(inputs))
            composed_fn = lambda inputs, p=primary_id, s=secondary_id, t=tertiary_id: \
                self.registry.execute_function(t, [
                    self.registry.execute_function(p, inputs),
                    self.registry.execute_function(s, inputs)
                ])
            arity = self.registry.metadata[primary_id]['arity']
        
        else:
            # Single function, no composition
            composed_fn = lambda inputs, p=primary_id: \
                self.registry.execute_function(p, inputs)
            arity = self.registry.metadata[primary_id]['arity']
        
        # Register in registry
        layer = max(m['layer'] for m in self.registry.metadata.values()) + 1
        fid = self.registry.register(name, composed_fn, arity, layer)
        
        # IMPORTANT: Save composition metadata for reconstruction
        self.registry.compositions[fid] = {
            'composition': composition,
            'primary_id': primary_id,
            'secondary_id': secondary_id,
            'tertiary_id': tertiary_id,
            'loop_count': loop_count
        }
        
        print(f"  [Memory] Saved '{name}' as Function ID {fid}")
        
        # Expand TRM vocabulary
        self._expand_vocabulary(1)
        
        return fid
    
    def train_on_curriculum(self, num_epochs: int = 20):
        """Train model on curriculum tasks."""
        self.trainer.train_on_curriculum(
            self.curriculum_gen,
            self.format_examples,  # ← This is the format_fn
            self.training_history,
            num_epochs=num_epochs
        )
    
    def train_on_program(self, program: dict, examples: List[Tuple[List[int], Any]], 
                        num_epochs: int = 30):
        """Train TRM to predict a discovered program."""
        print(f"  [Training] Teaching TRM to predict this composition...")
        
        for epoch in range(num_epochs):
            loss = self.trainer.train_step(examples, program, self.format_examples)
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss = {loss:.4f}")
        
        # Add to training history
        self.training_history.append({
            'name': program.get('name', 'Unknown'),
            'id': None,  # Will be filled when saved
            'program': program,
            'examples': examples
        })
        
        print(f"  [Training] Complete!")
    
    def resize_model(self, new_vocab_size: int):
        """Resize model vocabulary (alias for resize_model_for_new_abstraction)."""
        self.resize_model_for_new_abstraction()
    
    def save(self, model_path: str, registry_path: str):
        """Save model weights and registry."""
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        print(f"  [Agent] Model saved to {model_path}")
        
        # Save registry
        self.registry.save(registry_path)
    
    def load(self, model_path: str, registry_path: str):
        """Load model weights and registry."""
        # Load registry first
        self.registry.load(registry_path)
        
        # Update config with new vocab size
        vocab_size = self.registry.get_vocab_size()
        self.config['max_functions'] = vocab_size
        
        # Reinitialize model with new vocab size
        self.model = SymbolicTRMCore(self.config)
        
        # Reinitialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        
        # Reinitialize searcher and trainer with updated model
        self.searcher = ProgramSearcher(
            self.registry, self.executor, self.model, 
            self.config, self.training_history
        )
        self.trainer = TRMTrainer(
            self.model, self.optimizer, self.config, self.registry
        )
        
        print(f"  [Agent] Model loaded from {model_path}")
        print(f"  [Agent] Vocabulary size: {vocab_size}")
        print(f"  [Agent] Training history: {len(self.training_history)} episodes")
    
    def save_checkpoint(self, save_dir: str = "checkpoints"):
        """Save checkpoint with timestamp."""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "model.pt")
        registry_path = os.path.join(save_dir, "registry.pkl")
        self.save(model_path, registry_path)