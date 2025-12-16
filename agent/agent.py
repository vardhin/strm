import torch
from typing import List, Tuple, Any, Optional
from symbolic import SymbolicRegistry, ProgramExecutor, CurriculumGenerator
from trm import SymbolicTRMCore, SymbolicTRMCarry
from layers import CastedLinear
from .search import ProgramSearcher
from .training import TRMTrainer

class SymbolicAgent:
    """Main orchestrator for symbolic reasoning with TRM."""
    
    def __init__(self, registry: SymbolicRegistry, d_model=128, max_recursion=10):
        self.registry = registry
        self.executor = ProgramExecutor(registry)
        self.curriculum_gen = CurriculumGenerator(registry)
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
            'H_cycles': 2,
            'L_cycles': 2,
            'L_layers': 2,
            'halt_max_steps': max_recursion,
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
        """
        Convert I/O examples to structured format with explicit I/O separation.
        Shape: [batch, num_examples, 3] where each row is [input1, input2, output]
        """
        data = []
        for inputs, output in examples:
            # Pad unary inputs to binary (2 inputs)
            if len(inputs) == 1:
                padded_inputs = inputs + [0]
            else:
                padded_inputs = inputs[:2]
            
            data.append(padded_inputs + [output])
        
        # Pad to fixed size (4 examples)
        while len(data) < self.config['seq_len']:
            data.append([0, 0, 0])
        
        # Flatten for model input
        flat_data = [val for row in data for val in row]
        return torch.tensor([flat_data], dtype=torch.float32)

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
        """
        Use TRM's recursive reasoning with learned heuristics and exploration.
        """
        print(f"\n--- Searching for: {task_name} ---")
        
        # Bootstrap with exhaustive search for primitives only
        if self.registry.get_vocab_size() <= 3:
            print(f"  [Bootstrap] Using exhaustive search (vocab size = {self.registry.get_vocab_size()})")
            program = self.searcher.exhaustive_search(examples)
            if program is not None:
                self._print_found_program(program, exhaustive=True)
                return program
        
        # TRM-based learned search
        x_input = self.format_examples(examples)
        carry = self.initial_carry(batch_size=1)
        carry = self.model.reset_carry(carry.halted, carry)
        
        return self.searcher.trm_search(examples, task_name, x_input, carry, exploration_bonus)
    
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
        train_message = "Refining TRM search..." if program['step'] > 0 else "Found via exhaustive search, training TRM to predict it..."
        print(f"\n  [Training] {train_message}")
        
        for epoch in range(num_epochs):
            loss = self.trainer.train_step(examples, program, self.format_examples)
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Loss = {loss:.4f}")
        
        # Save
        def learned_function(*args):
            return self.executor.execute_program(
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
        """Pre-train on curriculum tasks."""
        self.trainer.train_on_curriculum(
            self.curriculum_gen, 
            self.format_examples, 
            self.training_history, 
            num_epochs
        )