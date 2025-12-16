import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict
from symbolic import SymbolicRegistry, CurriculumGenerator

class TRMTrainer:
    """Handles training of the TRM model."""
    
    def __init__(self, model, optimizer, config: Dict, registry: SymbolicRegistry):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.registry = registry
    
    def _format_target(self, target: Dict) -> str:
        """Format target program for display."""
        primary = self.registry.metadata[target['primary_id']]['name']
        
        if target.get('secondary_id') is None:
            return primary
        
        secondary = self.registry.metadata[target['secondary_id']]['name']
        comp_type = target['comp_type']
        
        if comp_type == 'none':
            return primary
        elif comp_type == 'sequential':
            loop_count = target.get('loop_count', 1)
            if secondary == 'LOOP' and loop_count > 1:
                return f"{primary} + LOOP({loop_count}) (sequential)"
            return f"{primary} + {secondary} (sequential)"
        elif comp_type == 'nested':
            return f"{primary} + {secondary} (nested)"
        elif comp_type == 'parallel':
            tertiary_id = target.get('tertiary_id')
            if tertiary_id is not None:
                tertiary = self.registry.metadata[tertiary_id]['name']
                return f"{primary} + {secondary} (parallel with {tertiary} combiner)"
            return f"{primary} + {secondary} (parallel)"
        
        return f"{primary} + {secondary}"
    
    def train_step(self, examples: List[Tuple[List[int], Any]], 
                   target_program: Dict, format_fn) -> float:
        """Train with deep supervision across recursive steps."""
        import torch
        import torch.nn.functional as F
        
        self.model.train()
        x_input = format_fn(examples)
        batch_size = x_input.shape[0]
        
        from trm import SymbolicTRMCarry
        carry = SymbolicTRMCarry(
            z_H=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            z_L=torch.zeros(batch_size, self.config['seq_len'], self.config['hidden_size']),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool)
        )
        carry = self.model.reset_carry(carry.halted, carry)
        
        total_loss = 0.0
        
        for step in range(min(3, self.config['halt_max_steps'])):
            carry, outputs = self.model(carry, x_input)
            
            # Check shapes and pool if needed
            # Expected: [batch_size, seq_len, vocab_size] -> pool to [batch_size, vocab_size]
            # But model might output [batch_size, vocab_size] directly
            primary_logits = outputs['primary_logits']
            if primary_logits.dim() == 3:
                # Has sequence dimension, pool it
                primary_logits = primary_logits.mean(dim=1)
                secondary_logits = outputs['secondary_logits'].mean(dim=1)
                composition_logits = outputs['composition_logits'].mean(dim=1)
            elif primary_logits.dim() == 2:
                # Already pooled
                secondary_logits = outputs['secondary_logits']
                composition_logits = outputs['composition_logits']
            else:
                # Unexpected shape, likely 1D - need to check what model returns
                raise ValueError(
                    f"Unexpected logits shape: {primary_logits.shape}. "
                    f"Expected [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]"
                )
            
            # Handle halt logits
            halt_logits = outputs['halt_logits']
            if halt_logits.dim() > 1:
                halt_logits = halt_logits.squeeze()
            
            # Create target tensors
            primary_target = torch.tensor([target_program['primary_id']] * batch_size, dtype=torch.long)
            
            secondary_id = target_program['secondary_id']
            if secondary_id is None:
                secondary_id = target_program['primary_id']
            secondary_target = torch.tensor([secondary_id] * batch_size, dtype=torch.long)
            
            comp_types = ['none', 'sequential', 'nested', 'parallel']
            comp_target_idx = comp_types.index(target_program['comp_type'])
            comp_target = torch.tensor([comp_target_idx] * batch_size, dtype=torch.long)
            
            # Compute losses
            primary_loss = F.cross_entropy(primary_logits, primary_target)
            secondary_loss = F.cross_entropy(secondary_logits, secondary_target)
            comp_loss = F.cross_entropy(composition_logits, comp_target)
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits, torch.ones_like(halt_logits))
            
            step_loss = primary_loss + secondary_loss + comp_loss + 0.1 * halt_loss
            total_loss += step_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def train_on_curriculum(self, curriculum_gen, format_fn, training_history, num_epochs=20):
        """Train on curriculum tasks."""
        print("\n" + "="*60)
        print("Training on Curriculum Tasks")
        print("="*60 + "\n")
        
        # Generate curriculum tasks
        tasks = curriculum_gen.generate_curriculum()
        
        for task in tasks:
            print(f"  [Curriculum] Learning: {task['name']}")
            print(f"    Target: {self._format_target(task['target'])}")
            
            examples = task['examples']
            target_program = task['target']
            
            # Train on this task
            for epoch in range(num_epochs):
                loss = self.train_step(examples, target_program, format_fn)
                if epoch % 10 == 0:
                    print(f"      Epoch {epoch}: Loss = {loss:.4f}")
            
            # Store in history
            training_history.append({
                'name': task['name'],
                'program': target_program,
                'examples': examples
            })
        
        print("\n  [Curriculum] Pre-training complete!\n")