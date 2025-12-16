import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Dict
from symbolic import SymbolicRegistry, CurriculumGenerator
from trm import SymbolicTRMCarry

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
        
        self.model.train()
        x_input = format_fn(examples)
        batch_size = x_input.shape[0]
        
        # Initialize carry
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
    
    def train_on_examples(self, x_input: torch.Tensor, comp_data: Dict, 
                         examples: List[Tuple[List[int], int]], num_epochs: int = 30):
        """Train model on examples."""
        
        batch_size = x_input.shape[0]
        seq_len = self.config['seq_len']
        d_model = self.config['hidden_size']
        
        # Initialize carry - use correct field names
        carry = SymbolicTRMCarry(
            z_H=torch.zeros(batch_size, seq_len, d_model),
            z_L=torch.zeros(batch_size, seq_len, d_model),
            steps=torch.zeros(batch_size, dtype=torch.int32),
            halted=torch.ones(batch_size, dtype=torch.bool)
        )
        carry = self.model.reset_carry(carry.halted, carry)
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            carry, output = self.model(carry, x_input)
            
            # Compute loss
            loss = self._compute_loss(output, comp_data, examples)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"      Epoch {epoch}: Loss = {loss.item():.4f}")
    
    def _compute_loss(self, output, comp_data: Dict, examples: List) -> torch.Tensor:
        """Compute training loss."""
        
        # Target function IDs
        primary_target = comp_data['primary_id']
        secondary_target = comp_data.get('secondary_id')
        
        # Composition type targets
        comp_type_map = {'none': 0, 'sequential': 1, 'nested': 2, 'parallel': 3}
        comp_type_target = comp_type_map[comp_data['comp_type']]
        
        # Pool logits if they have sequence dimension
        primary_logits = output['primary_logits']
        if primary_logits.dim() == 3:
            primary_logits = primary_logits.mean(dim=1)
            secondary_logits = output['secondary_logits'].mean(dim=1)
            composition_logits = output['composition_logits'].mean(dim=1)
        else:
            secondary_logits = output['secondary_logits']
            composition_logits = output['composition_logits']
        
        batch_size = primary_logits.shape[0]
        
        # Compute losses
        loss_primary = torch.nn.functional.cross_entropy(
            primary_logits,
            torch.tensor([primary_target] * batch_size, dtype=torch.long)
        )
        
        # Only compute secondary loss if secondary function is used
        if secondary_target is not None:
            loss_secondary = torch.nn.functional.cross_entropy(
                secondary_logits,
                torch.tensor([secondary_target] * batch_size, dtype=torch.long)
            )
        else:
            # For 'none' composition, predict 0 (or first function) as dummy
            loss_secondary = torch.nn.functional.cross_entropy(
                secondary_logits,
                torch.tensor([0] * batch_size, dtype=torch.long)
            ) * 0.1  # Reduced weight since it's not meaningful
        
        loss_comp_type = torch.nn.functional.cross_entropy(
            composition_logits,
            torch.tensor([comp_type_target] * batch_size, dtype=torch.long)
        )
        
        # Total loss
        total_loss = loss_primary + loss_secondary + loss_comp_type
        
        return total_loss