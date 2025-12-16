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
    
    def train_step(self, examples: List[Tuple[List[int], Any]], 
                   target_program: Dict, format_fn) -> float:
        """Train with deep supervision across recursive steps."""
        self.model.train()
        x_input = format_fn(examples)
        
        from trm import SymbolicTRMCarry
        carry = SymbolicTRMCarry(
            z_H=torch.zeros(1, self.config['seq_len'], self.config['hidden_size']),
            z_L=torch.zeros(1, self.config['seq_len'], self.config['hidden_size']),
            steps=torch.zeros(1, dtype=torch.int32),
            halted=torch.ones(1, dtype=torch.bool)
        )
        carry = self.model.reset_carry(carry.halted, carry)
        
        total_loss = 0.0
        
        for step in range(min(3, self.config['halt_max_steps'])):
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
    
    def train_on_curriculum(self, curriculum_gen: CurriculumGenerator, 
                           format_fn, training_history: List, num_epochs: int = 20):
        """Pre-train on curriculum tasks to teach compositional patterns."""
        print("\n" + "="*60)
        print("Training on Curriculum Tasks")
        print("="*60)
        
        tasks = curriculum_gen.generate_tasks()
        
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
                loss = self.train_step(examples, target_program, format_fn)
                if epoch % 10 == 0:
                    print(f"      Epoch {epoch}: Loss = {loss:.4f}")
            
            # Store in history for exploration bonus
            training_history.append({
                'name': task_name,
                'id': None,
                'program': target_program,
                'examples': examples
            })
        
        print(f"\n  [Curriculum] Pre-training complete!")