from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

class NeurosymbolicTrainer:
    """
    Trains TRM with ITERATIVE REFINEMENT
    Uses actual output error to guide program improvements
    """
    def __init__(self, primitives_manager, learning_rate=1e-3, device='cpu', verbose=True):
        from neurosymbolic_trm import NeurosymbolicTRM
        from symbolic_executor import SymbolicExecutor
        
        self.device = device
        self.prims = primitives_manager
        self.executor = SymbolicExecutor(primitives_manager, verbose=verbose)
        self.model = NeurosymbolicTRM(primitives_manager).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        self.verbose = verbose
        
        # Get pad token ID
        if "PAD" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["PAD"]
        elif "<pad>" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["<pad>"]
        else:
            self.pad_id = self.prims.token_to_id.get("DONE", 0)

    def train_phase(self, task_name: str, train_data: np.ndarray, 
                   n_inputs: int, n_outputs: int, num_epochs: int = 100,
                   detailed_example_idx=0, use_demo=False):
        """
        Train with ITERATIVE REFINEMENT using error gradient
        CORRECT PROCESS:
        - For each epoch:
          - For each example:
            - Do 16 refinement iterations
            - Carry latent state to next example
        """
        
        print(f"\n{'='*60}")
        print(f"Phase: {task_name.upper()}")
        print(f"  Inputs: {n_inputs}, Outputs: {n_outputs}")
        print(f"  Training examples: {len(train_data)}")
        print(f"{'='*60}\n")
        
        # Prepare training data
        inputs = train_data[:, :n_inputs].astype(np.float32)
        targets = train_data[:, n_inputs:].astype(np.float32)
        
        best_program = None
        best_error = float('inf')
        best_success_rate = 0.0
        
        n_refinement_iters = 16  # Number of refinement iterations per example
        
        for epoch in range(num_epochs):
            # Reset for each epoch
            z_scratchpad = None  # Latent state - carries over between examples
            epoch_best_program = None
            epoch_best_error = float('inf')
            
            # Iterate through each example
            for example_idx in range(len(inputs)):
                # Get single example
                example_input = inputs[example_idx:example_idx+1]  # Keep batch dimension
                example_target = targets[example_idx:example_idx+1]
                
                # Combine into problem data [batch=1, problem_len]
                problem_data = torch.tensor(
                    np.concatenate([example_input, example_target], axis=1),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Run multiple refinement iterations on THIS example
                current_program_ids = None
                
                for refine_iter in range(n_refinement_iters):
                    self.model.train()
                    
                    # Forward pass
                    program_logits, z_scratchpad, satisfaction, q_halt = self.model(
                        problem_data,
                        current_program_ids,
                        z_scratchpad
                    )
                    
                    # Sample a few programs and pick best
                    best_iter_program = None
                    best_iter_error = float('inf')
                    
                    for k in range(5):  # Sample 5 candidates
                        temp = 0.5 + k * 0.3
                        program = self._sample_program(
                            program_logits[0], 
                            temperature=temp, 
                            n_inputs=n_inputs
                        )
                        
                        if len(program) == 0:
                            continue
                        
                        # Execute and compute error on THIS example
                        input_tuple = tuple(example_input[0].astype(int))
                        target_tuple = tuple(example_target[0].astype(int))
                        
                        result = self.executor.execute_program(program, input_tuple, trace=False)
                        
                        if result is not None:
                            # Compute squared error
                            error = sum((pred - tgt) ** 2 for pred, tgt in zip(result, target_tuple))
                            
                            if error < best_iter_error:
                                best_iter_error = error
                                best_iter_program = program
                    
                    # Update if we found a valid program
                    if best_iter_program is not None:
                        # Update current program for next iteration
                        current_program_ids = torch.tensor(
                            [[self.prims.token_to_id.get(t, self.pad_id) for t in best_iter_program]],
                            device=self.device
                        )
                        
                        # Learn from this good program
                        self.optimizer.zero_grad()
                        
                        # Recompute forward pass for gradients
                        program_logits, z_scratchpad, satisfaction, q_halt = self.model(
                            problem_data,
                            current_program_ids,
                            z_scratchpad.detach() if z_scratchpad is not None else None
                        )
                        
                        # Compute loss
                        program_ids = [self.prims.token_to_id[t] for t in best_iter_program 
                                      if t in self.prims.token_to_id]
                        
                        if len(program_ids) > 0:
                            total_loss = 0
                            weight = 1.0 / (best_iter_error + 1.0)
                            
                            for pos in range(min(len(program_ids), program_logits.size(1))):
                                target_id = program_ids[pos]
                                logits = program_logits[0, pos]
                                
                                nll = F.cross_entropy(
                                    logits.unsqueeze(0),
                                    torch.tensor([target_id], device=self.device)
                                )
                                total_loss = total_loss + nll * weight
                            
                            # Add satisfaction loss - FIX THE SHAPE!
                            normalized_error = torch.clamp(
                                torch.tensor(best_iter_error, device=self.device, dtype=torch.float32) / 100.0,
                                0, 1
                            )
                            target_satisfaction = 1.0 - normalized_error
                            
                            # satisfaction[0] is shape [1], target should also be [1]
                            satisfaction_loss = F.binary_cross_entropy(
                                satisfaction[0], 
                                target_satisfaction.unsqueeze(0)  # Make it [1]
                            )
                            total_loss = total_loss + satisfaction_loss
                            
                            # Backprop
                            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss.item() > 0:
                                total_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                self.optimizer.step()
                        
                        # Track best for this example
                        if best_iter_error < epoch_best_error:
                            epoch_best_error = best_iter_error
                            epoch_best_program = best_iter_program
                
                # After all refinement iterations on this example,
                # z_scratchpad carries over to next example
            
            # Evaluate best program from this epoch on ALL examples
            if epoch_best_program is not None:
                total_error = 0
                n_correct = 0
                
                for i in range(len(inputs)):
                    input_tuple = tuple(inputs[i].astype(int))
                    target_tuple = tuple(targets[i].astype(int))
                    
                    result = self.executor.execute_program(epoch_best_program, input_tuple, trace=False)
                    
                    if result is not None:
                        error = sum((pred - tgt) ** 2 for pred, tgt in zip(result, target_tuple))
                        total_error += error
                        
                        if result == target_tuple:
                            n_correct += 1
                
                avg_error = total_error / len(inputs)
                success_rate = n_correct / len(inputs)
                
                # Update global best
                if avg_error < best_error:
                    best_error = avg_error
                    best_program = epoch_best_program
                    best_success_rate = success_rate
            
            # Print progress
            if epoch % 10 == 0 or best_success_rate >= 0.95:
                print(f"Epoch {epoch}: Error={best_error:.4f}, Success={best_success_rate*100:.0f}%")
                if best_program:
                    print(f"  Program: {' '.join(best_program[:10])}")
            
            # Early stopping
            if best_success_rate >= 1.0:
                print(f"\n✅ Task solved at epoch {epoch}!\n")
                return True, ' '.join(best_program)
        
        print()
        # Final result
        if best_program and best_success_rate >= 0.95:
            return True, ' '.join(best_program)
        else:
            return False, ' '.join(best_program) if best_program else "No solution found"
    
    def _sample_program(self, logits, temperature=1.0, max_len=10, n_inputs=1):
        """
        Sample a program from the model's distribution
        Enforce validity constraints
        """
        program = []
        stack_depth = 0
        
        for pos in range(min(max_len, logits.size(0))):
            # Get logits for this position
            pos_logits = logits[pos] / temperature
            
            # Mask invalid tokens based on stack depth
            mask = torch.zeros_like(pos_logits)
            mask[:] = -float('inf')
            
            # Always allow LOADs and CONSTs
            for i in range(n_inputs):
                load_id = self.prims.token_to_id.get(f'LOAD_{i}')
                if load_id is not None:
                    mask[load_id] = 0
            
            for i in range(10):
                const_id = self.prims.token_to_id.get(f'CONST_{i}')
                if const_id is not None:
                    mask[const_id] = 0
            
            # Allow ops based on stack depth
            if stack_depth >= 1:
                for op in ['NEG', 'SQUARE']:
                    op_id = self.prims.token_to_id.get(op)
                    if op_id is not None:
                        mask[op_id] = 0
            
            if stack_depth >= 2:
                for op in ['ADD', 'SUB', 'MUL', 'DIV']:
                    op_id = self.prims.token_to_id.get(op)
                    if op_id is not None:
                        mask[op_id] = 0
            
            # Sample token
            masked_logits = pos_logits + mask
            probs = F.softmax(masked_logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            token = self.prims.id_to_token[token_id]
            
            # Skip control tokens
            if token in ['PAD', 'START', 'DONE']:
                continue
            
            program.append(token)
            
            # Update stack depth
            if token.startswith('LOAD_') or token.startswith('CONST_'):
                stack_depth += 1
            elif token in ['NEG', 'SQUARE']:
                pass  # Stack depth unchanged (1 in, 1 out)
            elif token in ['ADD', 'SUB', 'MUL', 'DIV']:
                stack_depth -= 1  # 2 in, 1 out
            
            # Stop if we have a complete valid program (stack_depth == 1)
            # After at least 1 token
            if stack_depth == 1 and len(program) >= 1:
                # For simple programs (just LOAD or small programs), stop more aggressively
                if len(program) == 1:
                    # Just a LOAD - always stop
                    break
                elif len(program) <= 3:
                    # Small program - 80% chance to stop
                    if np.random.random() < 0.8:
                        break
                else:
                    # Longer program - 50% chance to stop
                    if np.random.random() < 0.5:
                        break
            
            # Safety: if stack depth is wrong, stop
            if stack_depth < 0 or stack_depth > 5:
                break
        
        return program
    
    def _eval_program(self, program, inputs, targets):
        """Helper to count correct predictions"""
        n_correct = 0
        for i in range(len(inputs)):
            input_tuple = tuple(inputs[i].astype(int))
            target_tuple = tuple(targets[i].astype(int))
            
            result = self.executor.execute_program(program, input_tuple, trace=False)
            
            if result is not None and result == target_tuple:
                n_correct += 1
        return n_correct
    
    def _extract_equation(self, programs: List[List[str]], task_name: str,
                         n_inputs: int, n_outputs: int) -> str:
        """Extract the most common valid program"""
        cleaned = []
        for p in programs:
            clean_p = [t for t in p if t not in ['PAD', 'DONE', 'START']]
            if clean_p:
                cleaned.append(tuple(clean_p))
        
        if not cleaned:
            return f"No valid program found"
        
        most_common_tuple, count = Counter(cleaned).most_common(1)[0]
        program_tokens = list(most_common_tuple)
        
        return f"{' '.join(program_tokens[:15])}"