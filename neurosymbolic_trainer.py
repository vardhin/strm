from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

class NeurosymbolicTrainer:
    """
    Trains TRM with FULL VISIBILITY into reasoning process
    Uses random exploration + reinforcement learning
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
        
        # Demo programs ONLY for verification, not for cheating
        self.demo_programs = {
            "identity": ["LOAD_0"],
            "increment": ["LOAD_0", "CONST_1", "ADD"],
            "addition": ["LOAD_0", "LOAD_1", "ADD"],
        }
        
        # Get pad token ID
        if "PAD" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["PAD"]
        elif "<pad>" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["<pad>"]
        else:
            self.pad_id = self.prims.token_to_id.get("DONE", 0)

    def train_phase(self, task_name: str, train_data: np.ndarray, 
                   n_inputs: int, n_outputs: int, num_epochs: int = 50,
                   detailed_example_idx=0, use_demo=False):
        """
        Train with pure exploration + reinforcement learning
        NO CHEATING WITH TEMPLATES!
        """
        
        print(f"\n{'='*60}")
        print(f"Phase: {task_name.upper()}")
        print(f"  Inputs: {n_inputs}, Outputs: {n_outputs}")
        print(f"  Training examples: {len(train_data)}")
        print(f"{'='*60}\n")
        
        # Prepare training data
        inputs = train_data[:, :n_inputs].astype(np.float32)
        targets = train_data[:, n_inputs:].astype(np.float32)
        
        # Convert to torch tensors
        problem_data = torch.tensor(
            np.concatenate([inputs, targets], axis=1), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Track best program found through exploration
        best_program = None
        best_success_rate = 0.0
        programs_tried = set()
        
        for epoch in range(num_epochs):
            # PHASE 1: Exploration - sample programs from model
            self.model.eval()
            with torch.no_grad():
                program_logits, z_scratchpad, satisfaction, q_halt = self.model(problem_data)
                
                # Sample multiple programs using different temperatures
                sampled_programs = []
                for temp in [0.5, 1.0, 2.0]:
                    for _ in range(10):  # Sample 10 programs per temperature
                        program_tokens = self._sample_program(
                            program_logits[0], 
                            temperature=temp,
                            n_inputs=n_inputs
                        )
                        
                        # Deduplicate
                        prog_key = tuple(program_tokens)
                        if prog_key not in programs_tried:
                            programs_tried.add(prog_key)
                            sampled_programs.append(program_tokens)
            
            # PHASE 2: Evaluate all sampled programs
            program_rewards = []
            for program in sampled_programs:
                n_correct = self._eval_program(program, inputs, targets)
                success_rate = n_correct / len(inputs)
                program_rewards.append((program, success_rate))
                
                # Track best
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_program = program
            
            # PHASE 3: Reinforcement learning - learn from good programs
            if program_rewards:
                # Sort by reward
                program_rewards.sort(key=lambda x: x[1], reverse=True)
                
                # Take top programs that have non-zero reward
                top_programs = [p for p, r in program_rewards[:5] if r > 0]
                
                if top_programs:
                    self.model.train()
                    total_loss = 0
                    
                    for program in top_programs:
                        # Get reward
                        reward = self._eval_program(program, inputs, targets) / len(inputs)
                        
                        if reward == 0:
                            continue
                        
                        # Convert program to IDs
                        program_ids = [self.prims.token_to_id[t] for t in program 
                                      if t in self.prims.token_to_id]
                        
                        if len(program_ids) == 0:
                            continue
                        
                        self.optimizer.zero_grad()
                        
                        # Forward pass
                        program_logits, z_scratchpad, satisfaction, q_halt = self.model(problem_data)
                        
                        # Policy gradient loss (REINFORCE)
                        loss = 0
                        for pos, target_id in enumerate(program_ids[:min(len(program_ids), program_logits.size(1))]):
                            logits = program_logits[0, pos]
                            log_prob = F.log_softmax(logits, dim=-1)[target_id]
                            
                            # Weighted by reward
                            loss = loss - log_prob * reward
                        
                        # Add entropy bonus for exploration
                        probs = F.softmax(program_logits[0], dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                        loss = loss - 0.01 * entropy  # Encourage exploration
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            total_loss += loss.item()
                    
                    avg_loss = total_loss / max(len(top_programs), 1)
                else:
                    avg_loss = 0.0
            else:
                avg_loss = 0.0
            
            # Print progress
            if epoch % 5 == 0 or best_success_rate >= 0.95:
                print(f"Epoch {epoch}: Best={best_success_rate*100:.0f}%, Loss={avg_loss:.4f}")
                if best_program:
                    print(f"  Program: {' '.join(best_program[:10])}")
            
            # Early stopping
            if best_success_rate >= 1.0:
                print(f"\n✅ Task solved at epoch {epoch}!\n")
                return True, ' '.join(best_program)
        
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
                pass  # Stack depth unchanged
            elif token in ['ADD', 'SUB', 'MUL', 'DIV']:
                stack_depth -= 1
            
            # Stop if valid program (stack_depth == 1)
            if stack_depth == 1 and len(program) >= 2:
                if np.random.random() < 0.3:  # 30% chance to stop
                    break
        
        return program
    
    def _eval_program(self, program, inputs, targets):
        """Helper to evaluate a program on all examples"""
        # CRITICAL: Reject programs that don't use any inputs!
        uses_input = any(token.startswith('LOAD_') for token in program)
        if not uses_input and len(inputs[0]) > 0:
            return 0  # Programs MUST use inputs!
        
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