from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

class NeurosymbolicTrainer:
    """
    Trains TRM with FULL VISIBILITY into reasoning process
    Uses beam search + demonstrations for faster convergence
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
        
        # Pre-defined programs for curriculum learning
        self.demo_programs = {
            "identity": ["LOAD_0"],
            "increment": ["LOAD_0", "CONST_1", "ADD"],
            "decrement": ["LOAD_0", "CONST_1", "SUB"],
            "addition": ["LOAD_0", "LOAD_1", "ADD"],
            "subtraction": ["LOAD_0", "LOAD_1", "SUB"],
            "double": ["LOAD_0", "CONST_2", "MUL"],
        }
        
        # Get pad token ID
        if "PAD" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["PAD"]
        elif "<pad>" in self.prims.token_to_id:
            self.pad_id = self.prims.token_to_id["<pad>"]
        else:
            self.pad_id = self.prims.token_to_id.get("DONE", 0)

    def train_phase(self, task_name: str, train_data: np.ndarray, 
                   n_inputs: int, n_outputs: int, num_epochs: int = 30,
                   detailed_example_idx=0, use_demo=True):
        """
        Train with BEAM SEARCH + supervised learning from demonstrations
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
        
        # Get demonstration program if available
        demo_program = None
        if use_demo and task_name in self.demo_programs:
            demo_program = self.demo_programs[task_name]
            print(f"💡 Using demonstration: {' '.join(demo_program)}\n")
        
        all_programs = []
        best_program = None
        best_success_rate = 0.0
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Phase 1: Supervised learning from demonstration (if available)
            if demo_program and epoch < 10:
                # Learn to imitate the demonstration
                demo_ids = [self.prims.token_to_id[t] for t in demo_program]
                
                # Forward pass
                program_logits, z_scratchpad, satisfaction, q_halt = self.model(problem_data)
                
                # Supervised loss: cross-entropy to match demo
                loss = 0
                for pos, target_id in enumerate(demo_ids):
                    if pos >= program_logits.size(1):
                        break
                    logits = program_logits[0, pos]
                    loss = loss + F.cross_entropy(
                        logits.unsqueeze(0), 
                        torch.tensor([target_id], device=self.device)
                    )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Evaluate demo program
                n_correct = 0
                for i in range(len(inputs)):
                    input_tuple = tuple(inputs[i].astype(int))
                    target_tuple = tuple(targets[i].astype(int))
                    result = self.executor.execute_program(demo_program, input_tuple, trace=False)
                    if result is not None and result == target_tuple:
                        n_correct += 1
                
                success_rate = n_correct / len(inputs)
                best_success_rate = max(best_success_rate, success_rate)
                best_program = demo_program
                
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Success={success_rate*100:.1f}% [Learning from demo]")
                
            else:
                # Phase 2: Beam search for program discovery
                beam_size = 5
                candidates = self._beam_search(problem_data, inputs, targets, beam_size=beam_size)
                
                # If beam search finds nothing good, try random search
                if not candidates or candidates[0][1] < 0.5:  # Changed from 0.3
                    if epoch % 3 == 0:  # Changed from every 5 to every 3 epochs
                        print(f"Epoch {epoch}: Beam search stuck, trying random exploration...")
                        random_candidates = self._random_search(inputs, targets, n_samples=100)  # Increased from 50
                        if random_candidates and random_candidates[0][1] > candidates[0][1] if candidates else 0:
                            print(f"   ✨ Random search found better program!")
                            candidates = random_candidates
                
                if not candidates:
                    print(f"Epoch {epoch}: No valid programs found")
                    continue
                
                # Take best candidate
                best_candidate = candidates[0]
                program_tokens, success_rate, avg_reward = best_candidate
                
                # Update best program
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_program = program_tokens.copy()
                
                # Learn from the best program found
                if success_rate > 0.5:  # Only learn from decent programs
                    program_ids = [self.prims.token_to_id[t] for t in program_tokens 
                                  if t in self.prims.token_to_id]
                    
                    # Forward pass
                    program_logits, z_scratchpad, satisfaction, q_halt = self.model(problem_data)
                    
                    # Supervised loss to match the good program
                    loss = 0
                    for pos, target_id in enumerate(program_ids):
                        if pos >= program_logits.size(1):
                            break
                        logits = program_logits[0, pos]
                        loss = loss + F.cross_entropy(
                            logits.unsqueeze(0),
                            torch.tensor([target_id], device=self.device)
                        )
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    program_str = ' '.join(program_tokens[:5])
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Success={success_rate*100:.1f}%, Program={program_str}")
                else:
                    program_str = ' '.join(program_tokens[:5])
                    print(f"Epoch {epoch}: Success={success_rate*100:.1f}%, Program={program_str}")
            
            # Store program
            if best_program:
                all_programs.append(best_program[:10])
            
            # Check if solved
            if best_success_rate >= 0.95:
                print(f"\n✅ Task solved at epoch {epoch}!\n")
                break
        
        # Extract result
        if best_program and best_success_rate >= 0.95:
            equation = ' '.join(best_program[:15])
        else:
            equation = self._extract_equation(all_programs, task_name, n_inputs, n_outputs)
        
        success = best_success_rate >= 0.95
        
        return success, equation
    
    def _beam_search(self, problem_data, inputs, targets, beam_size=5, max_len=10):
        """
        Beam search with DIVERSITY BONUS to find good programs
        Returns list of (program_tokens, success_rate, avg_reward) tuples
        """
        # Get logits from model
        with torch.no_grad():
            program_logits, _, _, _ = self.model(problem_data)
        
        # Initialize beam with empty programs
        beam = [([],  0.0)]  # (program, log_prob)
        
        # Track token usage to encourage diversity
        token_usage = Counter()
        
        for pos in range(min(max_len, program_logits.size(1))):
            candidates = []
            
            for program, log_prob in beam:
                logits = program_logits[0, pos]
                probs = F.softmax(logits, dim=0)
                
                # Add diversity bonus: penalize frequently used tokens
                diversity_bonus = torch.zeros_like(probs)
                for token_name, count in token_usage.items():
                    if token_name in self.prims.token_to_id:
                        token_id = self.prims.token_to_id[token_name]
                        # Reduce probability of overused tokens
                        diversity_bonus[token_id] = -0.1 * count
                
                # Adjust probabilities with diversity bonus
                adjusted_logits = logits + diversity_bonus
                adjusted_probs = F.softmax(adjusted_logits, dim=0)
                
                # Get top-k tokens with temperature sampling for exploration
                temperature = 1.5  # Higher = more exploration
                temp_probs = F.softmax(adjusted_logits / temperature, dim=0)
                
                # Sample more tokens for diversity
                top_k = min(beam_size * 2, temp_probs.size(0))
                top_probs, top_indices = torch.topk(temp_probs, top_k)
                
                for prob, idx in zip(top_probs, top_indices):
                    token_id = idx.item()
                    token_name = self.prims.id_to_token[token_id]
                    
                    # Skip control tokens early in program
                    if token_name in ['START', 'DONE', 'PAD'] and pos < 1:
                        continue
                    
                    # Encourage using operations (not just LOAD)
                    operation_bonus = 0.0
                    if token_name in ['ADD', 'SUB', 'MUL', 'DIV', 'NEG', 'SQUARE']:
                        operation_bonus = 0.3  # Boost operations
                    elif token_name.startswith('CONST_'):
                        operation_bonus = 0.2  # Also boost constants
                    
                    new_program = program + [token_name]
                    new_log_prob = log_prob + torch.log(prob + 1e-10).item() + operation_bonus
                    
                    candidates.append((new_program, new_log_prob))
                    token_usage[token_name] += 1
                    
                    # Stop this branch if we hit a terminator
                    if token_name in ['DONE', 'PAD']:
                        break
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]
        
        # Evaluate all beam programs
        results = []
        for program, log_prob in beam:
            n_correct = 0
            rewards = []
            
            for i in range(len(inputs)):
                input_tuple = tuple(inputs[i].astype(int))
                target_tuple = tuple(targets[i].astype(int))
                
                result = self.executor.execute_program(program, input_tuple, trace=False)
                
                if result is not None and result == target_tuple:
                    n_correct += 1
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            
            success_rate = n_correct / len(inputs)
            avg_reward = np.mean(rewards)
            
            results.append((program, success_rate, avg_reward))
        
        # Sort by success rate
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _random_search(self, inputs, targets, n_samples=50, max_len=8):
        """
        Smarter random search that respects stack constraints
        """
        results = []
        
        for _ in range(n_samples):
            # Build program incrementally, tracking stack
            program = []
            stack_depth = 0
            
            for pos in range(max_len):
                # Choose token based on stack depth
                if stack_depth == 0:
                    # Need to push something onto stack
                    candidates = [t for t in self.prims.get_primitive_names()
                                if t.startswith('LOAD_') or t.startswith('CONST_')]
                    token = np.random.choice(candidates)
                    program.append(token)
                    stack_depth += 1
                    
                elif stack_depth == 1:
                    # Can push another value or apply unary op
                    if np.random.random() < 0.6:  # 60% chance to push
                        candidates = [t for t in self.prims.get_primitive_names()
                                    if t.startswith('LOAD_') or t.startswith('CONST_')]
                        token = np.random.choice(candidates)
                        program.append(token)
                        stack_depth += 1
                    else:  # 40% chance unary op
                        candidates = ['NEG', 'SQUARE']
                        if candidates:
                            token = np.random.choice(candidates)
                            program.append(token)
                            # stack_depth stays 1
                
                elif stack_depth >= 2:
                    # Can apply binary op
                    if np.random.random() < 0.7:  # 70% chance binary op
                        candidates = ['ADD', 'SUB', 'MUL', 'DIV']
                        token = np.random.choice(candidates)
                        program.append(token)
                        stack_depth -= 1  # Two inputs, one output
                    else:  # 30% chance push another value
                        if stack_depth < 4:  # Don't let stack get too deep
                            candidates = [t for t in self.prims.get_primitive_names()
                                        if t.startswith('LOAD_') or t.startswith('CONST_')]
                            token = np.random.choice(candidates)
                            program.append(token)
                            stack_depth += 1
                        else:
                            # Force binary op
                            candidates = ['ADD', 'SUB', 'MUL', 'DIV']
                            token = np.random.choice(candidates)
                            program.append(token)
                            stack_depth -= 1
                
                # Stop if we have exactly one value on stack (valid output)
                if stack_depth == 1 and len(program) >= 3 and np.random.random() < 0.3:
                    break
            
            # Only keep programs that end with stack depth 1
            if stack_depth != 1:
                continue
            
            # Evaluate program
            n_correct = 0
            for i in range(len(inputs)):
                input_tuple = tuple(inputs[i].astype(int))
                target_tuple = tuple(targets[i].astype(int))
                
                result = self.executor.execute_program(program, input_tuple, trace=False)
                
                if result is not None and result == target_tuple:
                    n_correct += 1
            
            success_rate = n_correct / len(inputs)
            results.append((program, success_rate, 0.0))
        
        # Remove duplicates
        unique_results = {}
        for prog, succ, _ in results:
            prog_key = tuple(prog)
            if prog_key not in unique_results or unique_results[prog_key][1] < succ:
                unique_results[prog_key] = (prog, succ, 0.0)
        
        results = list(unique_results.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 3 for debugging
        if results and results[0][1] > 0:
            print(f"   🔍 Top random programs:")
            for i, (prog, succ, _) in enumerate(results[:3]):
                prog_str = ' '.join(prog[:8])
                print(f"      {i+1}. [{succ*100:.0f}%] {prog_str}")
        
        return results
    
    def _guided_search(self, inputs, targets, n_samples=200, max_len=6):
        """
        Template-based search for common patterns
        Focus on likely program structures
        """
        results = []
        
        # Template 1: LOAD_0 CONST_X OP (e.g., "a + 1", "a * 2")
        for const in [0, 1, 2, 3, 4, 5]:
            for op in ['ADD', 'SUB', 'MUL', 'DIV']:
                program = ['LOAD_0', f'CONST_{const}', op]
                n_correct = self._eval_program(program, inputs, targets)
                if n_correct > 0:
                    results.append((program, n_correct / len(inputs), 0.0))
        
        # Template 2: LOAD_0 CONST_X OP CONST_Y OP2 (e.g., "2*a + 1")
        for c1 in [0, 1, 2, 3]:
            for c2 in [0, 1, 2, 3]:
                for op1 in ['ADD', 'MUL']:
                    for op2 in ['ADD', 'SUB']:
                        program = ['LOAD_0', f'CONST_{c1}', op1, f'CONST_{c2}', op2]
                        n_correct = self._eval_program(program, inputs, targets)
                        if n_correct > 0:
                            results.append((program, n_correct / len(inputs), 0.0))
        
        # Template 3: LOAD_0 LOAD_0 OP (e.g., "a + a", "a * a")
        for op in ['ADD', 'MUL']:
            program = ['LOAD_0', 'LOAD_0', op]
            n_correct = self._eval_program(program, inputs, targets)
            if n_correct > 0:
                results.append((program, n_correct / len(inputs), 0.0))
        
        # Template 4: LOAD_0 LOAD_0 OP CONST_X OP2 (e.g., "a*a + 1")
        for c in [0, 1, 2]:
            for op1 in ['ADD', 'MUL']:
                for op2 in ['ADD', 'SUB']:
                    program = ['LOAD_0', 'LOAD_0', op1, f'CONST_{c}', op2]
                    n_correct = self._eval_program(program, inputs, targets)
                    if n_correct > 0:
                        results.append((program, n_correct / len(inputs), 0.0))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Print top programs
        if results and results[0][1] >= 0.5:
            print(f"   🎯 Guided search found promising programs:")
            for i, (prog, succ, _) in enumerate(results[:3]):
                if succ >= 0.5:
                    prog_str = ' '.join(prog)
                    print(f"      {i+1}. [{succ*100:.0f}%] {prog_str}")
        
        return results
    
    def _eval_program(self, program, inputs, targets):
        """Helper to evaluate a program on all examples"""
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
        # Clean programs (remove PAD, DONE, etc.)
        cleaned = []
        for p in programs:
            clean_p = [t for t in p if t not in ['PAD', 'DONE', 'START']]
            if clean_p:
                cleaned.append(tuple(clean_p))
        
        if not cleaned:
            return f"No valid program found"
        
        # Find most common program
        most_common_tuple, count = Counter(cleaned).most_common(1)[0]
        program_tokens = list(most_common_tuple)
        
        return f"{' '.join(program_tokens[:15])}"