from typing import List, Tuple, Dict, Optional, Set
import torch

class CompositionSimplifier:
    """Simplifies complex compositions to minimal equivalent forms."""
    
    def __init__(self, registry, model, executor):
        self.registry = registry
        self.model = model
        self.executor = executor
    
    def simplify(self, composition: List[Tuple[int, List[int]]], 
                 examples: List[Tuple[List[int], int]],
                 max_attempts: int = 5) -> List[Tuple[int, List[int]]]:
        """Try to find a simpler composition that produces same outputs.
        
        Strategy:
        1. Check if composition can use higher-level abstractions
        2. Try to reduce number of terms
        3. Look for known patterns (e.g., NOT(XOR) instead of complex form)
        """
        best_composition = composition
        best_complexity = self._compute_complexity(composition)
        
        print(f"\n[Simplify] Original complexity: {best_complexity}")
        print(f"  Terms: {[(self.registry.metadata[cid]['name'], args) for cid, args in composition]}")
        
        # Attempt 1: Check if we can replace with a single existing function
        single_func = self._try_single_function(examples)
        if single_func and self._compute_complexity(single_func) < best_complexity:
            best_composition = single_func
            best_complexity = self._compute_complexity(single_func)
            print(f"  ✓ Reduced to single function (complexity: {best_complexity})")
            print(f"  Terms: {[(self.registry.metadata[cid]['name'], args) for cid, args in best_composition]}")
            return best_composition
        
        # Attempt 2: Check if we can replace subsequences with existing functions
        for attempt in range(max_attempts):
            simplified = self._try_pattern_matching(best_composition, examples)
            if simplified and self._compute_complexity(simplified) < best_complexity:
                if self._validate_composition(simplified, examples):
                    best_composition = simplified
                    best_complexity = self._compute_complexity(simplified)
                    print(f"  ✓ Found simpler form (complexity: {best_complexity})")
                    break  # Found improvement, no need to continue
        
        # Attempt 3: Try to eliminate redundant intermediate steps
        pruned = self._prune_redundant_steps(best_composition, examples)
        if pruned and self._compute_complexity(pruned) < best_complexity:
            if self._validate_composition(pruned, examples):
                best_composition = pruned
                best_complexity = self._compute_complexity(pruned)
                print(f"  ✓ Pruned redundant steps (complexity: {best_complexity})")
        
        # Attempt 4: Use TRM to search for shorter compositions
        shorter = self._search_shorter_composition(best_composition, examples)
        if shorter and self._compute_complexity(shorter) < best_complexity:
            if self._validate_composition(shorter, examples):
                best_composition = shorter
                best_complexity = self._compute_complexity(shorter)
                print(f"  ✓ Found even shorter form (complexity: {best_complexity})")
        
        print(f"\n[Simplify] Final complexity: {best_complexity}")
        print(f"  Terms: {[(self.registry.metadata[cid]['name'], args) for cid, args in best_composition]}")
        
        return best_composition
    
    def _compute_complexity(self, composition: List[Tuple[int, List[int]]]) -> int:
        """Compute complexity score (lower is better).
        
        Factors:
        - Number of terms (primary)
        - Layer depth of functions used (secondary)
        - Total arity (tertiary)
        """
        if not composition:
            return float('inf')
        
        num_terms = len(composition)
        max_layer = max(self.registry.metadata[cid]['layer'] for cid, _ in composition)
        total_arity = sum(len(args) for _, args in composition)
        
        # Weighted score: heavily penalize more terms
        return num_terms * 100 + max_layer * 10 + total_arity
    
    def _try_single_function(self, examples: List[Tuple[List[int], int]]) -> Optional[List[Tuple[int, List[int]]]]:
        """Check if a single existing function solves the problem."""
        input_arity = len(examples[0][0])
        
        # Check all functions in registry
        for func_id in range(self.registry.get_vocab_size()):
            func_meta = self.registry.metadata.get(func_id)
            if not func_meta:
                continue
            
            # Only check functions with matching arity
            if func_meta['arity'] != input_arity:
                continue
            
            # Skip primitives (layer 0) - we want abstractions
            if func_meta['layer'] == 0:
                continue
            
            try:
                all_match = True
                for inputs, expected in examples:
                    result = self.registry.execute_function(func_id, inputs)
                    if result != expected:
                        all_match = False
                        break
                
                if all_match:
                    # Found a single function that works!
                    print(f"  → Found existing function: {func_meta['name']}")
                    return [(func_id, list(range(input_arity)))]
            except:
                continue
        
        return None
    
    def _try_pattern_matching(self, composition: List[Tuple[int, List[int]]], 
                              examples: List[Tuple[List[int], int]]) -> Optional[List[Tuple[int, List[int]]]]:
        """Try to match patterns like NOT(existing_func), or replace subsequences."""
        input_arity = len(examples[0][0])
        
        # Pattern 1: Check if NOT(existing_func) solves it
        not_id = self._find_function_by_name('NOT')
        if not_id is not None:
            for func_id in range(self.registry.get_vocab_size()):
                func_meta = self.registry.metadata.get(func_id)
                if not func_meta or func_meta['arity'] != input_arity:
                    continue
                
                try:
                    all_match = True
                    for inputs, expected in examples:
                        result = self.registry.execute_function(func_id, inputs)
                        result = self.registry.execute_function(not_id, [result])
                        if result != expected:
                            all_match = False
                            break
                    
                    if all_match:
                        # Found NOT(existing_func) pattern!
                        print(f"  → Found pattern: NOT({func_meta['name']})")
                        return [
                            (func_id, list(range(input_arity))),
                            (not_id, [input_arity])
                        ]
                except:
                    continue
        
        # Pattern 2: Check if any subsequence can be replaced with a single function
        # Try to replace pairs of operations with a single equivalent
        for i in range(len(composition) - 1):
            # Get the two operations
            op1_id, op1_args = composition[i]
            op2_id, op2_args = composition[i + 1]
            
            # Check if there's a function that combines these two
            for func_id in range(self.registry.get_vocab_size()):
                func_meta = self.registry.metadata.get(func_id)
                if not func_meta:
                    continue
                
                # Create test composition with replacement
                test_comp = composition[:i] + [(func_id, op1_args)] + composition[i+2:]
                
                if self._validate_composition(test_comp, examples):
                    print(f"  → Replaced {self.registry.metadata[op1_id]['name']}+{self.registry.metadata[op2_id]['name']} with {func_meta['name']}")
                    return test_comp
        
        return None
    
    def _prune_redundant_steps(self, composition: List[Tuple[int, List[int]]], 
                               examples: List[Tuple[List[int], int]]) -> Optional[List[Tuple[int, List[int]]]]:
        """Remove steps that don't contribute to the final result."""
        if len(composition) <= 1:
            return None
        
        # Try removing each step and see if it still works
        for i in range(len(composition) - 1):  # Don't remove the last step
            # Create composition without step i
            test_comp = composition[:i] + composition[i+1:]
            
            # Adjust argument indices for steps after the removed one
            adjusted_comp = []
            for step_idx, (func_id, args) in enumerate(test_comp):
                if step_idx < i:
                    adjusted_comp.append((func_id, args))
                else:
                    # Adjust indices that reference removed step or later
                    new_args = []
                    for arg_idx in args:
                        if arg_idx < len(examples[0][0]) + i:
                            new_args.append(arg_idx)
                        elif arg_idx > len(examples[0][0]) + i:
                            new_args.append(arg_idx - 1)
                        else:
                            # This references the removed step - can't prune
                            break
                    
                    if len(new_args) != len(args):
                        # Can't adjust properly, skip this attempt
                        break
                    
                    adjusted_comp.append((func_id, new_args))
            
            if len(adjusted_comp) == len(test_comp):
                # Successfully adjusted, test it
                if self._validate_composition(adjusted_comp, examples):
                    print(f"  → Pruned redundant step {i}: {self.registry.metadata[composition[i][0]]['name']}")
                    return adjusted_comp
        
        return None
    
    def _search_shorter_composition(self, composition: List[Tuple[int, List[int]]], 
                                   examples: List[Tuple[List[int], int]]) -> Optional[List[Tuple[int, List[int]]]]:
        """Use exhaustive search for shorter compositions with same output."""
        current_length = len(composition)
        
        if current_length <= 2:
            # Already quite simple, don't search
            return None
        
        input_arity = len(examples[0][0])
        
        # Try all combinations of 1 or 2 functions
        for target_length in range(1, current_length):
            candidates = self._generate_all_compositions_of_length(target_length, input_arity)
            
            for candidate in candidates:
                if self._validate_composition(candidate, examples):
                    func_names = [self.registry.metadata[cid]['name'] for cid, _ in candidate]
                    print(f"  → Found shorter composition: {' → '.join(func_names)}")
                    return candidate
        
        return None
    
    def _generate_all_compositions_of_length(self, length: int, input_arity: int, 
                                            max_candidates: int = 1000) -> List[List[Tuple[int, List[int]]]]:
        """Generate all possible compositions of given length."""
        candidates = []
        
        # Get all available functions
        all_func_ids = list(range(self.registry.get_vocab_size()))
        
        def generate_recursive(current_comp: List[Tuple[int, List[int]]], 
                              available_values: int):
            if len(current_comp) == length:
                candidates.append(current_comp.copy())
                return
            
            if len(candidates) >= max_candidates:
                return
            
            # Try each function
            for func_id in all_func_ids:
                func_meta = self.registry.metadata.get(func_id)
                if not func_meta:
                    continue
                
                arity = func_meta['arity']
                if arity < 0:  # Skip special functions like LOOP
                    continue
                
                # Try all valid argument combinations
                if arity <= available_values:
                    # For simplicity, just try sequential arguments
                    for start_idx in range(available_values - arity + 1):
                        args = list(range(start_idx, start_idx + arity))
                        current_comp.append((func_id, args))
                        generate_recursive(current_comp, available_values + 1)
                        current_comp.pop()
        
        generate_recursive([], input_arity)
        return candidates
    
    def _validate_composition(self, composition: List[Tuple[int, List[int]]], 
                             examples: List[Tuple[List[int], int]]) -> bool:
        """Check if composition produces correct outputs for all examples."""
        try:
            for inputs, expected in examples:
                # Execute composition step by step
                available_values = list(inputs)
                
                for child_id, arg_indices in composition:
                    # Check if all indices are valid
                    if any(idx >= len(available_values) for idx in arg_indices):
                        return False
                    
                    child_inputs = [available_values[i] for i in arg_indices]
                    result = self.registry.execute_function(child_id, child_inputs)
                    available_values.append(result)
                
                if available_values[-1] != expected:
                    return False
            
            return True
        except Exception as e:
            return False
    
    def _find_function_by_name(self, name: str) -> Optional[int]:
        """Find function ID by name."""
        for func_id, meta in self.registry.metadata.items():
            if meta['name'] == name:
                return func_id
        return None