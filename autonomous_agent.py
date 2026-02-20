import random
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import time

class AutonomousAgent:
    """Self-directed agent that explores function space autonomously"""
    
    def __init__(self, agent, registry, config=None):
        self.agent = agent
        self.registry = registry
        
        # Configuration
        self.config = config or {
            'example_generation_budget': 100,
            'min_examples_per_function': 5,
            'max_examples_per_function': 20,
            'exploration_rate': 0.3,
            'pruning_threshold': 0.1,  # Functions with <10% usage get pruned
            'merge_similarity_threshold': 0.9,
            'curiosity_bonus': 0.5,
            'complexity_penalty': 0.1,
        }
        
        # Tracking metrics
        self.function_usage_counts = defaultdict(int)
        self.function_success_rates = defaultdict(lambda: {'success': 0, 'total': 0})
        self.discovery_history = []
        self.merge_candidates = []
        
    # ==========================================
    # 1. AUTONOMOUS EXAMPLE GENERATION
    # ==========================================
    
    def generate_examples_for_target(self, target_function_name: str, 
                                     arity: int, num_examples: int = 10) -> List[Tuple[List, int]]:
        """Generate examples by composing existing functions"""
        examples = []
        available_functions = [fid for fid in self.registry.metadata.keys()]
        
        for _ in range(num_examples):
            # Generate random inputs
            inputs = [random.randint(0, 20) for _ in range(arity)]
            
            # Try to compute output using function composition
            try:
                # Randomly sample functions to compose
                num_compositions = random.randint(1, 3)
                result = inputs[0] if inputs else 0
                
                for _ in range(num_compositions):
                    func_id = random.choice(available_functions)
                    func_arity = len(self.registry.metadata[func_id]['signature'])
                    
                    if func_arity == 1:
                        result = self.registry.execute_function(func_id, [result])
                    elif func_arity == 2 and len(inputs) > 1:
                        result = self.registry.execute_function(func_id, [result, inputs[1]])
                
                examples.append((inputs[:arity], result))
            except:
                continue
        
        return examples
    
    def generate_synthetic_examples(self, pattern: str, num_examples: int = 10) -> List[Tuple[List, int]]:
        """Generate examples based on mathematical patterns"""
        examples = []
        
        patterns = {
            'arithmetic_sequence': lambda i: ([i], i * 2),
            'geometric_sequence': lambda i: ([i], 2 ** i if i < 10 else 0),
            'fibonacci': lambda i: ([i], self._fib(i) if i < 15 else 0),
            'prime_check': lambda i: ([i], int(self._is_prime(i))),
            'square': lambda i: ([i], i * i),
            'cube': lambda i: ([i], i * i * i),
            'modulo_pattern': lambda i: ([i, 3], i % 3),
        }
        
        if pattern in patterns:
            for i in range(1, num_examples + 1):
                examples.append(patterns[pattern](i))
        
        return examples
    
    def _fib(self, n):
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    
    def _is_prime(self, n):
        if n < 2: return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0: return False
        return True
    
    # ==========================================
    # 2. SELF-PLAY AND EXPLORATION
    # ==========================================
    
    def self_play_episode(self, num_steps: int = 10):
        """Run one episode of self-directed exploration"""
        print(f"\n{'='*60}")
        print(f"Self-Play Episode: {num_steps} steps")
        print(f"{'='*60}")
        
        for step in range(num_steps):
            action = self._choose_exploration_action()
            
            if action == 'discover_new':
                self._discover_new_function()
            elif action == 'compose_existing':
                self._compose_existing_functions()
            elif action == 'refine_existing':
                self._refine_existing_function()
            elif action == 'explore_property':
                self._explore_mathematical_property()
            
            # Periodic maintenance
            if step % 5 == 0:
                self._prune_unused_functions()
                self._identify_merge_candidates()
    
    def _choose_exploration_action(self) -> str:
        """Choose next exploration action based on curiosity and utility"""
        actions = ['discover_new', 'compose_existing', 'refine_existing', 'explore_property']
        
        # Weight actions by expected value
        weights = [
            0.3,  # discover_new - high curiosity
            0.3,  # compose_existing - medium utility
            0.2,  # refine_existing - medium utility
            0.2,  # explore_property - high learning value
        ]
        
        # Adjust based on current state
        num_functions = len(self.registry.metadata)
        if num_functions < 5:
            weights[0] *= 2  # Focus on discovery early
        elif num_functions > 20:
            weights[2] *= 2  # Focus on refinement later
        
        weights = np.array(weights) / np.sum(weights)
        return np.random.choice(actions, p=weights)
    
    def _discover_new_function(self):
        """Attempt to discover a new useful function"""
        print("\n[Discovery] Searching for new function...")
        
        # Propose a new function hypothesis
        patterns = ['arithmetic_sequence', 'square', 'cube', 'modulo_pattern', 'prime_check']
        pattern = random.choice(patterns)
        
        examples = self.generate_synthetic_examples(pattern, num_examples=15)
        function_name = f"AUTO_{pattern.upper()}_{len(self.registry.metadata)}"
        
        print(f"  Hypothesis: {function_name}")
        print(f"  Pattern: {pattern}")
        print(f"  Examples: {len(examples)}")
        
        success = self.agent.learn_abstraction(
            function_name, 
            examples, 
            num_epochs=30,
            exploration_bonus=self.config['curiosity_bonus']
        )
        
        if success:
            print(f"  ✓ Discovered {function_name}!")
            self.discovery_history.append({
                'name': function_name,
                'pattern': pattern,
                'timestamp': time.time()
            })
        else:
            print(f"  ✗ Failed to discover {function_name}")
    
    def _compose_existing_functions(self):
        """Try to create new function by composing existing ones"""
        print("\n[Composition] Combining existing functions...")
        
        available = list(self.registry.metadata.keys())
        if len(available) < 2:
            return
        
        # Pick 2-3 functions to compose
        func_ids = random.sample(available, min(3, len(available)))
        func_names = [self.registry.metadata[fid]['name'] for fid in func_ids]
        
        print(f"  Composing: {' -> '.join(func_names)}")
        
        # Generate examples by chaining
        examples = []
        for _ in range(10):
            x = random.randint(0, 10)
            result = x
            try:
                for fid in func_ids:
                    arity = len(self.registry.metadata[fid]['signature'])
                    if arity == 1:
                        result = self.registry.execute_function(fid, [result])
                    elif arity == 2:
                        y = random.randint(0, 10)
                        result = self.registry.execute_function(fid, [result, y])
                examples.append(([x], result))
            except:
                continue
        
        if examples:
            composition_name = f"COMP_{'_'.join(func_names[:2])}_{len(self.registry.metadata)}"
            success = self.agent.learn_abstraction(composition_name, examples, num_epochs=25)
            if success:
                print(f"  ✓ Created composition: {composition_name}")
    
    def _refine_existing_function(self):
        """Refine an existing function with more examples"""
        print("\n[Refinement] Improving existing function...")
        
        # Pick a function with low success rate
        candidates = [
            (fid, meta) for fid, meta in self.registry.metadata.items()
            if not meta.get('is_primitive', False) and 'signature' in meta
        ]
        
        if not candidates:
            print("  No non-primitive functions to refine yet")
            return
        
        func_id, meta = random.choice(candidates)
        func_name = meta['name']
        
        print(f"  Refining: {func_name}")
        
        # Generate more diverse examples
        new_examples = self.generate_examples_for_target(
            func_name, 
            len(meta['signature']), 
            num_examples=15
        )
        
        if new_examples:
            success = self.agent.learn_abstraction(func_name, new_examples, num_epochs=20)
            print(f"  {'✓' if success else '✗'} Refinement {'succeeded' if success else 'failed'}")
        else:
            print("  ✗ Could not generate examples")
    
    def _explore_mathematical_property(self):
        """Explore mathematical properties like commutativity, associativity"""
        print("\n[Exploration] Testing mathematical properties...")
        
        available = [
            (fid, meta) for fid, meta in self.registry.metadata.items()
            if 'signature' in meta and len(meta['signature']) == 2
        ]
        
        if not available:
            print("  No binary functions available to test")
            return
        
        func_id, meta = random.choice(available)
        func_name = meta['name']
        
        # Test commutativity
        commutative = True
        test_count = 0
        for _ in range(5):
            a, b = random.randint(0, 10), random.randint(0, 10)
            try:
                r1 = self.registry.execute_function(func_id, [a, b])
                r2 = self.registry.execute_function(func_id, [b, a])
                test_count += 1
                if r1 != r2:
                    commutative = False
                    break
            except:
                continue
        
        if test_count > 0:
            print(f"  {func_name} is {'commutative' if commutative else 'non-commutative'}")
            
            # Store property
            if 'properties' not in meta:
                meta['properties'] = {}
            meta['properties']['commutative'] = commutative
        else:
            print(f"  Could not test {func_name} (execution failed)")
    
    # ==========================================
    # 3. FUNCTION PRUNING AND MERGING
    # ==========================================
    
    def _prune_unused_functions(self):
        """Remove functions that are rarely used"""
        to_prune = []
        total_usage = sum(self.function_usage_counts.values()) or 1
        
        for fid, meta in self.registry.metadata.items():
            if meta.get('is_primitive', False):
                continue
            
            usage_rate = self.function_usage_counts[fid] / total_usage
            if usage_rate < self.config['pruning_threshold']:
                to_prune.append((fid, meta['name'], usage_rate))
        
        if to_prune:
            print(f"\n[Pruning] Removing {len(to_prune)} unused functions:")
            for fid, name, rate in to_prune[:3]:  # Prune top 3
                print(f"  - {name} (usage: {rate:.2%})")
                # Don't actually delete yet - just mark for review
                self.registry.metadata[fid]['marked_for_pruning'] = True
    
    def _identify_merge_candidates(self):
        """Find functions that might be equivalent or similar"""
        functions = [
            (fid, meta) for fid, meta in self.registry.metadata.items()
            if not meta.get('is_primitive', False) and 'signature' in meta
        ]
        
        self.merge_candidates = []
        
        if len(functions) < 2:
            return
        
        for i, (fid1, meta1) in enumerate(functions):
            for fid2, meta2 in functions[i+1:]:
                # Safety check for signature
                sig1 = meta1.get('signature', [])
                sig2 = meta2.get('signature', [])
                
                if len(sig1) != len(sig2):
                    continue
                
                # Test on random inputs
                similarity = 0
                tests = 10
                arity = len(sig1)
                
                if arity == 0:
                    continue
                
                for _ in range(tests):
                    inputs = [random.randint(0, 10) for _ in range(arity)]
                    try:
                        r1 = self.registry.execute_function(fid1, inputs)
                        r2 = self.registry.execute_function(fid2, inputs)
                        if r1 == r2:
                            similarity += 1
                    except:
                        continue
                
                similarity_rate = similarity / tests if tests > 0 else 0
                if similarity_rate > self.config['merge_similarity_threshold']:
                    self.merge_candidates.append({
                        'func1': meta1['name'],
                        'func2': meta2['name'],
                        'similarity': similarity_rate
                    })
        
        if self.merge_candidates:
            print(f"\n[Merge Candidates] Found {len(self.merge_candidates)} similar functions:")
            for candidate in self.merge_candidates[:3]:
                print(f"  - {candidate['func1']} ≈ {candidate['func2']} ({candidate['similarity']:.2%})")
    
    # ==========================================
    # 4. CONTINUOUS LEARNING LOOP
    # ==========================================
    
    def run_continuous_learning(self, num_episodes: int = 10, steps_per_episode: int = 10):
        """Run autonomous learning for multiple episodes"""
        print(f"\n{'='*60}")
        print(f"Autonomous Learning: {num_episodes} episodes")
        print(f"{'='*60}")
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            self.self_play_episode(steps_per_episode)
            
            # Periodic checkpointing
            if (episode + 1) % 5 == 0:
                self.agent.save_checkpoint()
                self.registry.save()
                self._print_statistics()
    
    def _print_statistics(self):
        """Print learning statistics"""
        print(f"\n{'='*60}")
        print("Learning Statistics")
        print(f"{'='*60}")
        print(f"Total functions: {len(self.registry.metadata)}")
        print(f"Discovered: {len(self.discovery_history)}")
        print(f"Merge candidates: {len(self.merge_candidates)}")
        print(f"\nRecent discoveries:")
        for discovery in self.discovery_history[-5:]:
            print(f"  - {discovery['name']} ({discovery['pattern']})")