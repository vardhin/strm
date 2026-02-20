import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time

class BootstrappingAgent:
    """Self-bootstrapping agent that builds on its own learned functions"""
    
    def __init__(self, agent, registry, config=None):
        self.agent = agent
        self.registry = registry
        
        self.config = config or {
            'examples_per_composition': 8,
            'min_arity': 1,
            'max_arity': 3,
            'composition_depth_range': (1, 3),
            'learning_epochs': 25,
            'success_threshold': 0.8,
        }
        
        # Tracking
        self.generation = 0
        self.successful_functions = []
        self.failed_attempts = []
        
    # ==========================================
    # CORE: Self-Bootstrapping Loop
    # ==========================================
    
    def bootstrap_loop(self, num_iterations: int = 50):
        """Main bootstrapping loop: compose -> generate examples -> learn -> repeat"""
        print(f"\n{'='*60}")
        print(f"Bootstrapping Agent: {num_iterations} iterations")
        print(f"{'='*60}")
        
        for iteration in range(num_iterations):
            self.generation += 1
            
            print(f"\n{'─'*60}")
            print(f"Generation {self.generation} / {num_iterations}")
            print(f"{'─'*60}")
            print(f"Current library: {len(self.registry.metadata)} functions")
            
            # Step 1: Create random composition
            composition_spec = self._create_random_composition()
            if not composition_spec:
                print("  ✗ Could not create composition (insufficient functions)")
                continue
            
            print(f"  Composition: {composition_spec['description']}")
            
            # Step 2: Generate examples using this composition
            examples = self._generate_examples_from_composition(composition_spec)
            if not examples:
                print("  ✗ Could not generate valid examples")
                self.failed_attempts.append(composition_spec)
                continue
            
            print(f"  Generated {len(examples)} examples")
            self._print_sample_examples(examples, n=3)
            
            # Step 3: Try to learn it
            function_name = f"GEN{self.generation}_{composition_spec['type']}"
            success = self.agent.learn_abstraction(
                function_name,
                examples,
                num_epochs=self.config['learning_epochs']
            )
            
            if success:
                print(f"  ✓ Successfully learned {function_name}!")
                self.successful_functions.append({
                    'name': function_name,
                    'generation': self.generation,
                    'composition': composition_spec,
                    'examples': len(examples)
                })
            else:
                print(f"  ✗ Failed to learn {function_name}")
                self.failed_attempts.append(composition_spec)
            
            # Periodic stats
            if self.generation % 10 == 0:
                self._print_progress_stats()
                self.agent.save_checkpoint()
                self.registry.save()
    
    # ==========================================
    # COMPOSITION GENERATION
    # ==========================================
    
    def _create_random_composition(self) -> Optional[Dict]:
        """Create a random composition specification"""
        available = self._get_usable_functions()
        
        if len(available) < 2:
            return None
        
        composition_type = random.choice([
            'sequential',  # f(g(x))
            'parallel',    # h(f(x), g(y))
            'conditional', # if f(x) then g(y) else h(z)
            'iterative',   # loop or accumulate
        ])
        
        if composition_type == 'sequential':
            return self._create_sequential_composition(available)
        elif composition_type == 'parallel':
            return self._create_parallel_composition(available)
        elif composition_type == 'conditional':
            return self._create_conditional_composition(available)
        elif composition_type == 'iterative':
            return self._create_iterative_composition(available)
    
    def _get_usable_functions(self) -> List[Tuple[int, Dict]]:
        """Get functions that can be used in compositions"""
        usable = []
        for fid, meta in self.registry.metadata.items():
            # Get arity
            if 'signature' in meta:
                arity = len(meta['signature'])
            else:
                # Primitive - check name
                name = meta['name']
                if name in ['OR', 'AND', 'LT', 'LTE', 'GT', 'GTE', 'EQ', 'NEQ']:
                    arity = 2
                elif name in ['NOT', 'INC', 'DEC', 'CONST']:
                    arity = 1
                elif name in ['COND']:
                    arity = 3
                else:
                    continue  # Skip LOOP, WHILE, ACCUM for now
            
            usable.append((fid, meta, arity))
        
        return usable
    
    def _create_sequential_composition(self, available) -> Dict:
        """f(g(x)) - chain two functions"""
        # Pick two functions where output of second feeds into first
        func1_id, func1_meta, func1_arity = random.choice(available)
        func2_id, func2_meta, func2_arity = random.choice(available)
        
        # For simplicity, chain unary functions or use first output
        return {
            'type': 'sequential',
            'functions': [func1_id, func2_id],
            'description': f"{func1_meta['name']}({func2_meta['name']}(x))",
            'arity': func2_arity,
            'compose_fn': lambda inputs: self._execute_sequential(func1_id, func2_id, inputs)
        }
    
    def _create_parallel_composition(self, available) -> Dict:
        """h(f(x), g(y)) - apply two functions then combine"""
        # Pick two functions and a combiner
        func1_id, func1_meta, func1_arity = random.choice(available)
        func2_id, func2_meta, func2_arity = random.choice(available)
        
        # Pick a binary combiner
        binary_funcs = [(fid, m, a) for fid, m, a in available if a == 2]
        if not binary_funcs:
            return self._create_sequential_composition(available)
        
        combiner_id, combiner_meta, _ = random.choice(binary_funcs)
        
        total_arity = func1_arity + func2_arity
        
        return {
            'type': 'parallel',
            'functions': [func1_id, func2_id, combiner_id],
            'description': f"{combiner_meta['name']}({func1_meta['name']}(x), {func2_meta['name']}(y))",
            'arity': total_arity,
            'compose_fn': lambda inputs: self._execute_parallel(
                func1_id, func2_id, combiner_id, func1_arity, inputs
            )
        }
    
    def _create_conditional_composition(self, available) -> Dict:
        """COND(pred(x), then(y), else(z))"""
        # Find COND primitive
        cond_id = None
        for fid, meta in self.registry.metadata.items():
            if meta['name'] == 'COND':
                cond_id = fid
                break
        
        if cond_id is None:
            return self._create_sequential_composition(available)
        
        # Pick predicate and two branches
        pred_id, pred_meta, pred_arity = random.choice(available)
        then_id, then_meta, then_arity = random.choice(available)
        else_id, else_meta, else_arity = random.choice(available)
        
        return {
            'type': 'conditional',
            'functions': [cond_id, pred_id, then_id, else_id],
            'description': f"COND({pred_meta['name']}, {then_meta['name']}, {else_meta['name']})",
            'arity': max(pred_arity, then_arity, else_arity),
            'compose_fn': lambda inputs: self._execute_conditional(
                cond_id, pred_id, then_id, else_id, inputs
            )
        }
    
    def _create_iterative_composition(self, available) -> Dict:
        """LOOP or ACCUM with a function"""
        # Find iteration primitive
        loop_id = None
        for fid, meta in self.registry.metadata.items():
            if meta['name'] == 'LOOP':
                loop_id = fid
                break
        
        if loop_id is None:
            return self._create_sequential_composition(available)
        
        body_id, body_meta, body_arity = random.choice(available)
        
        return {
            'type': 'iterative',
            'functions': [loop_id, body_id],
            'description': f"LOOP({body_meta['name']}, n)",
            'arity': 2,  # value and count
            'compose_fn': lambda inputs: self._execute_iterative(loop_id, body_id, inputs)
        }
    
    # ==========================================
    # EXECUTION HELPERS
    # ==========================================
    
    def _execute_sequential(self, f1_id, f2_id, inputs):
        """Execute f(g(x))"""
        intermediate = self.registry.execute_function(f2_id, inputs)
        return self.registry.execute_function(f1_id, [intermediate])
    
    def _execute_parallel(self, f1_id, f2_id, combiner_id, split_at, inputs):
        """Execute combine(f(x), g(y))"""
        inputs1 = inputs[:split_at]
        inputs2 = inputs[split_at:]
        
        result1 = self.registry.execute_function(f1_id, inputs1)
        result2 = self.registry.execute_function(f2_id, inputs2) if inputs2 else 0
        
        return self.registry.execute_function(combiner_id, [result1, result2])
    
    def _execute_conditional(self, cond_id, pred_id, then_id, else_id, inputs):
        """Execute COND(pred, then, else)"""
        pred_result = self.registry.execute_function(pred_id, inputs[:1])
        
        if pred_result:
            return self.registry.execute_function(then_id, inputs[:1])
        else:
            return self.registry.execute_function(else_id, inputs[:1])
    
    def _execute_iterative(self, loop_id, body_id, inputs):
        """Execute LOOP(body, n)"""
        if len(inputs) < 2:
            return inputs[0] if inputs else 0
        
        value, count = inputs[0], min(inputs[1], 10)  # Cap iterations
        
        for _ in range(count):
            value = self.registry.execute_function(body_id, [value])
        
        return value
    
    # ==========================================
    # EXAMPLE GENERATION
    # ==========================================
    
    def _generate_examples_from_composition(self, spec: Dict) -> List[Tuple[List, int]]:
        """Generate examples by executing the composition"""
        examples = []
        compose_fn = spec['compose_fn']
        arity = spec['arity']
        
        attempts = 0
        max_attempts = 50
        
        while len(examples) < self.config['examples_per_composition'] and attempts < max_attempts:
            attempts += 1
            
            # Generate random inputs
            inputs = [random.randint(0, 15) for _ in range(arity)]
            
            try:
                output = compose_fn(inputs)
                
                # Validate output
                if isinstance(output, (int, np.integer)) and -1000 < output < 1000:
                    examples.append((inputs, int(output)))
            except:
                continue
        
        return examples
    
    # ==========================================
    # UTILITIES
    # ==========================================
    
    def _print_sample_examples(self, examples, n=3):
        """Print sample examples"""
        for inputs, output in examples[:n]:
            input_str = ', '.join(map(str, inputs))
            print(f"    f({input_str}) = {output}")
    
    def _print_progress_stats(self):
        """Print progress statistics"""
        success_rate = len(self.successful_functions) / max(self.generation, 1)
        
        print(f"\n{'='*60}")
        print(f"Progress Report - Generation {self.generation}")
        print(f"{'='*60}")
        print(f"Total functions: {len(self.registry.metadata)}")
        print(f"Learned this session: {len(self.successful_functions)}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Failed attempts: {len(self.failed_attempts)}")
        
        if self.successful_functions:
            print(f"\nRecent successes:")
            for func in self.successful_functions[-5:]:
                print(f"  - {func['name']}: {func['composition']['description']}")