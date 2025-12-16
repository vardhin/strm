import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Callable

class SymbolicRegistry:
    """Registry for symbolic functions with save/load capabilities."""
    
    def __init__(self):
        self.functions: Dict[int, Callable] = {}
        self.metadata: Dict[int, Dict] = {}
        self.compositions: Dict[int, Dict] = {}
        self.loop_id = None
        self._next_id = 0
        
        # Initialize with primitives
        self._initialize_primitives()
    
    def save(self, filepath: str):
        """Save registry to disk."""
        data = {
            'metadata': self.metadata,
            'compositions': self.compositions,
            'loop_id': self.loop_id,
            'next_id': self._next_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  [Registry] Saved to {filepath}")
    
    def load(self, filepath: str):
        """Load registry from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Clear existing data
        self.functions = {}
        self.metadata = {}
        self.compositions = {}
        
        # Restore metadata and compositions
        self.metadata = data['metadata']
        self.compositions = data['compositions']
        self.loop_id = data['loop_id']
        self._next_id = data['next_id']
        
        # First pass: rebuild all primitive functions
        for fid in sorted(self.metadata.keys()):
            if fid not in self.compositions:
                # This is a primitive function
                self._rebuild_primitive(fid)
        
        # Second pass: rebuild all composed functions
        # (must be done after primitives are available)
        for fid in sorted(self.compositions.keys()):
            comp_data = self.compositions[fid]
            self._rebuild_composition(fid, comp_data)
        
        print(f"  [Registry] Loaded from {filepath}")
        print(f"  [Registry] Vocabulary size: {len(self.metadata)}")
        print(f"  [Registry] Functions: {[m['name'] for m in self.metadata.values()]}")
    
    def _rebuild_primitive(self, fid: int):
        """Rebuild a primitive function."""
        name = self.metadata[fid]['name']
        
        if name == 'OR':
            self.functions[fid] = lambda inputs: inputs[0] | inputs[1]
        elif name == 'AND':
            self.functions[fid] = lambda inputs: inputs[0] & inputs[1]
        elif name == 'NOT':
            self.functions[fid] = lambda inputs: ~inputs[0]
        elif name == 'INC':
            self.functions[fid] = lambda inputs: inputs[0] + 1
        elif name == 'DEC':
            self.functions[fid] = lambda inputs: inputs[0] - 1
        elif name == 'LOOP':
            # LOOP is special - it's a higher-order function handled elsewhere
            pass
        else:
            raise ValueError(f"Unknown primitive function: {name}")
    
    def _rebuild_composition(self, fid: int, comp_data: Dict):
        """Rebuild a composed function from saved data."""
        comp_type = comp_data['composition']
        primary_id = comp_data['primary_id']
        secondary_id = comp_data['secondary_id']
        tertiary_id = comp_data.get('tertiary_id')
        loop_count = comp_data.get('loop_count', 1)
        
        if comp_type == 'sequential':
            if secondary_id == self.loop_id:
                # Rebuild LOOP composition
                def loop_fn(inputs, p=primary_id, lc=loop_count):
                    result = inputs[0] if len(inputs) == 1 else inputs
                    for _ in range(lc):
                        result = self.execute_function(p, [result] if not isinstance(result, list) else result)
                    return result
                self.functions[fid] = loop_fn
            else:
                # Rebuild sequential composition: f2(f1(inputs))
                def seq_fn(inputs, p=primary_id, s=secondary_id):
                    intermediate = self.execute_function(p, inputs)
                    return self.execute_function(s, [intermediate])
                self.functions[fid] = seq_fn
        
        elif comp_type == 'nested':
            # Rebuild nested composition: f1(f2(x) for each x in inputs)
            # But check if secondary expects multiple inputs
            secondary_arity = self.metadata[secondary_id]['arity']
            
            if secondary_arity == 1:
                # Apply secondary to each element individually
                def nested_fn(inputs, p=primary_id, s=secondary_id):
                    transformed = [self.execute_function(s, [x]) for x in inputs]
                    return self.execute_function(p, transformed)
                self.functions[fid] = nested_fn
            else:
                # Secondary needs multiple inputs - just pass inputs through
                def nested_fn(inputs, p=primary_id, s=secondary_id):
                    transformed = self.execute_function(s, inputs)
                    return self.execute_function(p, [transformed])
                self.functions[fid] = nested_fn
        
        elif comp_type == 'parallel':
            # Rebuild parallel composition: tertiary(primary(inputs), secondary(inputs))
            def parallel_fn(inputs, p=primary_id, s=secondary_id, t=tertiary_id):
                result1 = self.execute_function(p, inputs)
                result2 = self.execute_function(s, inputs)
                return self.execute_function(t, [result1, result2])
            self.functions[fid] = parallel_fn

    def _create_loop_function(self, primary_id: int, loop_count: int):
        """Create a loop function closure."""
        def loop_fn(inputs):
            result = inputs[0] if len(inputs) == 1 else inputs
            for _ in range(loop_count):
                result = self.execute_function(primary_id, [result] if not isinstance(result, list) else result)
            return result
        return loop_fn

    def _initialize_primitives(self):
        """Register minimal primitive set for Turing completeness."""
        # Bitwise operations (for bit manipulation)
        self.register("OR", lambda inputs: inputs[0] | inputs[1], arity=2)
        self.register("AND", lambda inputs: inputs[0] & inputs[1], arity=2)
        self.register("NOT", lambda inputs: ~inputs[0], arity=1)
        
        # Arithmetic primitives (successor and predecessor)
        self.register("INC", lambda inputs: inputs[0] + 1, arity=1)
        self.register("DEC", lambda inputs: inputs[0] - 1, arity=1)
        
        # Meta-function for iteration (higher-order)
        self.loop_id = self.register("LOOP", None, arity=-1)

    def register(self, name: str, func: Callable, arity: int, layer: int = -1) -> int:
        """Saves a new function to the registry."""
        fid = self._next_id
        self.functions[fid] = func
        if layer == -1:
            layer = self._compute_layer(func)
        self.metadata[fid] = {"name": name, "arity": arity, "layer": layer}
        self._next_id += 1
        return fid

    def _compute_layer(self, func: Callable) -> int:
        """Compute abstraction layer based on function composition."""
        if not self.metadata:
            return 0
        return max(m['layer'] for m in self.metadata.values()) + 1

    def get_vocab_size(self) -> int:
        return self._next_id

    def execute_function(self, func_id: int, inputs: List[Any]) -> Any:
        """Execute a function by ID with given inputs.
        
        Args:
            func_id: ID of function to execute
            inputs: List of input values
            
        Returns:
            Result of function execution
        """
        if func_id not in self.functions:
            raise ValueError(f"Function {func_id} not found in registry")
        
        func = self.functions[func_id]
        
        # All functions now expect inputs as a list
        return func(inputs)