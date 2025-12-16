import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from symbolic_db import SymbolicDB

class SymbolicRegistry:
    """Registry for symbolic functions with SQLite database storage."""
    
    def __init__(self):
        self.functions: Dict[int, Callable] = {}
        self.metadata: Dict[int, Dict] = {}
        self.compositions: Dict[int, Dict] = {}
        self.loop_id = None
        self._next_id = 0
        
        # Initialize DB first
        self.db = SymbolicDB()
        
        # Then initialize primitives (which will register to DB)
        self._initialize_primitives()
    
    def _initialize_primitives(self):
        """Register minimal primitive set for Turing completeness."""
        # Bitwise operations (for bit manipulation)
        or_id = self.register("OR", lambda inputs: inputs[0] | inputs[1], arity=2)
        and_id = self.register("AND", lambda inputs: inputs[0] & inputs[1], arity=2)
        not_id = self.register("NOT", lambda inputs: ~inputs[0], arity=1)
        
        # Arithmetic primitives (successor and predecessor)
        inc_id = self.register("INC", lambda inputs: inputs[0] + 1, arity=1)
        dec_id = self.register("DEC", lambda inputs: inputs[0] - 1, arity=1)
        
        # Meta-function for iteration (higher-order)
        self.loop_id = self.register("LOOP", None, arity=-1)
        
        # Register all primitives to DB
        for fid in [or_id, and_id, not_id, inc_id, dec_id, self.loop_id]:
            meta = self.metadata[fid]
            self.db.add_primitive(fid, meta['name'], meta['arity'])

    def register(self, name: str, func: Callable, arity: int, is_primitive: bool = True) -> int:
        """Register a new function (primitive or learned)."""
        fid = self._next_id
        self.functions[fid] = func
        self.metadata[fid] = {"name": name, "arity": arity, "layer": 0}  # layer will be set by DB
        self._next_id += 1
        
        return fid

    def register_composition(self, name: str, arity: int, composition: List[Tuple[int, List[int]]]) -> int:
        """Register a learned abstraction and save to DB.
        
        Args:
            name: Function name
            arity: Number of arguments
            composition: List of (child_func_id, arg_indices) tuples
        
        Returns:
            New function ID
        """
        func_id = self._next_id
        
        # Create executable function from composition
        def composed_fn(inputs, comp=composition):
            # Start with original inputs as a list
            available_values = list(inputs) if isinstance(inputs, list) else [inputs]
            
            # Execute each function in the composition
            for child_id, arg_indices in comp:
                # Get the child function's metadata
                child_meta = self.metadata[child_id]
                child_arity = child_meta['arity']
                
                # Build child inputs by mapping indices to available values
                child_inputs = []
                for idx in arg_indices:
                    if idx < len(available_values):
                        child_inputs.append(available_values[idx])
                    else:
                        raise ValueError(f"Arg index {idx} out of range (have {len(available_values)} values)")
                
                # Ensure we have the right number of arguments
                if len(child_inputs) != child_arity:
                    raise ValueError(f"Function {child_meta['name']} expects {child_arity} args, got {len(child_inputs)}")
                
                # Execute the child function
                result = self.execute_function(child_id, child_inputs)
                
                # Add result to available values for next function
                available_values.append(result)
            
            # Return the last computed value
            return available_values[-1]
        
        self.functions[func_id] = composed_fn
        self.metadata[func_id] = {"name": name, "arity": arity, "layer": -1}  # will be set by DB
        self.compositions[func_id] = {
            'terms': composition
        }
        self._next_id += 1
        
        # Save to database - this calculates layer automatically AND commits
        self.db.add_abstraction(func_id, name, arity, composition)
        
        # Update layer in metadata from DB
        db_func = self.db.get_function(func_id)
        if db_func:
            self.metadata[func_id]['layer'] = db_func['layer']
        
        return func_id

    def save(self, path: str = "checkpoints/registry.pkl"):
        """Save registry - now just prints DB summary."""
        print(f"  [Registry] All data saved to database: {self.db.db_path}")
        self.db.print_summary()
    
    def load(self, db_path: str = "checkpoints/symbolic.db"):
        """Load registry from database."""
        self.db = SymbolicDB(db_path)
        
        # Clear existing data
        self.functions = {}
        self.metadata = {}
        self.compositions = {}
        self._next_id = 0
        
        # Rebuild from database
        all_funcs = self.db.get_all_functions()
        
        for func_data in all_funcs:
            fid = func_data['id']
            name = func_data['name']
            arity = func_data['arity']
            layer = func_data['layer']
            
            self.metadata[fid] = {
                'name': name,
                'arity': arity,
                'layer': layer
            }
            
            # Check if primitive by layer
            if layer == 0:
                self._rebuild_primitive(fid)
            else:
                composition = self.db.get_composition(fid)
                self._rebuild_composition(fid, composition)
            
            self._next_id = max(self._next_id, fid + 1)
        
        print(f"  [Registry] Loaded {len(all_funcs)} functions from {db_path}")
        self.db.print_summary()
    
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
            self.loop_id = fid
            # LOOP is special - handled by agent
        else:
            raise ValueError(f"Unknown primitive function: {name}")
    
    def _rebuild_composition(self, fid: int, composition: List[Tuple[int, List[int]]]):
        """Rebuild a composed function from DB data."""
        def composed_fn(inputs, comp=composition):
            # Start with original inputs as a list
            available_values = list(inputs) if isinstance(inputs, list) else [inputs]
            
            # Execute each function in the composition
            for child_id, arg_indices in comp:
                # Get the child function's metadata
                child_meta = self.metadata[child_id]
                child_arity = child_meta['arity']
                
                # Build child inputs by mapping indices to available values
                child_inputs = []
                for idx in arg_indices:
                    if idx < len(available_values):
                        child_inputs.append(available_values[idx])
                    else:
                        raise ValueError(f"Arg index {idx} out of range (have {len(available_values)} values)")
                
                # Ensure we have the right number of arguments
                if len(child_inputs) != child_arity:
                    raise ValueError(f"Function {child_meta['name']} expects {child_arity} args, got {len(child_inputs)}")
                
                # Execute the child function
                result = self.execute_function(child_id, child_inputs)
                
                # Add result to available values for next function
                available_values.append(result)
            
            # Return the last computed value
            return available_values[-1]
        
        self.functions[fid] = composed_fn
        self.compositions[fid] = {'terms': composition}

    def get_vocab_size(self) -> int:
        return self._next_id

    def execute_function(self, func_id: int, inputs: List[Any]) -> Any:
        """Execute a function by ID with given inputs."""
        if func_id not in self.functions:
            raise ValueError(f"Function {func_id} not found in registry")
        
        func = self.functions[func_id]
        return func(inputs)