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
        """Register a new composite function."""
        # Validate composition
        if not composition:
            raise ValueError("Composition cannot be empty")
        
        # Check if name already exists
        for fid, meta in self.metadata.items():
            if meta['name'] == name:
                print(f"  [Warning] Function '{name}' already exists with ID {fid}")
                return fid
        
        # Create composed function
        def composed_fn(inputs):
            available_values = list(inputs)
            
            for func_id, args in composition:
                # Special handling for LOOP
                if func_id == self.loop_id:
                    # LOOP expects: [body_fn_id, count, init_value]
                    # args = [body_fn_id, count_idx, init_idx]
                    if len(args) != 3:
                        raise ValueError(f"LOOP expects 3 args [fn_id, count_idx, init_idx], got {len(args)}")
                    
                    body_fn_id = args[0]  # This is a direct function ID
                    count_idx = args[1]    # This is an index into available_values
                    init_idx = args[2]     # This is an index into available_values
                    
                    if count_idx >= len(available_values) or init_idx >= len(available_values):
                        raise ValueError(f"LOOP arg indices out of range: count_idx={count_idx}, init_idx={init_idx}, have {len(available_values)} values")
                    
                    count = available_values[count_idx]
                    init_value = available_values[init_idx]
                    
                    # Execute LOOP
                    result = self.execute_function(func_id, [body_fn_id, count, init_value])
                    available_values.append(result)
                
                # Special handling for LOOP with encoded function ID (nested LOOP)
                elif func_id == self.loop_id and len(args) > 0 and args[0] < 0:
                    # This is LOOP with a learned function
                    # args = [encoded_func_id, start_input_idx, count_input_idx]
                    from .executor import ProgramExecutor
                    executor = ProgramExecutor(self)
                    
                    # Extract the function ID from encoding
                    loop_func_id = -(args[0] + 1)
                    
                    # Get the actual input values for the loop
                    loop_inputs = [available_values[args[i]] for i in range(1, len(args))]
                    
                    # Execute loop with dynamic count (-1)
                    result = executor._execute_loop(loop_func_id, loop_inputs, loop_count=-1)
                    available_values.append(result)
                
                else:
                    # Normal composition step
                    step_inputs = []
                    for idx in args:
                        if idx < 0 or idx >= len(available_values):
                            raise ValueError(f"Arg index {idx} out of range (have {len(available_values)} values)")
                        step_inputs.append(available_values[idx])
                    
                    result = self.execute_function(func_id, step_inputs)
                    available_values.append(result)
            
            return available_values[-1]
        
        # Calculate layer (max of component layers + 1)
        max_layer = 0
        for func_id, _ in composition:
            if func_id in self.metadata:
                func_layer = self.metadata[func_id].get('layer', 0)
                max_layer = max(max_layer, func_layer)
        
        layer = max_layer + 1
        
        # Get next function ID
        func_id = max(self.functions.keys()) + 1 if self.functions else 0
        
        # Convert composition to JSON-serializable format
        composition_json = json.dumps(composition)
        
        # Add to database using SymbolicDB
        self.db.conn.execute('''
            INSERT INTO functions (id, name, arity, composition, layer)
            VALUES (?, ?, ?, ?, ?)
        ''', (func_id, name, arity, composition_json, layer))
        self.db.conn.commit()
        
        # Register in memory
        self.functions[func_id] = composed_fn
        self.metadata[func_id] = {
            'name': name,
            'arity': arity,
            'composition': composition,
            'layer': layer
        }
        
        print(f"  [DB] Added {name} at layer {layer} with {len(composition)} terms (id={func_id})")
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
            # LOOP implementation: applies a function repeatedly
            # Expects: [func_id, count, init_value]
            def loop_impl(inputs):
                if len(inputs) != 3:
                    raise ValueError(f"LOOP expects 3 args [func_id, count, init], got {len(inputs)}")
                
                func_id = inputs[0]
                count = inputs[1]
                init_value = inputs[2]
                
                result = init_value
                for _ in range(count):
                    result = self.execute_function(func_id, [result])
                
                return result
            
            self.functions[fid] = loop_impl
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
                    # Negative indices encode literal function IDs
                    if idx < 0:
                        # Decode: idx = -func_id-1, so func_id = -idx-1
                        func_id = -idx - 1
                        child_inputs.append(func_id)
                    elif idx < len(available_values):
                        child_inputs.append(available_values[idx])
                    else:
                        raise ValueError(f"Arg index {idx} out of range (have {len(available_values)} values)")
                
                # Special handling for LOOP (arity=-1)
                if child_id == self.loop_id and child_meta['name'] == 'LOOP':
                    if len(child_inputs) != 3:
                        raise ValueError(f"LOOP expects 3 args: (func_id, start, count), got {len(child_inputs)}")
                    
                    # child_inputs = [func_id, start_value, count]
                    loop_func_id = child_inputs[0]
                    start_value = child_inputs[1]
                    count = child_inputs[2]
                    
                    # Apply the function 'count' times
                    result = start_value
                    for _ in range(count):
                        result = self.execute_function(loop_func_id, [result])
                
                # Normal function execution
                elif child_arity != -1:
                    # Ensure we have the right number of arguments
                    if len(child_inputs) != child_arity:
                        raise ValueError(f"Function {child_meta['name']} expects {child_arity} args, got {len(child_inputs)}")
                    
                    # Execute the child function
                    result = self.execute_function(child_id, child_inputs)
                
                else:
                    # Variable arity function (not LOOP) - just pass all args
                    result = self.execute_function(child_id, child_inputs)
                
                # Add result to available values for next function
                available_values.append(result)
            
            # Return the last computed value
            return available_values[-1]
        
        self.functions[fid] = composed_fn
        self.compositions[fid] = {'terms': composition}

    def get_vocab_size(self) -> int:
        """Get vocabulary size from database (actual number of registered functions)."""
        cursor = self.db.conn.execute("SELECT COUNT(*) FROM functions")
        count = cursor.fetchone()[0]
        # Ensure _next_id is at least count (for new registrations)
        self._next_id = max(self._next_id, count)
        return count

    def execute_function(self, func_id: int, inputs: List[Any]) -> Any:
        """Execute a function by ID with given inputs."""
        if func_id not in self.functions:
            raise ValueError(f"Function {func_id} not found in registry")
        
        func = self.functions[func_id]
        return func(inputs)