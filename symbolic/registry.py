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
        
        # Comparison primitives (for conditionals)
        lt_id = self.register("LT", lambda inputs: 1 if inputs[0] < inputs[1] else 0, arity=2)
        lte_id = self.register("LTE", lambda inputs: 1 if inputs[0] <= inputs[1] else 0, arity=2)
        gt_id = self.register("GT", lambda inputs: 1 if inputs[0] > inputs[1] else 0, arity=2)
        gte_id = self.register("GTE", lambda inputs: 1 if inputs[0] >= inputs[1] else 0, arity=2)
        eq_id = self.register("EQ", lambda inputs: 1 if inputs[0] == inputs[1] else 0, arity=2)
        neq_id = self.register("NEQ", lambda inputs: 1 if inputs[0] != inputs[1] else 0, arity=2)
        
        # Conditional primitive (if-then-else)
        def cond_impl(inputs):
            """COND(condition, then_value, else_value) - ternary operator"""
            if len(inputs) != 3:
                raise ValueError(f"COND expects 3 args [cond, then, else], got {len(inputs)}")
            return inputs[1] if inputs[0] != 0 else inputs[2]
        
        cond_id = self.register("COND", cond_impl, arity=3)
        
        # Constant/Identity primitive (for representing constants)
        const_id = self.register("CONST", lambda inputs: inputs[0], arity=1)
        
        # Enhanced LOOP: Can take function IDs or evaluate compositions dynamically
        def loop_impl(inputs):
            """
            LOOP(body_expr, count_expr, *args)
            
            Three modes:
            1. LOOP(func_id, count, init) - original: apply func 'count' times
            2. LOOP(body_composition, count_composition, arg1, arg2, ...) - dynamic evaluation
            
            Example: LOOP(ADD(2,3), ADD(1,2)) for MUL
              - Evaluates ADD(2,3) = 5 (value to add each iteration)
              - Evaluates ADD(1,2) = 3 (number of iterations)
              - Returns: 5 + 5 + 5 = 15 (accumulator starts at 0)
            """
            if len(inputs) < 2:
                raise ValueError(f"LOOP expects at least 2 args, got {len(inputs)}")
            
            body_expr = inputs[0]
            count_expr = inputs[1]
            remaining_args = inputs[2:] if len(inputs) > 2 else []
            
            # Case 1: Traditional LOOP(func_id, count, init)
            if len(inputs) == 3 and isinstance(body_expr, int) and isinstance(count_expr, int):
                func_id = body_expr
                count = count_expr
                init_value = remaining_args[0]
                
                result = init_value
                for _ in range(count):
                    result = self.execute_function(func_id, [result])
                
                return result
            
            # Case 2: Dynamic evaluation LOOP(body_composition, count_composition, args...)
            # Body and count are actually compositions or function calls
            # We need to evaluate them with the provided arguments
            
            # If body_expr is a function ID, evaluate it with remaining args
            if isinstance(body_expr, int) and body_expr in self.functions:
                # This is a function ID - execute it
                if len(remaining_args) >= self.metadata[body_expr]['arity']:
                    body_arity = self.metadata[body_expr]['arity']
                    body_args = remaining_args[:body_arity]
                    body_value = self.execute_function(body_expr, body_args)
                else:
                    # Not enough args, treat as the value itself
                    body_value = body_expr
            else:
                # It's already a value
                body_value = body_expr
            
            # Same for count
            if isinstance(count_expr, int) and count_expr in self.functions:
                if len(remaining_args) >= self.metadata[count_expr]['arity']:
                    count_arity = self.metadata[count_expr]['arity']
                    # Use different args if available, otherwise reuse
                    count_start = len(remaining_args) - count_arity if len(remaining_args) > count_arity else 0
                    count_args = remaining_args[count_start:count_start + count_arity]
                    count_value = self.execute_function(count_expr, count_args)
                else:
                    count_value = count_expr
            else:
                count_value = count_expr
            
            # Now perform the loop: accumulate body_value, count_value times
            accumulator = 0
            for _ in range(count_value):
                accumulator += body_value
            
            return accumulator
        
        self.loop_id = self.register("LOOP", loop_impl, arity=-1)
        
        # Conditional loop (while loop)
        def while_impl(inputs):
            """
            WHILE(cond_fn_id, body_fn_id, state, limit)
            Execute body_fn(state) while cond_fn(state) != 0
            Returns final state
            """
            if len(inputs) != 4:
                raise ValueError(f"WHILE expects 4 args [cond_fn, body_fn, state, limit], got {len(inputs)}")
            
            cond_fn_id = inputs[0]
            body_fn_id = inputs[1]
            state = inputs[2]
            limit = inputs[3]
            
            iterations = 0
            while iterations < limit:
                cond_result = self.execute_function(cond_fn_id, [state])
                if cond_result == 0:  # Condition false
                    break
                state = self.execute_function(body_fn_id, [state])
                iterations += 1
            
            return state
        
        while_id = self.register("WHILE", while_impl, arity=-1)
        
        # Accumulator (counting loop)
        def accum_impl(inputs):
            """
            ACCUM(cond_fn_id, body_fn_id, state, counter, limit)
            Like WHILE but returns counter (number of iterations)
            Used for operations like division (counting subtractions)
            """
            if len(inputs) != 5:
                raise ValueError(f"ACCUM expects 5 args [cond_fn, body_fn, state, counter, limit], got {len(inputs)}")
            
            cond_fn_id = inputs[0]
            body_fn_id = inputs[1]
            state = inputs[2]
            counter = inputs[3]
            limit = inputs[4]
            
            iterations = 0
            while iterations < limit:
                cond_result = self.execute_function(cond_fn_id, [state])
                if cond_result == 0:  # Condition false
                    break
                state = self.execute_function(body_fn_id, [state])
                counter += 1
                iterations += 1
            
            return counter
        
        accum_id = self.register("ACCUM", accum_impl, arity=-1)
        
        # Register all primitives to DB
        primitive_ids = [
            or_id, and_id, not_id,
            inc_id, dec_id,
            lt_id, lte_id, gt_id, gte_id, eq_id, neq_id,
            cond_id, const_id,
            self.loop_id, while_id, accum_id
        ]
        
        for fid in primitive_ids:
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
                    # Two modes based on number of args:
                    # 1. LOOP with 3 args: [body_fn_id, count_idx, init_idx] - traditional
                    # 2. LOOP with 2 args: [body_idx, count_idx] - dynamic (for MUL)
                    
                    if len(args) == 2:
                        # Dynamic mode: LOOP(body_fn, count_fn, *original_inputs)
                        body_idx = args[0]
                        count_idx = args[1]
                        
                        if body_idx >= len(available_values) or count_idx >= len(available_values):
                            raise ValueError(f"LOOP arg indices out of range: body_idx={body_idx}, count_idx={count_idx}, have {len(available_values)} values")
                        
                        # Get the function IDs or values
                        body_fn_or_val = available_values[body_idx]
                        count_fn_or_val = available_values[count_idx]
                        
                        # Call LOOP with these + original inputs for evaluation
                        # LOOP will evaluate them and accumulate
                        result = self.execute_function(func_id, [body_fn_or_val, count_fn_or_val] + list(inputs))
                        available_values.append(result)
                    
                    elif len(args) == 3:
                        # Traditional mode: LOOP(body_fn_id, count, init_value)
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
                    else:
                        raise ValueError(f"LOOP expects 2 or 3 args, got {len(args)}")
                
                # Special handling for LOOP with encoded function ID (nested LOOP)
                elif func_id == self.loop_id and len(args) > 0 and args[0] < 0:
                    # This is LOOP with a learned function
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
        elif name == 'LT':
            self.functions[fid] = lambda inputs: 1 if inputs[0] < inputs[1] else 0
        elif name == 'LTE':
            self.functions[fid] = lambda inputs: 1 if inputs[0] <= inputs[1] else 0
        elif name == 'GT':
            self.functions[fid] = lambda inputs: 1 if inputs[0] > inputs[1] else 0
        elif name == 'GTE':
            self.functions[fid] = lambda inputs: 1 if inputs[0] >= inputs[1] else 0
        elif name == 'EQ':
            self.functions[fid] = lambda inputs: 1 if inputs[0] == inputs[1] else 0
        elif name == 'NEQ':
            self.functions[fid] = lambda inputs: 1 if inputs[0] != inputs[1] else 0
        elif name == 'COND':
            self.functions[fid] = lambda inputs: inputs[1] if inputs[0] != 0 else inputs[2]
        elif name == 'CONST':
            self.functions[fid] = lambda inputs: inputs[0]
        elif name == 'LOOP':
            self.loop_id = fid
            def loop_impl(inputs):
                if len(inputs) < 2:
                    raise ValueError(f"LOOP expects at least 2 args, got {len(inputs)}")
                
                body_expr = inputs[0]
                count_expr = inputs[1]
                remaining_args = inputs[2:] if len(inputs) > 2 else []
                
                # Traditional mode: LOOP(func_id, count, init)
                if len(inputs) == 3 and isinstance(body_expr, int) and isinstance(count_expr, int):
                    func_id = body_expr
                    count = count_expr
                    init_value = remaining_args[0]
                    
                    result = init_value
                    for _ in range(count):
                        result = self.execute_function(func_id, [result])
                    
                    return result
                
                # Dynamic evaluation mode
                if isinstance(body_expr, int) and body_expr in self.functions:
                    if len(remaining_args) >= self.metadata[body_expr]['arity']:
                        body_arity = self.metadata[body_expr]['arity']
                        body_args = remaining_args[:body_arity]
                        body_value = self.execute_function(body_expr, body_args)
                    else:
                        body_value = body_expr
                else:
                    body_value = body_expr
                
                if isinstance(count_expr, int) and count_expr in self.functions:
                    if len(remaining_args) >= self.metadata[count_expr]['arity']:
                        count_arity = self.metadata[count_expr]['arity']
                        count_start = len(remaining_args) - count_arity if len(remaining_args) > count_arity else 0
                        count_args = remaining_args[count_start:count_start + count_arity]
                        count_value = self.execute_function(count_expr, count_args)
                    else:
                        count_value = count_expr
                else:
                    count_value = count_expr
                
                # Accumulate
                accumulator = 0
                for _ in range(count_value):
                    accumulator += body_value
                
                return accumulator
            
            self.functions[fid] = loop_impl
        elif name == 'WHILE':
            def while_impl(inputs):
                if len(inputs) != 4:
                    raise ValueError(f"WHILE expects 4 args [cond_fn, body_fn, state, limit], got {len(inputs)}")
                cond_fn_id = inputs[0]
                body_fn_id = inputs[1]
                state = inputs[2]
                limit = inputs[3]
                iterations = 0
                while iterations < limit:
                    cond_result = self.execute_function(cond_fn_id, [state])
                    if cond_result == 0:
                        break
                    state = self.execute_function(body_fn_id, [state])
                    iterations += 1
                return state
            self.functions[fid] = while_impl
        elif name == 'ACCUM':
            def accum_impl(inputs):
                if len(inputs) != 5:
                    raise ValueError(f"ACCUM expects 5 args [cond_fn, body_fn, state, counter, limit], got {len(inputs)}")
                cond_fn_id = inputs[0]
                body_fn_id = inputs[1]
                state = inputs[2]
                counter = inputs[3]
                limit = inputs[4]
                iterations = 0
                while iterations < limit:
                    cond_result = self.execute_function(cond_fn_id, [state])
                    if cond_result == 0:
                        break
                    state = self.execute_function(body_fn_id, [state])
                    counter += 1
                    iterations += 1
                return counter
            self.functions[fid] = accum_impl
        else:
            raise ValueError(f"Unknown primitive function: {name}")
    
    def _rebuild_composition(self, fid: int, composition: List[Tuple[int, List[int]]]):
        """Rebuild a composed function from DB data."""
        def composed_fn(inputs, comp=composition):
            available_values = list(inputs) if isinstance(inputs, list) else [inputs]
            
            for func_id, args in comp:
                # Special handling for LOOP
                if func_id == self.loop_id:
                    # Two modes based on number of args:
                    # 1. LOOP with 3 args: [body_fn_id, count_idx, init_idx] - traditional
                    # 2. LOOP with 2 args: [body_idx, count_idx] - dynamic (for MUL)
                    
                    if len(args) == 2:
                        # Dynamic mode: LOOP(body_fn, count_fn, *original_inputs)
                        body_idx = args[0]
                        count_idx = args[1]
                        
                        if body_idx >= len(available_values) or count_idx >= len(available_values):
                            raise ValueError(f"LOOP arg indices out of range: body_idx={body_idx}, count_idx={count_idx}, have {len(available_values)} values")
                        
                        # Get the function IDs or values
                        body_fn_or_val = available_values[body_idx]
                        count_fn_or_val = available_values[count_idx]
                        
                        # Call LOOP with these + original inputs for evaluation
                        # LOOP will evaluate them and accumulate
                        result = self.execute_function(func_id, [body_fn_or_val, count_fn_or_val] + list(inputs))
                        available_values.append(result)
                    
                    elif len(args) == 3:
                        # Traditional mode: LOOP(body_fn_id, count, init_value)
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
                    else:
                        raise ValueError(f"LOOP expects 2 or 3 args, got {len(args)}")
                
                # Special handling for LOOP with encoded function ID (nested LOOP)
                elif func_id == self.loop_id and len(args) > 0 and args[0] < 0:
                    # This is LOOP with a learned function
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