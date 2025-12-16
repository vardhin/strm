from typing import Optional, Any, List, Tuple
from .registry import SymbolicRegistry

class ProgramExecutor:
    """Executes symbolic programs with composition support."""
    
    def __init__(self, registry):
        self.registry = registry
    
    def execute_program(self, primary_id: int, secondary_id: Optional[int], 
                       comp_type: str, inputs: List[int],
                       tertiary_id: Optional[int] = None, loop_count: int = 1) -> int:
        """Execute a composed program."""
        
        if comp_type == 'none':
            return self.registry.functions[primary_id](inputs)
        
        elif comp_type == 'sequential':
            # Check if this is a LOOP composition
            if secondary_id == self.registry.loop_id:
                # Use the dedicated loop executor
                return self._execute_loop(primary_id, inputs, loop_count)
            else:
                # Regular sequential composition: f(g(inputs))
                intermediate = self.registry.functions[secondary_id](inputs)
                return self.registry.functions[primary_id]([intermediate])
        
        elif comp_type == 'nested':
            # f(g(x), g(y), ...) - apply g to each input separately
            transformed = [self.registry.functions[secondary_id]([inp]) for inp in inputs]
            return self.registry.functions[primary_id](transformed)
        
        elif comp_type == 'parallel':
            # combiner(f(inputs), g(inputs))
            if tertiary_id is None:
                result_f = self.registry.functions[primary_id](inputs)
                result_g = self.registry.functions[secondary_id](inputs) if secondary_id else result_f
                return result_f
            else:
                result_f = self.registry.functions[primary_id](inputs)
                result_g = self.registry.functions[secondary_id](inputs)
                return self.registry.functions[tertiary_id]([result_f, result_g])
        
        else:
            raise ValueError(f"Unknown composition type: {comp_type}")
    
    def validate_program(self, primary_id: int, secondary_id: Optional[int],
                        comp_type: str, examples: List[Tuple[List[int], Any]], 
                        tertiary_id: Optional[int] = None,
                        loop_count: int = 1) -> bool:
        """Validate a symbolic program against examples."""
        for inputs, expected_output in examples:
            try:
                result = self.execute_program(
                    primary_id, secondary_id, comp_type, inputs, 
                    tertiary_id, loop_count
                )
                if result != expected_output:
                    return False
            except (IndexError, TypeError, KeyError, Exception):
                return False
        return True
    
    def execute_parallel(self, primary_id: int, secondary_id: int, tertiary_id: int, inputs: List[int]) -> int:
        """Execute parallel composition: tertiary(primary(inputs), secondary(inputs))
        
        This allows building functions like XOR:
        XOR(a,b) = AND(OR(a,b), NOT(AND(a,b)))
        Which is: primary=OR, secondary=AND, tertiary=NOT, combiner would be AND
        
        But we need a 4th function to combine! Let's use tertiary as the combiner instead:
        XOR(a,b) = combiner(f1(a,b), f2(a,b))
        Where f1 and f2 are applied to inputs, then combined.
        
        For XOR: AND(OR(a,b), NOT(AND(a,b)))
        - primary = OR -> OR(a,b)
        - secondary = sequential(AND, NOT) -> NOT(AND(a,b))
        - tertiary = AND (combiner) -> AND(result1, result2)
        
        So parallel means: tertiary(primary(inputs), secondary(inputs))
        """
        if not self._is_arity_compatible(primary_id, len(inputs)):
            raise ValueError(f"Primary function arity incompatible with {len(inputs)} inputs")
        if not self._is_arity_compatible(secondary_id, len(inputs)):
            raise ValueError(f"Secondary function arity incompatible with {len(inputs)} inputs")
        
        # Apply primary and secondary to inputs
        result1 = self.registry.execute_function(primary_id, inputs)
        result2 = self.registry.execute_function(secondary_id, inputs)
        
        # Combine results with tertiary function
        if not self._is_arity_compatible(tertiary_id, 2):
            raise ValueError(f"Tertiary function must accept 2 inputs, got arity {self.registry.metadata[tertiary_id]['arity']}")
        
        return self.registry.execute_function(tertiary_id, [result1, result2])
    
    def _execute_loop(self, func_id: int, inputs: List[int], loop_count: int) -> int:
        """Execute a function in a loop.
        
        For MUL: LOOP(ADD, inputs=[a, b])
        Should compute: 0 + a + a + ... (b times) = a * b
        """
        # Handle encoded function ID
        if func_id < 0:
            actual_func_id = -(func_id + 1)
        else:
            actual_func_id = func_id
        
        # Get count
        if loop_count == -1:
            count = inputs[1]
        else:
            count = loop_count
        
        # Special case: if count is 0, return 0 for multiplication
        if count == 0:
            return 0
        
        # Get function metadata
        func_meta = self.registry.metadata.get(actual_func_id)
        if not func_meta:
            raise ValueError(f"Unknown function ID: {actual_func_id}")
        
        # Execute loop
        accumulator = 0  # CRITICAL: Start at 0 for multiplication pattern
        
        for _ in range(count):
            if func_meta['arity'] == 1:
                # Unary: f(accumulator)
                accumulator = self.registry.execute_function(actual_func_id, [accumulator])
            elif func_meta['arity'] == 2:
                # Binary: ADD(accumulator, inputs[0])
                # This gives us: 0 + a + a + ... = a * b
                accumulator = self.registry.execute_function(actual_func_id, [accumulator, inputs[0]])
            else:
                raise ValueError(f"LOOP only supports unary or binary functions")
        
        return accumulator