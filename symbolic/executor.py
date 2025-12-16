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
            # f(g(inputs))
            if secondary_id == self.registry.loop_id and loop_count == -1:
                # Dynamic loop: apply f multiple times based on second input
                result = inputs[0]
                times = inputs[1]
                for _ in range(times):
                    result = self.registry.functions[primary_id]([result])
                return result
            elif secondary_id == self.registry.loop_id:
                # Fixed loop
                result = inputs[0]
                for _ in range(loop_count):
                    result = self.registry.functions[primary_id]([result])
                return result
            else:
                # Regular sequential composition
                intermediate = self.registry.functions[secondary_id](inputs)
                return self.registry.functions[primary_id]([intermediate])
        
        elif comp_type == 'nested':
            # f(g(x), g(y), ...) - apply g to each input separately
            transformed = [self.registry.functions[secondary_id]([inp]) for inp in inputs]
            return self.registry.functions[primary_id](transformed)
        
        elif comp_type == 'parallel':
            # NEW: combiner(f(inputs), g(inputs))
            # Compute both functions on the same inputs, then combine
            if tertiary_id is None:
                # Default: just return first result (shouldn't happen)
                result_f = self.registry.functions[primary_id](inputs)
                result_g = self.registry.functions[secondary_id](inputs) if secondary_id else result_f
                return result_f
            else:
                # Apply both functions, then combine with tertiary
                result_f = self.registry.functions[primary_id](inputs)
                result_g = self.registry.functions[secondary_id](inputs)
                
                # Combiner takes both results as inputs
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