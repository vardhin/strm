from typing import Optional, Any, List, Tuple
from .registry import SymbolicRegistry

class ProgramExecutor:
    """Executes symbolic programs with composition support."""
    
    def __init__(self, registry):
        self.registry = registry
    
    def execute_program(self, primary_id: int, secondary_id: Optional[int], 
                       comp_type: str, inputs: List[int], 
                       tertiary_id: Optional[int] = None,
                       loop_count: int = 1) -> Any:
        """
        Execute a symbolic program on given inputs.
        
        For LOOP: LOOP(f, n)(x) applies f to x n times.
        - If f is unary: f(f(f(...f(x)...)))
        - If f is binary and only x given: f(x, x) iteratively using result as both args
        - The input x defaults to 0 if not provided
        """
        primary_meta = self.registry.metadata[primary_id]
        
        if comp_type == 'none':
            # Single function
            if primary_meta['arity'] == 1:
                return self.registry.execute_function(primary_id, [inputs[0]])
            else:
                return self.registry.execute_function(primary_id, inputs)
        
        # Special case: LOOP composition
        # LOOP(f, n)(x) applies f(x) n times, using result as next input
        if comp_type == 'sequential' and secondary_id == self.registry.loop_id:
            if loop_count <= 0:
                return inputs[0] if len(inputs) > 0 else 0
            
            # Get initial value (defaults to 0 if not provided)
            acc = inputs[0] if len(inputs) > 0 else 0
            
            # Apply primary function loop_count times
            for _ in range(loop_count):
                if primary_meta['arity'] == 1:
                    # Unary function: f(acc)
                    acc = self.registry.execute_function(primary_id, [acc])
                elif primary_meta['arity'] == 2:
                    # Binary function: f(fixed_arg, acc)
                    # inputs[0] = initial accumulator
                    # inputs[1] = fixed argument to pass each iteration
                    if len(inputs) >= 2:
                        # MUL(a, b) = LOOP(ADD, b)([0, a])
                        # Each iteration: acc = ADD(a, acc)
                        acc = self.registry.execute_function(primary_id, [inputs[1], acc])
                    else:
                        # No fixed arg: f(acc, acc)
                        acc = self.registry.execute_function(primary_id, [acc, acc])
                else:
                    # Higher arity not yet supported in LOOP
                    return None
            
            return acc
                
        elif comp_type == 'sequential':
            # f2(f1(x, y)) - handle arity mismatches
            if primary_meta['arity'] == 1:
                result1 = self.registry.execute_function(primary_id, [inputs[0]])
            else:
                result1 = self.registry.execute_function(primary_id, inputs[:primary_meta['arity']])
            
            secondary_meta = self.registry.metadata[secondary_id]
            
            if secondary_meta['arity'] == 1:
                return self.registry.execute_function(secondary_id, [result1])
            elif secondary_meta['arity'] == 2:
                if len(inputs) >= 2:
                    return self.registry.execute_function(secondary_id, [result1, inputs[1]])
                else:
                    return self.registry.execute_function(secondary_id, [result1, inputs[0]])
            else:
                return None
                
        elif comp_type == 'nested':
            # f1(x, f2(x, y))
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
                if primary_meta['arity'] == 2:
                    return self.registry.execute_function(primary_id, [inputs[0], result2])
                else:
                    return self.registry.execute_function(primary_id, [result2])
            elif secondary_meta['arity'] == 2 and len(inputs) >= 2:
                result2 = self.registry.execute_function(secondary_id, inputs[:2])
                if primary_meta['arity'] == 2:
                    return self.registry.execute_function(primary_id, [inputs[0], result2])
                else:
                    return self.registry.execute_function(primary_id, [result2])
            else:
                return None
                
        elif comp_type == 'parallel':
            # f_tertiary(f_primary(x, y), f_secondary(x, y))
            primary_meta = self.registry.metadata[primary_id]
            secondary_meta = self.registry.metadata[secondary_id]
            
            if primary_meta['arity'] == 1:
                result1 = self.registry.execute_function(primary_id, [inputs[0]])
            else:
                result1 = self.registry.execute_function(primary_id, inputs[:primary_meta['arity']])
                
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
            else:
                result2 = self.registry.execute_function(secondary_id, inputs[:secondary_meta['arity']])
            
            if tertiary_id is not None:
                tertiary_meta = self.registry.metadata[tertiary_id]
                if tertiary_meta['arity'] == 2:
                    return self.registry.execute_function(tertiary_id, [result1, result2])
                elif tertiary_meta['arity'] == 1:
                    return self.registry.execute_function(tertiary_id, [result1])
            
            # Default: AND as combiner
            if isinstance(result1, int) and isinstance(result2, int):
                return result1 & result2
            else:
                return (result1, result2)
            
        return None
    
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