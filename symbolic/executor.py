from typing import Optional, Any, List, Tuple
from .registry import SymbolicRegistry

class ProgramExecutor:
    """Handles execution and validation of symbolic programs."""
    
    def __init__(self, registry: SymbolicRegistry):
        self.registry = registry
    
    def execute_program(self, primary_id: int, secondary_id: Optional[int], 
                       comp_type: str, inputs: List[int], 
                       tertiary_id: Optional[int] = None) -> Any:
        """Execute a symbolic program on given inputs (supports 3-function composition)."""
        primary_meta = self.registry.metadata[primary_id]
        
        if comp_type == 'none':
            # Single function
            if primary_meta['arity'] == 1:
                return self.registry.execute_function(primary_id, [inputs[0]])
            else:
                return self.registry.execute_function(primary_id, inputs)
                
        elif comp_type == 'sequential':
            # f2(f1(x, y))
            result1 = self.registry.execute_function(primary_id, inputs)
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                return self.registry.execute_function(secondary_id, [result1])
            else:
                return self.registry.execute_function(secondary_id, [result1] + inputs[1:])
                
        elif comp_type == 'nested':
            # f1(x, f2(x, y))
            secondary_meta = self.registry.metadata[secondary_id]
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            elif secondary_meta['arity'] == 2 and len(inputs) >= 2:
                result2 = self.registry.execute_function(secondary_id, inputs)
                return self.registry.execute_function(primary_id, [inputs[0], result2])
            else:
                return None
                
        elif comp_type == 'parallel':
            # f_tertiary(f_primary(x, y), f_secondary(x, y))
            primary_meta = self.registry.metadata[primary_id]
            secondary_meta = self.registry.metadata[secondary_id]
            
            # Execute primary and secondary with correct arity
            if primary_meta['arity'] == 1:
                result1 = self.registry.execute_function(primary_id, [inputs[0]])
            else:
                result1 = self.registry.execute_function(primary_id, inputs[:primary_meta['arity']])
                
            if secondary_meta['arity'] == 1:
                result2 = self.registry.execute_function(secondary_id, [inputs[0]])
            else:
                result2 = self.registry.execute_function(secondary_id, inputs[:secondary_meta['arity']])
            
            # If tertiary_id specified, use it as combiner
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
                        tertiary_id: Optional[int] = None) -> bool:
        """Validate a symbolic program against examples."""
        for inputs, expected_output in examples:
            try:
                result = self.execute_program(primary_id, secondary_id, comp_type, inputs, tertiary_id)
                if result != expected_output:
                    return False
            except (IndexError, TypeError, KeyError, Exception):
                return False
        return True