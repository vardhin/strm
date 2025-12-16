from typing import Dict, Callable, Any, List

class SymbolicRegistry:
    """
    Manages the available functions (primitives + learned abstractions).
    Acts as the persistent storage for abstraction layers.
    """
    def __init__(self):
        self.functions = {}
        self.metadata = {}
        self.loop_id = None
        self._next_id = 0
        
        # Register primitive bitwise operations on integers
        self.register("OR", lambda x, y: x | y, arity=2)      # Bitwise OR
        self.register("AND", lambda x, y: x & y, arity=2)     # Bitwise AND
        self.register("NOT", lambda x: ~x, arity=1)           # Bitwise NOT (complement)
        
        # Register LOOP meta-function
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

    def execute_function(self, func_id: int, args: List[Any]) -> Any:
        """Execute a single function by ID."""
        func = self.functions[func_id]
        return func(*args)