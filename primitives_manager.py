"""
Manages primitive operations - ONLY for mathematical equations
"""
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import torch

@dataclass
class Primitive:
    """A single primitive operation"""
    name: str
    token_id: int
    n_inputs: int
    n_outputs: int
    symbolic_form: str
    function: Optional[Callable] = None  # Add this field
    is_learned: bool = False

class PrimitivesManager:
    """
    Registry of operations - MINIMAL set for equations
    """
    def __init__(self, max_inputs=5, max_outputs=3):
        self.primitives: Dict[str, Primitive] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._next_id = 0
        
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        
        # Initialize with ONLY equation primitives
        self._init_equation_primitives()
    
    def _init_equation_primitives(self):
        """Bootstrap with ONLY operations needed for equations"""
        base_primitives = [
            # Load arguments (MUST use these) - no function needed
            *[Primitive(f"LOAD_{i}", self._get_next_id(), 0, 1, f"arg[{i}]", None) 
              for i in range(self.max_inputs)],
            
            # Basic arithmetic (the ONLY operations we need)
            Primitive("ADD", self._get_next_id(), 2, 1, "a + b", lambda a, b: a + b),
            Primitive("SUB", self._get_next_id(), 2, 1, "a - b", lambda a, b: a - b),
            Primitive("MUL", self._get_next_id(), 2, 1, "a * b", lambda a, b: a * b),
            Primitive("DIV", self._get_next_id(), 2, 1, "a / b", lambda a, b: a / b if b != 0 else 0),
            
            # Unary operations (sometimes useful)
            Primitive("NEG", self._get_next_id(), 1, 1, "-x", lambda x: -x),
            Primitive("SQUARE", self._get_next_id(), 1, 1, "x * x", lambda x: x * x),
            
            # Constants (sometimes needed) - no function needed
            Primitive("CONST_0", self._get_next_id(), 0, 1, "0", None),
            Primitive("CONST_1", self._get_next_id(), 0, 1, "1", None),
            Primitive("CONST_2", self._get_next_id(), 0, 1, "2", None),
            
            # Control tokens - no function needed
            Primitive("START", self._get_next_id(), 0, 0, "<start>", None),
            Primitive("DONE", self._get_next_id(), 0, 0, "<done>", None),
            Primitive("PAD", self._get_next_id(), 0, 0, "<pad>", None),
        ]
        
        for prim in base_primitives:
            self.primitives[prim.name] = prim
            self.token_to_id[prim.name] = prim.token_id
            self.id_to_token[prim.token_id] = prim.name
    
    def _get_next_id(self) -> int:
        id = self._next_id
        self._next_id += 1
        return id
    
    def add_learned_rule(self, name: str, program: List[str], 
                        n_inputs: int, n_outputs: int, symbolic_form: str):
        """Add a newly discovered rule"""
        if name in self.primitives:
            print(f"⚠ Rule {name} already exists, skipping")
            return
        
        new_prim = Primitive(
            name=name,
            token_id=self._get_next_id(),
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            symbolic_form=symbolic_form,
            function=None,  # Learned rules don't need functions
            is_learned=True
        )
        
        self.primitives[name] = new_prim
        self.token_to_id[name] = new_prim.token_id
        self.id_to_token[new_prim.token_id] = name
        
        print(f"✓ Learned new primitive: {symbolic_form}")
    
    def get_vocab_size(self) -> int:
        return len(self.primitives)
    
    def get_primitive_names(self) -> List[str]:
        return list(self.primitives.keys())
    
    def tokens_to_program(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token.get(tid, "PAD") for tid in token_ids]
    
    def validate_program_uses_all_inputs(self, program: List[str], n_inputs: int) -> bool:
        """Check if program uses all input arguments"""
        used_inputs = set()
        for token in program:
            if token.startswith("LOAD_"):
                idx = int(token.split("_")[1])
                if idx < n_inputs:
                    used_inputs.add(idx)
        
        # Must use ALL inputs
        return len(used_inputs) == n_inputs
    
    def from_program(self, program: List[str]):
        """Convert a program (list of tokens) to a PyTorch computation graph"""
        # Initialize an empty list of nodes
        nodes = []
        
        # Map to keep track of output tensors from each primitive
        output_tensors = {}
        
        for token in program:
            if token.startswith("LOAD_"):
                # Handle input loading
                idx = int(token.split("_")[1])
                # For loading, we just use the identity function
                nodes.append(("LOAD", idx, idx))
                output_tensors[idx] = idx  # Map the output tensor
            
            elif token in {"ADD", "SUB", "MUL", "DIV"}:
                # Handle binary operations
                a = output_tensors.popitem()[1]  # Get last two inputs
                b = output_tensors.popitem()[1]
                
                if token == "ADD":
                    nodes.append(("ADD", a, b, a + b))
                    output_tensors[a + b] = a + b
                
                elif token == "SUB":
                    nodes.append(("SUB", a, b, a - b))
                    output_tensors[a - b] = a - b
                
                elif token == "MUL":
                    nodes.append(("MUL", a, b, a * b))
                    output_tensors[a * b] = a * b
                
                elif token == "DIV":
                    nodes.append(("DIV", a, b, a / b))
                    output_tensors[a / b] = a / b
            
            elif token == "NEG":
                # Handle negation
                a = output_tensors.popitem()[1]
                nodes.append(("NEG", a, -a))
                output_tensors[-a] = -a
            
            elif token == "SQUARE":
                # Handle squaring
                a = output_tensors.popitem()[1]
                nodes.append(("SQUARE", a, a * a))
                output_tensors[a * a] = a * a
            
            elif token.startswith("CONST_"):
                # Handle constants
                const_val = int(token.split("_")[1])
                nodes.append(("CONST", const_val, const_val))
                output_tensors[const_val] = const_val
            
            elif token == "START":
                # No operation needed for START
                continue
            
            elif token == "DONE":
                # No operation needed for DONE
                continue
            
            elif token == "PAD":
                # No operation needed for PAD
                continue
            
            else:
                raise ValueError(f"Unknown token: {token}")
        
        # The final output should be the last computed tensor
        final_output = output_tensors.popitem()[1]
        
        return nodes, final_output
