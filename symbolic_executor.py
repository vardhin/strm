"""
Executes symbolic programs with full tracing
"""
from typing import List, Optional, Any, Tuple, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class ExecutionStep:
    """Records one step of execution"""
    step_num: int
    operation: str
    inputs: List[Any]
    output: Any
    stack_state: List[Any]

class SymbolicExecutor:
    """
    Executes programs with full visibility
    """
    def __init__(self, primitives_manager, verbose=False):
        self.prims = primitives_manager
        self.verbose = verbose
        self.execution_trace = []
    
    def execute_program(self, program_tokens: List[str], 
                       inputs: Tuple[int, ...],
                       trace=False, max_steps=100) -> Optional[Tuple[int, ...]]:
        """
        Execute program with AUTO-FILL for missing stack values
        """
        self.execution_trace = []
        stack = []
        step = 0
        n_inputs = len(inputs)  # Store number of valid inputs
        
        if trace:
            print(f"\n{'='*60}")
            print(f"Executing: {' '.join(program_tokens)}")
            print(f"Inputs: {inputs}")
            print(f"{'='*60}")
        
        for token in program_tokens:
            if step >= max_steps:
                if trace:
                    print(f"⚠️ Max steps ({max_steps}) exceeded")
                return None
            
            step += 1
            
            # Skip control tokens
            if token in {'START', 'DONE', 'PAD'}:
                continue
            
            if token not in self.prims.primitives:
                if trace:
                    print(f"❌ Unknown token: {token}")
                return None
            
            prim = self.prims.primitives[token]
            
            # STRICT INPUT VALIDATION: Check if LOAD_X is within bounds
            if token.startswith('LOAD_'):
                try:
                    idx = int(token.split('_')[1])
                    if idx >= n_inputs:
                        if trace:
                            print(f"❌ {token}: Index {idx} out of bounds (only {n_inputs} inputs)")
                        return None
                except (ValueError, IndexError):
                    if trace:
                        print(f"❌ Invalid LOAD token: {token}")
                    return None
            
            # LOAD operations
            if token.startswith('LOAD_'):
                arg_idx = int(token.split('_')[1])
                if arg_idx < len(inputs):
                    value = inputs[arg_idx]
                else:
                    # Out of bounds - use last available input
                    value = inputs[-1] if inputs else 0
                stack.append(value)
                
                if trace:
                    self.execution_trace.append(ExecutionStep(
                        step_num=step,
                        operation=f"arg[{arg_idx}]",
                        inputs=[],
                        output=value,
                        stack_state=stack.copy()
                    ))
                continue
            
            # CONST operations
            if token.startswith('CONST_'):
                const_val = int(token.split('_')[1])
                stack.append(const_val)
                
                if trace:
                    self.execution_trace.append(ExecutionStep(
                        step_num=step,
                        operation=str(const_val),
                        inputs=[],
                        output=const_val,
                        stack_state=stack.copy()
                    ))
                continue
            
            # Get primitive
            prim = self.prims.primitives.get(token)
            if prim is None:
                if self.verbose:
                    print(f"Unknown operation: {token}")
                return None
            
            # Execute with AUTO-FILL based on token name
            binary_ops = {'ADD', 'SUB', 'MUL', 'DIV'}
            unary_ops = {'NEG', 'SQUARE'}
            
            if token in binary_ops:
                # Binary operation
                if len(stack) >= 2:
                    b = stack.pop()
                    a = stack.pop()
                elif len(stack) == 1:
                    # Only 1 value - use it twice!
                    a = stack[0]
                    b = stack[0]
                    stack.clear()
                else:
                    # Empty stack - use first input or 0
                    a = inputs[0] if inputs else 0
                    b = a
                
                try:
                    if prim.function is None:
                        if self.verbose:
                            print(f"Primitive {token} has no callable function")
                        return None
                    result = prim.function(a, b)
                    # Handle division by zero and invalid results
                    if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):
                        return None
                except Exception as e:
                    if self.verbose:
                        print(f"Error executing {token}({a}, {b}): {e}")
                    return None
                
                if trace:
                    self.execution_trace.append(ExecutionStep(
                        step_num=step,
                        operation=token,
                        inputs=[a, b],
                        output=result,
                        stack_state=stack.copy() + [result]
                    ))
                
            elif token in unary_ops:
                # Unary operation
                if len(stack) >= 1:
                    a = stack.pop()
                else:
                    # Empty stack - use first input or 0
                    a = inputs[0] if inputs else 0
                
                try:
                    # Use .function attribute consistently
                    if prim.function is None:
                        if self.verbose:
                            print(f"Primitive {token} has no callable function")
                        return None
                    result = prim.function(a)
                    # Handle invalid results
                    if not isinstance(result, (int, float)) or np.isnan(result) or np.isinf(result):
                        return None
                except Exception as e:
                    if self.verbose:
                        print(f"Error executing {token}({a}): {e}")
                    return None
                
                if trace:
                    self.execution_trace.append(ExecutionStep(
                        step_num=step,
                        operation=token,
                        inputs=[a],
                        output=result,
                        stack_state=stack.copy() + [result]
                    ))
            
            else:
                # Unknown operation type
                if self.verbose:
                    print(f"Unknown operation type for: {token}")
                return None
            
            stack.append(result)
        
        # Return final stack as output
        if len(stack) > 0:
            return tuple(stack)
        else:
            return None
    
    def is_correct(self, program_tokens: List[str], 
                   inputs: Tuple[int, ...], 
                   targets: Tuple[int, ...]) -> bool:
        """Check if program produces correct output"""
        result = self.execute_program(program_tokens, inputs, trace=False)
        if result is None:
            return False
        return result == targets
    
    def print_execution_trace(self, trace: List[ExecutionStep]):
        """Pretty print execution trace"""
        print("\n" + "="*80)
        print("EXECUTION TRACE")
        print("="*80)
        for step in trace:
            print(f"\nStep {step.step_num}: {step.operation}")
            print(f"  Inputs:  {step.inputs}")
            print(f"  Output:  {step.output}")
            print(f"  Stack:   {step.stack_state}")
        print("="*80 + "\n")