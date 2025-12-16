from symbolic import SymbolicRegistry
from agent import SymbolicAgent
import os

def main():
    print("="*60)
    print("Debug SUB Learning - Check Candidate Generation")
    print("="*60)
    
    if not os.path.exists("checkpoints/symbolic.db"):
        print("\n[Error] No saved registry found!")
        return
    
    registry = SymbolicRegistry()
    registry.load("checkpoints/symbolic.db")
    
    agent = SymbolicAgent(registry, d_model=128, max_recursion=10, input_dim=32, max_composition_depth=4)
    
    if os.path.exists("checkpoints/model.pt"):
        agent.load_checkpoint("checkpoints/model.pt")
    
    print(f"\nLoaded functions: {[m['name'] for m in registry.metadata.values()]}")
    
    # Get function IDs
    loop_id = None
    dec_id = None
    inc_id = None
    
    for fid, meta in registry.metadata.items():
        if meta['name'] == 'LOOP':
            loop_id = fid
        elif meta['name'] == 'DEC':
            dec_id = fid
        elif meta['name'] == 'INC':
            inc_id = fid
    
    print(f"\n✓ LOOP id={loop_id}, DEC id={dec_id}, INC id={inc_id}")
    
    # Simple subtraction examples
    sub_examples = [
        ([1, 1], 0),   # 1 - 1 = 0
        ([2, 1], 1),   # 2 - 1 = 1
        ([3, 1], 2),   # 3 - 1 = 2
        ([3, 2], 1),   # 3 - 2 = 1
        ([5, 2], 3),   # 5 - 2 = 3
    ]
    
    print("\n" + "="*60)
    print("Test: Does LOOP(DEC) work manually?")
    print("="*60)
    
    # Manually test if LOOP(DEC, count=b, init=a) gives SUB(a,b)
    for inputs, expected in sub_examples:
        a, b = inputs
        try:
            # LOOP expects [body_fn_id, count, init_value]
            loop_inputs = [dec_id, b, a]
            result = registry.execute_function(loop_id, loop_inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} LOOP(DEC, count={b}, init={a}) = {result} (expected {expected})")
        except Exception as e:
            print(f"  ✗ LOOP(DEC, count={b}, init={a}) = ERROR: {e}")
    
    print("\n" + "="*60)
    print("Checking if agent's candidate generator includes LOOP+DEC")
    print("="*60)
    
    # Check what the agent's _generate_candidates produces
    print("\nGenerating candidates with max_terms=2...")
    
    # We need to look at the agent's search process
    # Let's manually check if LOOP+DEC candidates are in the search space
    
    vocab_size = len(registry.metadata)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Available functions: {list(registry.metadata.keys())}")
    
    # Check if the search would generate LOOP(DEC) as a candidate
    print("\n" + "="*60)
    print("Key question: Is LOOP(DEC, ...) being generated?")
    print("="*60)
    print("\nIn _generate_candidates, for max_terms=2:")
    print("  - Should try LOOP with body_fn=DEC")
    print("  - LOOP has arity=-1 (variable), so it needs:")
    print("    [body_fn_id, count_param, init_param]")
    print("\nExpected candidate structure for SUB:")
    print(f"  composition_type: 'none' (direct LOOP usage)")
    print(f"  components: [{loop_id}, {dec_id}, 'param:1', 'param:0']")
    print(f"    where param:0 = first input (a)")
    print(f"    where param:1 = second input (b)")
    
    print("\n" + "="*60)
    print("Attempting SUB learning with VERBOSE logging")
    print("="*60)
    
    # Try to learn - we want to see what's actually being generated
    success = agent.learn_abstraction(
        "SUB_DEBUG",
        sub_examples,
        num_epochs=50,  # Fewer epochs, just to see candidates
        exploration_bonus=0.9,
        max_terms=2  # Start with just 2 terms
    )
    
    if success:
        print("\n✓ Found SUB!")
    else:
        print("\n✗ Still couldn't find SUB")
        print("\nDEBUG NEEDED:")
        print("  1. Check agent._generate_candidates() - does it generate LOOP(DEC)?")
        print("  2. Check if LOOP's variable arity is handled correctly")
        print("  3. Verify param mapping: LOOP needs [fn_id, count, init]")
        print("  4. Check if 'none' composition type includes direct LOOP usage")

if __name__ == "__main__":
    main()