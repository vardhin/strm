from symbolic import SymbolicRegistry
from agent import SymbolicAgent
import os

def main():
    print("="*60)
    print("Symbolic Regression with TRM")
    print("Learning Division using SUB (Conditional Iteration)")
    print("="*60)
    
    # Check if checkpoints exist
    if not os.path.exists("checkpoints/symbolic.db"):
        print("\n[Error] No saved registry found!")
        print("Please run curriculum_training_main.py first to train the foundation.")
        return
    
    # Load pre-trained model and registry
    print("\n" + "="*60)
    print("Loading Pre-trained Model")
    print("="*60)
    
    registry = SymbolicRegistry()
    registry.load("checkpoints/symbolic.db")
    
    agent = SymbolicAgent(registry, d_model=128, max_recursion=10, input_dim=32, max_composition_depth=4)
    
    # Check if model checkpoint exists
    if os.path.exists("checkpoints/model.pt"):
        agent.load_checkpoint("checkpoints/model.pt")
    else:
        print("\n[Warning] No model checkpoint found, starting fresh")
    
    print(f"\nLoaded functions: {[m['name'] for m in registry.metadata.values()]}")
    
    # Verify SUB exists (needed for division)
    sub_id = None
    for fid, meta in registry.metadata.items():
        if meta['name'] == 'SUB':
            sub_id = fid
            break
    
    if sub_id is None:
        print("\n[Error] SUB function not found!")
        print("Please ensure SUB was learned in curriculum training.")
        return
    
    print(f"\n✓ Found SUB (id={sub_id})")
    
    # Test SUB first
    print("\nVerifying SUB works:")
    test_cases = [([5, 2], 3), ([10, 3], 7), ([7, 7], 0)]
    for inputs, expected in test_cases:
        result = registry.execute_function(sub_id, inputs)
        status = '✓' if result == expected else '✗'
        print(f"  {status} SUB({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Learn: Integer Division (a // b)
    # Challenge: Requires conditional iteration!
    # DIV(12, 3) = count how many times we can subtract 3 from 12
    # 12 - 3 = 9 (1)
    # 9 - 3 = 6 (2)
    # 6 - 3 = 3 (3)
    # 3 - 3 = 0 (4) -> answer is 4
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: DIV (Integer Division)")
    print("Goal: a // b (quotient only)")
    print("Challenge: Requires CONDITIONAL iteration - unknown loop count!")
    print("Expected: Count iterations of SUB(a, b) until result <= 0")
    print("="*60)
    
    # Division examples (integer division, no remainders first)
    div_examples = [
        # Simple cases
        ([1, 1], 1),   # 1 / 1 = 1
        ([2, 1], 2),   # 2 / 1 = 2
        ([3, 1], 3),   # 3 / 1 = 3
        ([4, 1], 4),   # 4 / 1 = 4
        
        # Divide by 2
        ([2, 2], 1),   # 2 / 2 = 1
        ([4, 2], 2),   # 4 / 2 = 2
        ([6, 2], 3),   # 6 / 2 = 3
        ([8, 2], 4),   # 8 / 2 = 4
        
        # Divide by 3
        ([3, 3], 1),   # 3 / 3 = 1
        ([6, 3], 2),   # 6 / 3 = 2
        ([9, 3], 3),   # 9 / 3 = 3
        ([12, 3], 4),  # 12 / 3 = 4
        
        # Mixed
        ([10, 2], 5),  # 10 / 2 = 5
        ([15, 3], 5),  # 15 / 3 = 5
        ([20, 4], 5),  # 20 / 4 = 5
    ]
    
    print("\nTarget examples (exact division only):")
    for inputs, expected in div_examples:
        print(f"  DIV({inputs[0]}, {inputs[1]}) = {expected}")
    
    print("\n" + "="*60)
    print("CRITICAL CHALLENGE:")
    print("="*60)
    print("Division needs WHILE-loop semantics:")
    print("  result = 0")
    print("  while a >= b:")
    print("      a = a - b")
    print("      result += 1")
    print("  return result")
    print("\nCurrent TRM primitives may not support conditional iteration!")
    print("You may need to add a WHILE or REPEAT_UNTIL primitive.")
    print("="*60)
    
    print("\nAttempting to learn with current TRM...")
    print("(This will likely fail without conditional iteration primitives)")
    
    # Try to learn DIV - this will be challenging!
    success = agent.learn_abstraction(
        "DIV", 
        div_examples, 
        num_epochs=200,  # Many epochs for this hard problem
        exploration_bonus=0.9,  # Maximum exploration
        max_terms=4  # Allow deep compositions
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ DIV learned successfully! (Impressive!)")
        print("="*60)
        div_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'DIV')
        
        print("\nTraining examples:")
        for inputs, expected in div_examples[:8]:
            result = registry.execute_function(div_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} DIV({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
        
        # Test on new examples
        print("\nGeneralization test:")
        test_examples = [
            ([14, 2], 7),
            ([18, 3], 6),
            ([25, 5], 5),
        ]
        for inputs, expected in test_examples:
            try:
                result = registry.execute_function(div_id, inputs)
                status = '✓' if result == expected else '✗'
                print(f"  {status} DIV({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
            except Exception as e:
                print(f"  ✗ DIV({inputs[0]}, {inputs[1]}) = ERROR: {e}")
        
        # Save updated model
        agent.save_checkpoint()
        print("\n✓ Model saved!")
    
    else:
        print("\n" + "="*60)
        print("✗ Could not learn DIV (Expected!)")
        print("="*60)
        print("\nThis is expected because division requires:")
        print("  1. CONDITIONAL iteration (unknown loop count)")
        print("  2. State tracking (counting iterations)")
        print("  3. Termination condition (a < b)")
        print("\nSuggested next steps:")
        print("  1. Add WHILE(condition, body, state) primitive")
        print("  2. Add REPEAT_UNTIL(body, condition, init) primitive")
        print("  3. Add comparison primitives: LT, GTE, EQ")
        print("  4. Add counter/accumulator state management")
        print("\nAlternative: Try learning with remainders first")
        print("  DIV_WITH_REM(a, b) -> (quotient, remainder)")
    
    # Show final registry
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()