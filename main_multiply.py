from symbolic import SymbolicRegistry
from agent import SymbolicAgent
import os

def main():
    print("="*60)
    print("Symbolic Regression with TRM")
    print("Learning Multiplication using ADD")
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
    
    agent = SymbolicAgent(registry, d_model=128, max_recursion=10, input_dim=32, max_composition_depth=3)
    
    # Check if model checkpoint exists
    if os.path.exists("checkpoints/model.pt"):
        agent.load_checkpoint("checkpoints/model.pt")
    else:
        print("\n[Warning] No model checkpoint found, starting fresh")
    
    print(f"\nLoaded functions: {[m['name'] for m in registry.metadata.values()]}")
    
    # Verify ADD exists
    add_id = None
    for fid, meta in registry.metadata.items():
        if meta['name'] == 'ADD':
            add_id = fid
            break
    
    if add_id is None:
        print("\n[Error] ADD function not found!")
        print("Please run main.py first to learn ADD.")
        return
    
    print(f"\n✓ Found ADD (id={add_id})")
    
    # Test ADD first
    print("\nVerifying ADD works:")
    test_cases = [([2, 3], 5), ([4, 2], 6), ([5, 0], 5)]
    for inputs, expected in test_cases:
        result = registry.execute_function(add_id, inputs)
        status = '✓' if result == expected else '✗'
        print(f"  {status} ADD({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Learn: Multiplication (a * b)
    # Expected: LOOP(ADD(a), b) starting from 0
    # MUL(3, 4) = ADD(3) applied 4 times to 0 = 3+3+3+3 = 12
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: MUL")
    print("Goal: a * b (integer multiplication)")
    print("Expected: LOOP(ADD(a), count=b) starting from 0")
    print("="*60)
    
    # Multiplication examples (small numbers)
    mul_examples = [
        ([0, 0], 0),   # 0 * 0 = 0
        ([0, 1], 0),   # 0 * 1 = 0
        ([1, 0], 0),   # 1 * 0 = 0
        ([1, 1], 1),   # 1 * 1 = 1
        ([2, 1], 2),   # 2 * 1 = 2
        ([1, 2], 2),   # 1 * 2 = 2
        ([2, 2], 4),   # 2 * 2 = 4
        ([3, 2], 6),   # 3 * 2 = 6
        ([2, 3], 6),   # 2 * 3 = 6
        ([3, 3], 9),   # 3 * 3 = 9
        ([4, 2], 8),   # 4 * 2 = 8
        ([2, 4], 8),   # 2 * 4 = 8
    ]
    
    print("\nTarget examples:")
    for inputs, expected in mul_examples:
        print(f"  MUL({inputs[0]}, {inputs[1]}) = {expected}")
    
    print("\nAttempting to learn with pure TRM...")
    print("Note: This requires composing LOOP with the learned ADD function!")
    
    # Try to learn MUL with increased depth to use learned functions
    success = agent.learn_abstraction(
        "MUL", 
        mul_examples, 
        num_epochs=150,  # More epochs for harder composition
        exploration_bonus=0.8,  # High exploration
        max_terms=3  # Allow deeper compositions to use ADD
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ MUL learned successfully!")
        print("="*60)
        mul_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'MUL')
        
        print("\nTraining examples:")
        for inputs, expected in mul_examples[:6]:
            result = registry.execute_function(mul_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} MUL({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
        
        # Test on new examples
        print("\nGeneralization test:")
        test_examples = [
            ([5, 2], 10),
            ([3, 4], 12),
            ([6, 3], 18),
        ]
        for inputs, expected in test_examples:
            try:
                result = registry.execute_function(mul_id, inputs)
                status = '✓' if result == expected else '✗'
                print(f"  {status} MUL({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
            except Exception as e:
                print(f"  ✗ MUL({inputs[0]}, {inputs[1]}) = ERROR: {e}")
        
        # Save updated model
        agent.save_checkpoint()
        print("\n✓ Model saved!")
    
    else:
        print("\n✗ Could not learn MUL")
        print("\nDebugging suggestions:")
        print("  1. MUL requires LOOP(ADD(a), b) - needs to compose with learned ADD")
        print("  2. Increase max_composition_depth to 3 or 4")
        print("  3. Train for more epochs (try 200+)")
        print("  4. Add more curriculum examples with different products")
        print("  5. Check if _generate_deep_candidates tries learned functions with LOOP")
    
    # Show final registry
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()