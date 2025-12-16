from symbolic import SymbolicRegistry
from agent import SymbolicAgent
import os

def main():
    print("="*60)
    print("Symbolic Regression with TRM")
    print("Learning Addition using Learned Primitives")
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
    # Load registry from database instead of pickle
    registry.load("checkpoints/symbolic.db")
    
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    
    # Check if model checkpoint exists
    if os.path.exists("checkpoints/model.pt"):
        agent.load_checkpoint("checkpoints/model.pt")
    else:
        print("\n[Warning] No model checkpoint found, starting fresh")
    
    print(f"\nLoaded functions: {[m['name'] for m in registry.metadata.values()]}")
    
    # ==========================================
    # Learn: Addition (a + b)
    # Expected: LOOP(INC, dynamic) where loop count comes from input[1]
    # ADD(2, 3) = apply INC 3 times starting from 2
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: ADD")
    print("Goal: a + b (integer addition)")
    print("Expected: LOOP(INC, count=b) starting from a")
    print("="*60)
    
    # Simple addition examples (small numbers to start)
    add_examples = [
        ([0, 0], 0),   # 0 + 0 = 0 (apply INC 0 times to 0)
        ([1, 0], 1),   # 1 + 0 = 1 (apply INC 0 times to 1)
        ([0, 1], 1),   # 0 + 1 = 1 (apply INC 1 time to 0)
        ([1, 1], 2),   # 1 + 1 = 2 (apply INC 1 time to 1)
        ([2, 1], 3),   # 2 + 1 = 3 (apply INC 1 time to 2)
        ([1, 2], 3),   # 1 + 2 = 3 (apply INC 2 times to 1)
        ([2, 2], 4),   # 2 + 2 = 4 (apply INC 2 times to 2)
        ([3, 2], 5),   # 3 + 2 = 5 (apply INC 2 times to 3)
        ([2, 3], 5),   # 2 + 3 = 5 (apply INC 3 times to 2)
        ([5, 3], 8),   # 5 + 3 = 8 (apply INC 3 times to 5)
    ]
    
    print("\nTarget examples:")
    for inputs, expected in add_examples:
        print(f"  ADD({inputs[0]}, {inputs[1]}) = {expected}")
    
    print("\nAttempting to learn with pure TRM (no exhaustive search)...")
    
    # Try with high exploration to discover LOOP(INC, -1)
    success = agent.learn_abstraction(
        "ADD", 
        add_examples, 
        num_epochs=100,  # More epochs for harder task
        exploration_bonus=0.8  # High exploration
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ ADD learned successfully!")
        print("="*60)
        add_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'ADD')
        
        print("\nTraining examples:")
        for inputs, expected in add_examples[:5]:
            result = registry.execute_function(add_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} ADD({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
        
        # Test on new examples
        print("\nGeneralization test:")
        test_examples = [
            ([4, 1], 5),
            ([5, 2], 7),
            ([10, 5], 15),
        ]
        for inputs, expected in test_examples:
            try:
                result = registry.execute_function(add_id, inputs)
                status = '✓' if result == expected else '✗'
                print(f"  {status} ADD({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
            except Exception as e:
                print(f"  ✗ ADD({inputs[0]}, {inputs[1]}) = ERROR: {e}")
        
        # Save updated model
        agent.save_checkpoint()
        print("\n✓ Model saved!")
    
    else:
        print("\n✗ Could not learn ADD")
        print("\nDebugging suggestions:")
        print("  1. Check if LOOP can use input[1] as count (need loop_count=-1 mode)")
        print("  2. Increase exploration_bonus to 0.9")
        print("  3. Add more curriculum examples with different counts")
        print("  4. Train for more epochs (try 200+)")
    
    # Show final registry
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()