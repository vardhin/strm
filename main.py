from symbolic import SymbolicRegistry
from agent import SymbolicAgent
import os

def main():
    print("="*60)
    print("Symbolic Regression with TRM")
    print("Learning Addition using Learned Primitives")
    print("="*60)
    
    # Check if checkpoints exist
    if not os.path.exists("checkpoints/model.pt"):
        print("\n[Error] No saved model found!")
        print("Please run curriculum_training_main.py first to train the foundation.")
        return
    
    # Load pre-trained model and registry
    print("\n" + "="*60)
    print("Loading Pre-trained Model")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    agent.load("checkpoints/model.pt", "checkpoints/registry.pkl")
    
    print(f"\nLoaded functions: {[m['name'] for m in registry.metadata.values()]}")
    
    # ==========================================
    # Learn: Addition (a + b)
    # Addition can be built using XOR and AND with bit shifts
    # For simple integers, we can try: XOR for sum, AND for carry
    # This is a simplified version - real addition needs carry propagation
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: ADD (simple version)")
    print("Goal: a + b (integer addition)")
    print("Note: This is challenging! May need multiple attempts.")
    print("="*60)
    
    # Simple addition examples (small numbers to start)
    add_examples = [
        ([0, 0], 0+0),   # 0 + 0 = 0
        ([0, 1], 0+1),   # 0 + 1 = 1
        ([1, 0], 1+0),   # 1 + 0 = 1
        ([1, 1], 1+1),   # 1 + 1 = 2
        ([2, 1], 2+1),   # 2 + 1 = 3
        ([2, 2], 2+2),   # 2 + 2 = 4
        ([3, 2], 3+2),   # 3 + 2 = 5
        ([3, 3], 3+3),   # 3 + 3 = 6
    ]
    
    print("\nTarget examples:")
    for inputs, expected in add_examples:
        print(f"  ADD{tuple(inputs)} = {expected}")
    
    print("\nAttempting to learn...")
    success = agent.learn_abstraction("ADD", add_examples, num_epochs=50)
    
    if success:
        print("\n" + "="*60)
        print("Testing ADD")
        print("="*60)
        add_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'ADD')
        
        print("\nTraining examples:")
        for inputs, expected in add_examples:
            result = registry.execute_function(add_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} ADD{tuple(inputs)} = {result} (expected {expected})")
        
        # Test on new examples
        print("\nGeneralization test:")
        test_examples = [
            ([4, 1], 5),
            ([5, 2], 7),
            ([4, 4], 8),
        ]
        for inputs, expected in test_examples:
            try:
                result = registry.execute_function(add_id, inputs)
                status = '✓' if result == expected else '✗'
                print(f"  {status} ADD{tuple(inputs)} = {result} (expected {expected})")
            except Exception as e:
                print(f"  ✗ ADD{tuple(inputs)} = ERROR")
        
        # Save updated model
        print("\n" + "="*60)
        print("Saving Updated Model")
        print("="*60)
        agent.save_checkpoint()
    
    else:
        print("\n[Failed] Could not learn ADD")
        print("\nNote: Addition is complex and may require:")
        print("  - More training examples")
        print("  - Higher max_composition_depth")
        print("  - Additional primitive operations (shifts, carry)")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    for fid, meta in registry.metadata.items():
        arity_str = f"arity={meta['arity']}" if meta['arity'] != -1 else "higher-order"
        print(f"  [{fid}] {meta['name']} ({arity_str}, layer={meta['layer']})")

if __name__ == "__main__":
    main()