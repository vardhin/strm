from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Curriculum Training: Building Foundation")
    print("Learning NAND and XOR from Basic Logic Gates")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    
    print(f"\nInitial primitives: {[m['name'] for m in registry.metadata.values()]}")
    print("Primitives operate on integers via bitwise operations:")
    print("  OR(5, 3) = 5|3 = 7")
    print("  AND(5, 3) = 5&3 = 1")
    print("  NOT(5) = ~5 = -6")
    
    # ==========================================
    # Step 1: Pre-train on curriculum (primitives only)
    # ==========================================
    print("\n" + "="*60)
    print("Step 1: Curriculum Training on Primitives")
    print("="*60)
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Step 2: Learn NAND
    # ==========================================
    print("\n" + "="*60)
    print("Step 2: Learning NAND")
    print("Goal: NOT(x AND y)")
    print("="*60)
    
    nand_examples = [
        ([0, 0], ~(0&0)),
        ([0, 1], ~(0&1)),
        ([1, 0], ~(1&0)),
        ([1, 1], ~(1&1)),
        ([3, 5], ~(3&5)),
        ([7, 3], ~(7&3)),
        ([15, 10], ~(15&10)),
    ]
    
    success = agent.learn_abstraction("NAND", nand_examples, num_epochs=30)
    
    if success:
        print("\n  ✓ NAND learned successfully!")
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        for inputs, expected in nand_examples[:4]:
            result = registry.execute_function(nand_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"    {status} NAND{tuple(inputs)} = {result}")
    
    # Test NAND
    print("\n  Testing NAND composition:")
    nand_examples = [
        ([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0),
        ([3, 5], ~(3 & 5)), ([7, 3], ~(7 & 3)), ([15, 10], ~(15 & 10))
    ]
    
    for inputs, expected in nand_examples[:4]:  # Test first 4
        result = registry.execute_function(nand_id, inputs)  # ✓ Correct - inputs is already a list
        status = "✓" if result == expected else "✗"
        print(f"    {status} NAND({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 3: Learn XOR
    # ==========================================
    print("\n" + "="*60)
    print("Step 3: Learning XOR")
    print("Goal: x ^ y (uses NAND)")
    print("="*60)
    
    xor_examples = [
        ([0, 0], 0^0),
        ([0, 1], 0^1),
        ([1, 0], 1^0),
        ([1, 1], 1^1),
        ([3, 5], 3^5),
        ([7, 3], 7^3),
        ([15, 10], 15^10),
    ]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=30)
    
    if success:
        print("\n  ✓ XOR learned successfully!")
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        for inputs, expected in xor_examples[:4]:
            result = registry.execute_function(xor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"    {status} XOR{tuple(inputs)} = {result}")
    
    # ==========================================
    # Step 4: Save everything
    # ==========================================
    print("\n" + "="*60)
    print("Step 4: Saving Model and Registry")
    print("="*60)
    
    agent.save_checkpoint()
    
    print("\n" + "="*60)
    print("Curriculum Training Complete!")
    print("="*60)
    print(f"\nLearned functions: {[m['name'] for m in registry.metadata.values()]}")
    print("\nYou can now run main.py to use these learned functions!")

if __name__ == "__main__":
    main()