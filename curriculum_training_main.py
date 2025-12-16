from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Curriculum Training: Building Foundation")
    print("Learning NAND, XOR, NXOR from Basic Operations")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    
    print(f"\nInitial primitives: {[m['name'] for m in registry.metadata.values()]}")
    print("Primitives operate on integers:")
    print("  Bitwise: OR(5, 3) = 5|3 = 7")
    print("  Bitwise: AND(5, 3) = 5&3 = 1")
    print("  Bitwise: NOT(5) = ~5 = -6")
    print("  Arithmetic: INC(5) = 5+1 = 6")
    print("  Arithmetic: DEC(5) = 5-1 = 4")
    
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
        
        print("\n  Testing NAND composition:")
        for inputs, expected in nand_examples[:4]:
            result = registry.execute_function(nand_id, inputs)
            status = "✓" if result == expected else "✗"
            print(f"    {status} NAND({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 3: Learn XOR (harder - needs deeper composition)
    # ==========================================
    print("\n" + "="*60)
    print("Step 3: Learning XOR")
    print("Goal: x ^ y (exclusive OR)")
    print("="*60)
    
    xor_examples = [
        ([0, 0], 0 ^ 0),
        ([0, 1], 0 ^ 1),
        ([1, 0], 1 ^ 0),
        ([1, 1], 1 ^ 1),
        ([5, 3], 5 ^ 3),
        ([7, 2], 7 ^ 2),
    ]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=50, exploration_bonus=0.7, max_terms=5)
    
    if success:
        print("\n  ✓ XOR learned successfully!")
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        
        print("\n  Testing XOR composition:")
        for inputs, expected in xor_examples[:4]:
            result = registry.execute_function(xor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"    {status} XOR({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 4: Learn NXOR (XNOR - equivalence)
    # ==========================================
    print("\n" + "="*60)
    print("Step 4: Learning NXOR")
    print("Goal: NOT(x XOR y) - equivalence operator")
    print("="*60)
    
    nxor_examples = [
        ([0, 0], ~(0^0)),
        ([0, 1], ~(0^1)),
        ([1, 0], ~(1^0)),
        ([1, 1], ~(1^1)),
        ([3, 5], ~(3^5)),
        ([7, 3], ~(7^3)),
        ([15, 10], ~(15^10)),
    ]
    
    success = agent.learn_abstraction("NXOR", nxor_examples, num_epochs=30)
    
    if success:
        print("\n  ✓ NXOR learned successfully!")
        nxor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NXOR')
        
        print("\n  Testing NXOR composition:")
        for inputs, expected in nxor_examples[:4]:
            result = registry.execute_function(nxor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"    {status} NXOR({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 5: Save everything
    # ==========================================
    print("\n" + "="*60)
    print("Step 5: Saving Model and Registry")
    print("="*60)
    
    agent.save_checkpoint()
    
    print("\n" + "="*60)
    print("Curriculum Training Complete!")
    print("="*60)
    print(f"\nLearned functions: {[m['name'] for m in registry.metadata.values()]}")
    print("\nYou can now run test_add.py to learn arithmetic!")
    print("\nWith INC and DEC primitives, you can now learn:")
    print("  - ADD(a,b) = LOOP(INC, b) applied to a")
    print("  - SUB(a,b) = LOOP(DEC, b) applied to a")
    print("  - MUL(a,b) = LOOP(ADD(a), b) applied to 0")
    print("  - And eventually... FACTORIAL!")

if __name__ == "__main__":
    main()