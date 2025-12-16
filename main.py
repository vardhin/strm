from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Symbolic Regression with TRM Architecture")
    print("Learning XOR and NAND from Basic Logic Gates")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    
    print(f"\nInitial: {[m['name'] for m in registry.metadata.values()]}")
    print("Primitives operate on integers via bitwise operations:")
    print("  OR(5, 3) = 5|3 = 7")
    print("  AND(5, 3) = 5&3 = 1")
    print("  NOT(5) = ~5 = -6")
    
    # ==========================================
    # Step 0: Pre-train on curriculum (primitives only)
    # ==========================================
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Learn: NAND (NOT(x AND y))
    # NAND is simply: NOT(A AND B)
    # This should be very easy as it's just a 2-step composition
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: NAND")
    print("Goal: NOT(x AND y)")
    print("Can be built as: NOT(AND(x, y)) - sequential composition")
    print("="*60)
    
    nand_examples = [
        ([0, 0], ~(0&0)),  # NAND(0,0) = NOT(0) = -1
        ([0, 1], ~(0&1)),  # NAND(0,1) = NOT(0) = -1
        ([1, 0], ~(1&0)),  # NAND(1,0) = NOT(0) = -1
        ([1, 1], ~(1&1)),  # NAND(1,1) = NOT(1) = -2
        ([3, 5], ~(3&5)),  # NAND(3,5) = NOT(1) = -2
        ([7, 3], ~(7&3)),  # NAND(7,3) = NOT(3) = -4
        ([15, 10], ~(15&10)),  # NAND(15,10) = NOT(10) = -11
    ]
    
    success = agent.learn_abstraction("NAND", nand_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing NAND")
        print("="*60)
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        for inputs, expected in nand_examples:
            result = registry.execute_function(nand_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} NAND{tuple(inputs)} = {result} (expected {expected})")
    
    # ==========================================
    # Learn: XOR (x ^ y)
    # Now that we have NAND, XOR can be built as:
    # XOR(a,b) = AND(OR(a,b), NAND(a,b))
    # This is a 3-function parallel composition using NAND!
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: XOR")
    print("Goal: x ^ y (exclusive or)")
    print("Can be built as: AND(OR(a,b), NAND(a,b))")
    print("This uses NAND which we just learned!")
    print("="*60)
    
    xor_examples = [
        ([0, 0], 0^0),  # 0 XOR 0 = 0
        ([0, 1], 0^1),  # 0 XOR 1 = 1
        ([1, 0], 1^0),  # 1 XOR 0 = 1
        ([1, 1], 1^1),  # 1 XOR 1 = 0
        ([3, 5], 3^5),  # 3 XOR 5 = 6
        ([7, 3], 7^3),  # 7 XOR 3 = 4
        ([15, 10], 15^10),  # 15 XOR 10 = 5
    ]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing XOR")
        print("="*60)
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        for inputs, expected in xor_examples:
            result = registry.execute_function(xor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} XOR{tuple(inputs)} = {result} (expected {expected})")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    for fid, meta in registry.metadata.items():
        arity_str = f"arity={meta['arity']}" if meta['arity'] != -1 else "higher-order"
        print(f"  [{fid}] {meta['name']} ({arity_str}, layer={meta['layer']})")
    
    print("\n" + "="*60)
    print("Abstraction Hierarchy Learned!")
    print("="*60)
    
    print("\nNote: We learned these in order:")
    print("      1. NAND = NOT(AND(x,y)) - sequential composition")
    print("      2. XOR = AND(OR(x,y), NAND(x,y)) - parallel composition using NAND")
    print("\nThis demonstrates hierarchical learning - using previously")
    print("learned abstractions to build more complex ones!")

if __name__ == "__main__":
    main()