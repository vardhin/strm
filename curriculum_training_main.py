from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Curriculum Training: Building Complete Foundation")
    print("Learning Logic, Comparisons, and Arithmetic")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32, max_composition_depth=3)
    
    print(f"\nInitial primitives: {[m['name'] for m in registry.metadata.values()]}")
    print("\nPrimitive Categories:")
    print("  Bitwise: OR, AND, NOT")
    print("  Arithmetic: INC, DEC")
    print("  Comparison: LT, LTE, GT, GTE, EQ, NEQ")
    print("  Conditional: COND")
    print("  Identity: CONST")
    print("  Iteration: LOOP, WHILE, ACCUM")
    
    # ==========================================
    # Step 1: Pre-train on curriculum (primitives only)
    # ==========================================
    print("\n" + "="*60)
    print("Step 1: Curriculum Training on All Primitives")
    print("="*60)
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Step 2: Learn Logic Functions
    # ==========================================
    print("\n" + "="*60)
    print("Step 2: Learning Logic Functions")
    print("="*60)
    
    # 2a. Learn NAND
    print("\n[2a] Learning NAND (NOT-AND)...")
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
        print("\n✓ NAND learned successfully!")
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        
        print("\nVerifying NAND:")
        for inputs, expected in nand_examples[:4]:
            result = registry.execute_function(nand_id, inputs)
            status = "✓" if result == expected else "✗"
            print(f"  {status} NAND({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # 2b. Learn XOR (harder - needs deeper composition)
    print("\n[2b] Learning XOR (Exclusive OR)...")
    xor_examples = [
        ([0, 0], 0 ^ 0),
        ([0, 1], 0 ^ 1),
        ([1, 0], 1 ^ 0),
        ([1, 1], 1 ^ 1),
        ([5, 3], 5 ^ 3),
        ([7, 2], 7 ^ 2),
        ([15, 10], 15 ^ 10),
    ]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=50, exploration_bonus=0.7, max_terms=5)
    
    if success:
        print("\n✓ XOR learned successfully!")
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        
        print("\nVerifying XOR:")
        for inputs, expected in xor_examples[:4]:
            result = registry.execute_function(xor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} XOR({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # 2c. Learn NXOR (XNOR - equivalence)
    print("\n[2c] Learning NXOR (NOT-XOR, equivalence)...")
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
        print("\n✓ NXOR learned successfully!")
        nxor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NXOR')
        
        print("\nVerifying NXOR:")
        for inputs, expected in nxor_examples[:4]:
            result = registry.execute_function(nxor_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} NXOR({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 3: Learn Arithmetic Functions
    # ==========================================
    print("\n" + "="*60)
    print("Step 3: Learning Arithmetic Functions")
    print("="*60)
    
    # 3a. Learn ADD
    print("\n[3a] Learning ADD (Addition)...")
    add_examples = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 2),
        ([2, 3], 5),
        ([4, 2], 6),
        ([5, 5], 10),
        ([7, 3], 10),
    ]
    
    success = agent.learn_abstraction("ADD", add_examples, num_epochs=40)
    
    if success:
        print("\n✓ ADD learned successfully!")
        add_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'ADD')
        
        print("\nVerifying ADD:")
        for inputs, expected in add_examples[:6]:
            result = registry.execute_function(add_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} ADD({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # 3b. Learn SUB
    print("\n[3b] Learning SUB (Subtraction)...")
    sub_examples = [
        ([5, 2], 3),
        ([10, 3], 7),
        ([7, 7], 0),
        ([8, 4], 4),
        ([15, 5], 10),
        ([20, 8], 12),
    ]
    
    success = agent.learn_abstraction("SUB", sub_examples, num_epochs=40)
    
    if success:
        print("\n✓ SUB learned successfully!")
        sub_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'SUB')
        
        print("\nVerifying SUB:")
        for inputs, expected in sub_examples[:4]:
            result = registry.execute_function(sub_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} SUB({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # 3c. Learn MUL
    print("\n[3c] Learning MUL (Multiplication)...")
    mul_examples = [
        ([0, 0], 0),
        ([0, 5], 0),
        ([1, 5], 5),
        ([2, 3], 6),
        ([3, 4], 12),
        ([4, 5], 20),
        ([5, 5], 25),
    ]
    
    success = agent.learn_abstraction("MUL", mul_examples, num_epochs=50)
    
    if success:
        print("\n✓ MUL learned successfully!")
        mul_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'MUL')
        
        print("\nVerifying MUL:")
        for inputs, expected in mul_examples[:6]:
            result = registry.execute_function(mul_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} MUL({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 4: Learn Conditional Functions
    # ==========================================
    print("\n" + "="*60)
    print("Step 4: Learning Conditional Functions")
    print("="*60)
    
    # 4a. Learn ABS (absolute value)
    print("\n[4a] Learning ABS (Absolute Value)...")
    # Note: For non-negative integers, ABS is just identity
    # We'll use COND(LT(x, 0), NEG(x), x) but since we don't have NEG yet,
    # let's skip ABS for now or define it for positive numbers only
    
    # 4b. Learn MIN (minimum of two numbers)
    print("\n[4b] Learning MIN (Minimum)...")
    min_examples = [
        ([3, 5], 3),
        ([5, 3], 3),
        ([2, 2], 2),
        ([10, 7], 7),
        ([0, 5], 0),
        ([8, 8], 8),
    ]
    
    success = agent.learn_abstraction("MIN", min_examples, num_epochs=40)
    
    if success:
        print("\n✓ MIN learned successfully!")
        min_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'MIN')
        
        print("\nVerifying MIN:")
        for inputs, expected in min_examples[:4]:
            result = registry.execute_function(min_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} MIN({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # 4c. Learn MAX (maximum of two numbers)
    print("\n[4c] Learning MAX (Maximum)...")
    max_examples = [
        ([3, 5], 5),
        ([5, 3], 5),
        ([2, 2], 2),
        ([10, 7], 10),
        ([0, 5], 5),
        ([8, 8], 8),
    ]
    
    success = agent.learn_abstraction("MAX", max_examples, num_epochs=40)
    
    if success:
        print("\n✓ MAX learned successfully!")
        max_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'MAX')
        
        print("\nVerifying MAX:")
        for inputs, expected in max_examples[:4]:
            result = registry.execute_function(max_id, inputs)
            status = '✓' if result == expected else '✗'
            print(f"  {status} MAX({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # ==========================================
    # Step 5: Save everything
    # ==========================================
    print("\n" + "="*60)
    print("Step 5: Saving Model and Registry")
    print("="*60)
    
    agent.save_checkpoint()
    registry.save()
    
    print("\n" + "="*60)
    print("Complete Curriculum Training Finished!")
    print("="*60)
    print(f"\nLearned functions: {[m['name'] for m in registry.metadata.values()]}")
    
    print("\n" + "="*60)
    print("Next Steps - Ready to Learn:")
    print("="*60)
    print("\n1. Advanced Arithmetic:")
    print("   - DIV(a, b) = integer division (needs WHILE/ACCUM)")
    print("   - MOD(a, b) = modulo (remainder)")
    print("   - POW(a, b) = exponentiation")
    
    print("\n2. Mathematical Functions:")
    print("   - FACTORIAL(n)")
    print("   - FIBONACCI(n)")
    print("   - GCD(a, b) - greatest common divisor")
    
    print("\n3. List Operations:")
    print("   - SUM(list) - sum of elements")
    print("   - MAX_LIST(list) - maximum element")
    print("   - SORT(list) - sorting")
    
    print("\n4. Physics Equations (with CONST):")
    print("   - KE(m, v) = 0.5 * m * v^2")
    print("   - F_gravity(m1, m2, r) = G * m1 * m2 / r^2")
    
    print("\n" + "="*60)
    print("Database Summary:")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()