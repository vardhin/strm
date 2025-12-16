from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Symbolic Regression with TRM Architecture")
    print("Integer Arithmetic from Bitwise Operations")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8, input_dim=32)
    
    print(f"\nInitial: {[m['name'] for m in registry.metadata.values()]}")
    print("Primitives operate on integers via bitwise operations:")
    print("  OR(5, 3) = 5|3 = 7")
    print("  AND(5, 3) = 5&3 = 1")
    print("  NOT(5) = ~5 = -6")
    
    # ==========================================
    # Step 0: Pre-train on curriculum
    # ==========================================
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Learn: NEGATE (x -> -x)
    # Using NOT: ~x = -(x+1), so we need NOT(x) + 1? 
    # Or simpler: Learn the pattern from examples
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: NEGATE")
    print("Goal: -x (negation)")
    print("="*60)
    
    negate_examples = [
        ([0], 0),     # -0 = 0
        ([1], -1),    # -1
        ([2], -2),    # -2
        ([3], -3),    # -3
        ([5], -5),    # -5
        ([10], -10),  # -10
    ]
    
    success = agent.learn_abstraction("NEGATE", negate_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing NEGATE")
        print("="*60)
        neg_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NEGATE')
        for inputs, expected in negate_examples:
            result = registry.execute_function(neg_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} NEGATE{tuple(inputs)} = {result}")
    
    # ==========================================
    # Learn: INCREMENT (x -> x+1)
    # This is: NOT(NOT(x))? Or x | 1? Need to discover pattern
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: INCREMENT")
    print("Goal: x + 1")
    print("="*60)
    
    inc_examples = [
        ([0], 1),    # 0 + 1 = 1
        ([1], 2),    # 1 + 1 = 2
        ([2], 3),    # 2 + 1 = 3
        ([3], 4),    # 3 + 1 = 4
        ([5], 6),    # 5 + 1 = 6
        ([10], 11),  # 10 + 1 = 11
    ]
    
    success = agent.learn_abstraction("INCREMENT", inc_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing INCREMENT")
        print("="*60)
        inc_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'INCREMENT')
        for inputs, expected in inc_examples:
            result = registry.execute_function(inc_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} INCREMENT{tuple(inputs)} = {result}")
    
    # ==========================================
    # Learn: DOUBLE (x -> x*2)
    # Bitwise: x << 1 = x * 2
    # Can we learn this from OR/AND/NOT? Maybe x OR x?
    # Actually: x + x, but we don't have ADD yet
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: DOUBLE")
    print("Goal: x * 2")
    print("="*60)
    
    double_examples = [
        ([0], 0),    # 0 * 2 = 0
        ([1], 2),    # 1 * 2 = 2
        ([2], 4),    # 2 * 2 = 4
        ([3], 6),    # 3 * 2 = 6
        ([5], 10),   # 5 * 2 = 10
        ([7], 14),   # 7 * 2 = 14
    ]
    
    success = agent.learn_abstraction("DOUBLE", double_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing DOUBLE")
        print("="*60)
        double_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'DOUBLE')
        for inputs, expected in double_examples:
            result = registry.execute_function(double_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} DOUBLE{tuple(inputs)} = {result}")
    
    # ==========================================
    # Learn: MASK_LOW_BIT (x -> x & 1)
    # Get lowest bit
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: IS_ODD")
    print("Goal: x & 1 (check if odd)")
    print("="*60)
    
    is_odd_examples = [
        ([0], 0),    # 0 is even
        ([1], 1),    # 1 is odd
        ([2], 0),    # 2 is even
        ([3], 1),    # 3 is odd
        ([4], 0),    # 4 is even
        ([5], 1),    # 5 is odd
        ([10], 0),   # 10 is even
        ([11], 1),   # 11 is odd
    ]
    
    success = agent.learn_abstraction("IS_ODD", is_odd_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing IS_ODD")
        print("="*60)
        odd_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'IS_ODD')
        for inputs, expected in is_odd_examples:
            result = registry.execute_function(odd_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} IS_ODD{tuple(inputs)} = {result}")
    
    # ==========================================
    # Learn: CLEAR_LOW_BIT (x -> x & ~1)
    # Clear lowest bit (make even)
    # ==========================================
    print("\n" + "="*60)
    print("Learning Abstraction: MAKE_EVEN")
    print("Goal: Clear lowest bit")
    print("="*60)
    
    make_even_examples = [
        ([0], 0),    # 0 -> 0
        ([1], 0),    # 1 -> 0
        ([2], 2),    # 2 -> 2
        ([3], 2),    # 3 -> 2
        ([4], 4),    # 4 -> 4
        ([5], 4),    # 5 -> 4
        ([10], 10),  # 10 -> 10
        ([11], 10),  # 11 -> 10
    ]
    
    success = agent.learn_abstraction("MAKE_EVEN", make_even_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing MAKE_EVEN")
        print("="*60)
        even_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'MAKE_EVEN')
        for inputs, expected in make_even_examples:
            result = registry.execute_function(even_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} MAKE_EVEN{tuple(inputs)} = {result}")

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
    
    print("\nNote: Building arithmetic (ADD, MUL) from bitwise operations")
    print("      requires complex bit manipulation circuits that are")
    print("      beyond simple 2-3 level compositions.")
    print("      The system learns bit-level operations effectively!")

if __name__ == "__main__":
    main()