from symbolic import SymbolicRegistry
from agent import SymbolicAgent

def main():
    print("="*60)
    print("Symbolic Regression with TRM Architecture")
    print("="*60)
    
    registry = SymbolicRegistry()
    agent = SymbolicAgent(registry, d_model=128, max_recursion=8)
    
    print(f"\nInitial: {[m['name'] for m in registry.metadata.values()]}")
    
    # ==========================================
    # Step 0: Pre-train on curriculum
    # ==========================================
    agent.train_on_curriculum(num_epochs=20)
    
    # ==========================================
    # Layer 1: Learn NAND
    # ==========================================
    nand_examples = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    success = agent.learn_abstraction("NAND", nand_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing NAND")
        print("="*60)
        nand_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NAND')
        for inputs, expected in nand_examples:
            result = registry.execute_function(nand_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} NAND{tuple(inputs)} = {result}")
    
    # ==========================================
    # Layer 2: Learn XOR (using NAND)
    # ==========================================
    xor_examples = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    
    success = agent.learn_abstraction("XOR", xor_examples, num_epochs=30)
    
    if success:
        print("\n" + "="*60)
        print("Testing XOR")
        print("="*60)
        xor_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'XOR')
        for inputs, expected in xor_examples:
            result = registry.execute_function(xor_id, inputs)
            print(f"  {'✓' if result == expected else '✗'} XOR{tuple(inputs)} = {result}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("Final Function Registry")
    print("="*60)
    for fid, meta in registry.metadata.items():
        print(f"  [{fid}] {meta['name']} (arity={meta['arity']}, layer={meta['layer']})")
    
    print("\n" + "="*60)
    print("Abstraction Hierarchy Learned!")
    print("="*60)

if __name__ == "__main__":
    main()