from symbolic import SymbolicRegistry
import os

def main():
    print("="*60)
    print("Inspecting ADD to understand LOOP pattern")
    print("="*60)
    
    if not os.path.exists("checkpoints/symbolic.db"):
        print("\n[Error] No saved registry found!")
        return
    
    registry = SymbolicRegistry()
    registry.load("checkpoints/symbolic.db")
    
    # Find ADD
    add_id = None
    for fid, meta in registry.metadata.items():
        if meta['name'] == 'ADD':
            add_id = fid
            break
    
    if add_id is None:
        print("ADD not found!")
        return
    
    print(f"\n✓ Found ADD (id={add_id})")
    
    # Get ADD structure
    meta = registry.metadata[add_id]
    print(f"\nADD metadata:")
    print(f"  Name: {meta['name']}")
    print(f"  Arity: {meta['arity']}")
    print(f"  Composition type: {meta.get('composition_type', 'unknown')}")
    print(f"  Structure: {meta.get('structure', 'unknown')}")
    
    # Try to decode the structure
    if 'structure' in meta:
        structure = meta['structure']
        print(f"\n  Detailed structure:")
        print(f"    Type: {structure.get('type', 'unknown')}")
        print(f"    Components: {structure.get('components', [])}")
        if 'loop_config' in structure:
            print(f"    Loop config: {structure['loop_config']}")
    
    # Test ADD
    print(f"\nTesting ADD:")
    test_cases = [([2, 3], 5), ([5, 4], 9), ([0, 3], 3)]
    for inputs, expected in test_cases:
        result = registry.execute_function(add_id, inputs)
        status = '✓' if result == expected else '✗'
        print(f"  {status} ADD({inputs[0]}, {inputs[1]}) = {result} (expected {expected})")
    
    # Check if we can manually create SUB pattern
    print("\n" + "="*60)
    print("Attempting manual SUB construction")
    print("="*60)
    
    # Find DEC
    dec_id = None
    for fid, meta in registry.metadata.items():
        if meta['name'] == 'DEC':
            dec_id = fid
            break
    
    if dec_id:
        print(f"\n✓ Found DEC (id={dec_id})")
        
        # Test if we can create LOOP(DEC, count) manually
        print("\nTrying to understand LOOP execution...")
        print("Expected: LOOP(body_fn=DEC, count=b, init_value=a)")
        print("  SUB(5, 2) should apply DEC twice to 5 -> 3")

if __name__ == "__main__":
    main()