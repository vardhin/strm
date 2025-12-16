"""
Main script for neurosymbolic TRM with assembly-like primitives
Full visibility into reasoning process
"""
import torch
import numpy as np
from primitives_manager import PrimitivesManager
from neurosymbolic_trainer import NeurosymbolicTrainer

def generate_identity_data(n_samples=20, max_val=10):
    """
    Simplest task: (a) -> (a)  [just return the input]
    
    Returns:
        np.array of shape [n_samples, 2] where [:, 0] is input and [:, 1] is output
    """
    data = np.zeros((n_samples, 2), dtype=np.int32)
    data[:, 0] = np.random.randint(0, max_val, n_samples)  # a
    data[:, 1] = data[:, 0]  # target = a
    return data

def generate_addition_data(n_samples=30, max_val=10):
    """
    Generate addition dataset: (a, b) -> (a+b)
    
    Returns:
        np.array of shape [n_samples, 3] where [:, :2] are inputs and [:, 2] is output
    """
    data = np.zeros((n_samples, 3), dtype=np.int32)
    data[:, 0] = np.random.randint(0, max_val, n_samples)  # a
    data[:, 1] = np.random.randint(0, max_val, n_samples)  # b
    data[:, 2] = data[:, 0] + data[:, 1]  # target = a + b
    return data

def generate_increment_data(n_samples=20, max_val=10):
    """
    Simple task: (a) -> (a+1)
    
    Returns:
        np.array of shape [n_samples, 2]
    """
    data = np.zeros((n_samples, 2), dtype=np.int32)
    data[:, 0] = np.random.randint(0, max_val, n_samples)  # a
    data[:, 1] = data[:, 0] + 1  # target = a + 1
    return data

def generate_double_plus_one_data(n_samples=20, max_val=5):
    """
    Harder task: (a) -> (2*a + 1)
    No demonstration - must discover from scratch!
    
    Returns:
        np.array of shape [n_samples, 2]
    """
    data = np.zeros((n_samples, 2), dtype=np.int32)
    data[:, 0] = np.random.randint(0, max_val, n_samples)  # a
    data[:, 1] = 2 * data[:, 0] + 1  # target = 2*a + 1
    return data

def main():
    # Initialize primitive manager with assembly-like primitives
    prims = PrimitivesManager()
    
    print("\n" + "="*80)
    print("NEUROSYMBOLIC TRM - Assembly-Like Primitives")
    print("="*80)
    print("\nStarting with LOW-LEVEL primitives:")
    print("-" * 80)
    
    # Group primitives by category for better readability
    categories = {
        'Constants': ['ZERO', 'ONE'],
        'Load Args': [f'LOAD_{i}' for i in range(5)],
        'Arithmetic': ['ADD', 'SUB', 'MUL', 'DIV', 'MOD'],
        'Unary': ['INC', 'DEC', 'NEG'],
        'Comparison': ['EQ', 'LT', 'GT'],
        'Bitwise': ['AND', 'OR', 'XOR', 'NOT'],
        'Control': ['IF_THEN', 'STORE', 'RECALL', 'START', 'DONE', 'SEP']
    }
    
    for category, prim_names in categories.items():
        print(f"\n{category}:")
        for name in prim_names:
            if name in prims.primitives:
                prim = prims.primitives[name]
                print(f"  • {name:12s} → {prim.symbolic_form}")
    
    # Phase 0: Learn IDENTITY (just return input)
    print("\n" + "="*80)
    print("PHASE 0: Learning IDENTITY (warmup)")
    print("="*80)
    print("\n🎯 Goal: Learn to just return the input using LOAD_0")
    print("   Expected program: LOAD_0")
    
    trainer = NeurosymbolicTrainer(prims, learning_rate=1e-3, verbose=True)
    
    identity_data = generate_identity_data(n_samples=5, max_val=5)
    print(f"\n📊 Generated {len(identity_data)} identity examples:")
    for i in range(min(3, len(identity_data))):
        print(f"   {identity_data[i, 0]} -> {identity_data[i, 1]}")
    
    success, equation = trainer.train_phase(
        task_name="identity",
        train_data=identity_data,
        n_inputs=1,
        n_outputs=1,
        num_epochs=20,
        detailed_example_idx=0
    )
    
    if success:
        print(f"\n✅ Learned identity! Program: {equation}")
    else:
        print("\n⚠️  Couldn't learn identity - this is concerning!")
        print("   Model might need hyperparameter tuning")
    
    # Phase 1: Learn INCREMENT  
    print("\n" + "="*80)
    print("PHASE 1: Learning INCREMENT")
    print("="*80)
    print("\n🎯 Goal: Learn (a) -> (a+1) using LOAD_0, INC")
    print("   Expected program: LOAD_0 INC")
    
    trainer = NeurosymbolicTrainer(prims, learning_rate=1e-3, verbose=True)
    
    increment_data = generate_increment_data(n_samples=5, max_val=5)
    print(f"\n📊 Generated {len(increment_data)} increment examples:")
    for i in range(min(3, len(increment_data))):
        print(f"   {increment_data[i, 0]} + 1 = {increment_data[i, 1]}")
    
    success, equation = trainer.train_phase(
        task_name="increment",
        train_data=increment_data,
        n_inputs=1,
        n_outputs=1,
        num_epochs=20,
        detailed_example_idx=0
    )
    
    if success:
        print(f"\n✅ Learned increment! Program: {equation}")
    
    # Phase 2: Learn ADDITION (harder)
    print("\n" + "="*80)
    print("PHASE 2: Learning ADDITION")
    print("="*80)
    print("\n🎯 Goal: Learn (a, b) -> (a+b) using LOAD_0, LOAD_1, ADD")
    print("   Expected program: LOAD_0 LOAD_1 ADD")
    
    trainer = NeurosymbolicTrainer(prims, learning_rate=1e-3, verbose=False)
    
    addition_data = generate_addition_data(n_samples=10, max_val=5)
    print(f"\n📊 Generated {len(addition_data)} addition examples:")
    for i in range(min(3, len(addition_data))):
        print(f"   {addition_data[i, 0]} + {addition_data[i, 1]} = {addition_data[i, 2]}")
    
    success, equation = trainer.train_phase(
        task_name="addition",
        train_data=addition_data,
        n_inputs=2,
        n_outputs=1,
        num_epochs=50,
        detailed_example_idx=0
    )
    
    if success:
        print(f"\n✅ SUCCESS! Discovered addition!")
        print(f"   Program: {equation}")
    else:
        print("\n❌ Failed to discover addition")
        print("   This is expected - the task is hard!")
    
    # Phase 3: Learn DOUBLE_PLUS_ONE (harder - no demo!)
    print("\n" + "="*80)
    print("PHASE 3: Learning DOUBLE + 1 (no demonstration!)")
    print("="*80)
    print("\n🎯 Goal: Learn (a) -> (2*a + 1)")
    print("   Must compose: LOAD_0, CONST_2, MUL, CONST_1, ADD")
    print("   Expected program: LOAD_0 CONST_2 MUL CONST_1 ADD")
    
    trainer = NeurosymbolicTrainer(prims, learning_rate=1e-3, verbose=False)
    
    double_plus_one_data = generate_double_plus_one_data(n_samples=10, max_val=5)
    print(f"\n📊 Generated {len(double_plus_one_data)} examples:")
    for i in range(min(3, len(double_plus_one_data))):
        a = double_plus_one_data[i, 0]
        target = double_plus_one_data[i, 1]
        print(f"   2*{a} + 1 = {target}")
    
    success, equation = trainer.train_phase(
        "double_plus_one",
        double_plus_one_data,
        n_inputs=1,
        n_outputs=1,
        num_epochs=50,  # More epochs since no demo
        use_demo=False  # Force discovery!
    )
    
    if success:
        print(f"\n✅ SUCCESS! Discovered double+1!")
        print(f"   Program: {equation}")
        
        # Test generalization on NEW data
        print(f"\n🧪 Testing generalization on unseen examples:")
        test_data = generate_double_plus_one_data(n_samples=5, max_val=10)
        program_tokens = equation.split()
        
        test_correct = 0
        for i in range(len(test_data)):
            a = int(test_data[i, 0])
            expected = int(test_data[i, 1])
            
            result = trainer.executor.execute_program(
                program_tokens, 
                (a,),  # Single input
                trace=False
            )
            
            if result is not None:
                actual = result[0] if isinstance(result, tuple) else result
                is_correct = (actual == expected)
                test_correct += is_correct
                
                status = "✓" if is_correct else "✗"
                print(f"   {status} f({a}) = {actual} (expected {expected})")
            else:
                print(f"   ✗ f({a}) = ERROR (expected {expected})")
        
        test_accuracy = test_correct / len(test_data)
        print(f"\n   Test accuracy: {test_accuracy*100:.1f}%")
        
        if test_accuracy < 0.8:
            print(f"   ⚠️  Program doesn't generalize well!")
            print(f"   The discovered program may be overfitting to training examples")
    else:
        print(f"\n❌ Could not discover double+1")
        print(f"   Best attempt: {equation}")
        print(f"   This is expected - composition is hard!")
    
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()