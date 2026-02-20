from symbolic import SymbolicRegistry
from agent import SymbolicAgent
from bootstrapping_agent import BootstrappingAgent

def main():
    print("="*60)
    print("Self-Bootstrapping Agent")
    print("Building complexity from learned functions")
    print("="*60)
    
    # Initialize
    registry = SymbolicRegistry()
    agent = SymbolicAgent(
        registry,
        d_model=128,
        max_recursion=8,
        input_dim=32,
        max_composition_depth=3
    )
    
    # Quick curriculum on primitives
    print("\n[Phase 1] Learning primitives...")
    agent.train_on_curriculum(num_epochs=15)
    
    # Create bootstrapping agent
    config = {
        'examples_per_composition': 10,
        'learning_epochs': 30,
    }
    
    bootstrap_agent = BootstrappingAgent(agent, registry, config)
    
    # Run bootstrapping
    print("\n[Phase 2] Self-bootstrapping...")
    bootstrap_agent.bootstrap_loop(num_iterations=100)
    
    # Save
    agent.save_checkpoint()
    registry.save()
    
    print("\n" + "="*60)
    print("Bootstrapping Complete!")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()