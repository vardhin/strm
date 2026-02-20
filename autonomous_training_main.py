from symbolic import SymbolicRegistry
from agent import SymbolicAgent
from autonomous_agent import AutonomousAgent

def main():
    print("="*60)
    print("Autonomous Agent: Self-Directed Learning")
    print("="*60)
    
    # Initialize base system
    registry = SymbolicRegistry()
    agent = SymbolicAgent(
        registry, 
        d_model=128, 
        max_recursion=8, 
        input_dim=32, 
        max_composition_depth=3
    )
    
    # Train on basic curriculum first
    print("\n[Phase 1] Initial curriculum training...")
    agent.train_on_curriculum(num_epochs=10)
    
    # Create autonomous agent
    autonomous_config = {
        'exploration_rate': 0.4,
        'curiosity_bonus': 0.6,
        'pruning_threshold': 0.05,
        'merge_similarity_threshold': 0.95,
    }
    
    autonomous_agent = AutonomousAgent(agent, registry, autonomous_config)
    
    # Run autonomous learning
    print("\n[Phase 2] Autonomous exploration...")
    autonomous_agent.run_continuous_learning(
        num_episodes=20,
        steps_per_episode=8
    )
    
    # Save final state
    agent.save_checkpoint()
    registry.save()
    
    print("\n" + "="*60)
    print("Autonomous Learning Complete!")
    print("="*60)
    registry.db.print_summary()

if __name__ == "__main__":
    main()