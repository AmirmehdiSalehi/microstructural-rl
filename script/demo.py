"""
Simple Runner Script - Updated for Vectorized Sphere Selection Implementation
Shows exactly how to run training and evaluation with the new features.
"""

import sys
import os
import torch
import numpy as np
import time

# Add the build directory for C++ bindings
sys.path.append("build/python")

# Import our updated implementations
from environment.packing_env import TemperatureControlledPackingEnvironment, PackingConfig, VectorizedPackingEnvironment
from policy.gnn_policy import GraphPPO, TemperatureControlledActorCritic
from training.trainer import TrainingConfig, main, quick_test

def demo_sphere_selection_effects():
    """
    Demonstrate how temperature affects any-sphere selection (not just cores).
    """
    print("="*60)
    print("SPHERE SELECTION EFFECTS DEMONSTRATION")
    print("="*60)
    
    # Create environment
    config = PackingConfig(domain_size=200, min_cores=15)
    env = TemperatureControlledPackingEnvironment(config, initial_temperature=1.0)
    
    # Create model (FIXED)
    model = GraphPPO(env)
    
    # Test different temperatures
    temperatures = [2.0, 1.0, 0.5, 0.1]
    
    for temp in temperatures:
        print(f"\nTesting Temperature: {temp}")
        print("-" * 30)
        
        env.set_temperature(temp)
        obs, info = env.reset()
        
        # Take 8 steps and see which spheres are selected
        sphere_selections = []
        rewards = []
        
        for step in range(8):
            # FIXED: Use model.predict instead of env.predict
            action, _ = model.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            sphere_id = info.get('selected_sphere_id', -1)
            sphere_selections.append(sphere_id)
            rewards.append(reward)
            
            print(f"  Step {step}: Sphere {sphere_id}, Reward {reward:.3f}")
            
            if terminated or truncated:
                break
        
        # Analyze sphere selection diversity
        unique_spheres = len(set(sphere_selections))
        diversity = unique_spheres / len(sphere_selections) if sphere_selections else 0
        avg_reward = np.mean(rewards) if rewards else 0
        
        print(f"  Sphere diversity: {unique_spheres}/{len(sphere_selections)} = {diversity:.3f}")
        print(f"  Average reward: {avg_reward:.3f}")
        
        if temp > 1.0:
            print("  → High temperature: More exploration, diverse sphere selection")
        elif temp < 0.5:
            print("  → Low temperature: More exploitation, focused sphere selection")
        else:
            print("  → Balanced temperature: Mixed exploration/exploitation")

def demo_vectorized_environments():
    """
    Demonstrate vectorized environment functionality.
    """
    print("\n" + "="*60)
    print("VECTORIZED ENVIRONMENTS DEMONSTRATION")
    print("="*60)
    
    # Create configurations for vectorized environments
    env_configs = []
    n_envs = 4  # Small number for demo
    
    for i in range(n_envs):
        config = PackingConfig(
            domain_size=150 + 1 * 20,  # Different domain sizes
            min_cores=10 + i * 5
        )
        env_configs.append(config)
    
    # Create vectorized environment
    vectorized_env = VectorizedPackingEnvironment(env_configs, n_envs)
    
    print(f"Created {n_envs} parallel environments")
    print("Environment configurations:")
    for i, config in enumerate(env_configs):
        print(f"  Env {i}: {config.domain_size}³ domain, {config.min_cores} min cores")
    
    # Test vectorized operations
    print("\nTesting vectorized reset...")
    obs, infos = vectorized_env.reset()
    print(f"  Observations shape: {obs.shape}")
    print(f"  Number of info dicts: {len(infos)}")
    for i, info in enumerate(infos):
        print(f"    Env {i}: {info['n_cores']} cores")
    
    print("\nTesting vectorized step...")
    actions = np.random.randint(0, 6, size=(n_envs, 2))  # Random actions
    obs, rewards, terminated, truncated, infos = vectorized_env.step(actions)
    
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Rewards: {rewards}")
    
    print("\nTesting batched graph observations...")
    batch_graph = vectorized_env.get_batch_graph_observations()
    print(f"  Batched graph nodes: {batch_graph.x.shape[0]}")
    print(f"  Batched graph edges: {batch_graph.edge_index.shape[1]}")
    print(f"  Batch tensor: {batch_graph.batch.shape}")
    
    # Clean up
    vectorized_env.close()

def demo_cpp_integration():
    """
    Demonstrate improved C++ integration.
    """
    print("\n" + "="*60)
    print("C++ INTEGRATION DEMONSTRATION")
    print("="*60)
    
    try:
        import particle_packing as pp
        print("✓ C++ bindings loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import C++ bindings: {e}")
        print("  Please build the C++ components first:")
        print("  mkdir build && cd build && cmake .. && cmake --build .")
        return
    
    # Create environment
    config = PackingConfig(domain_size=200, min_cores=20)
    env = TemperatureControlledPackingEnvironment(config)
    
    # Reset to create C++ generator
    obs, info = env.reset()
    print(f"Created environment with {info['n_cores']} core spheres")
    
    # Access C++ generator directly
    generator = env.generator
    if generator:
        print("\nC++ PackingGenerator properties:")
        print(f"  Particle count: {generator.getParticleCount()}")
        print(f"  Current density: {generator.getCurrentDensity():.4f}")
        print(f"  Contact count: {generator.getContactCount()}")
        print(f"  Average particle radius: {generator.getAverageParticleRadius():.2f}")
        print(f"  Total volume: {generator.getTotalVolume()}")
        
        # Demonstrate contact pairs
        contact_pairs = generator.getContactPairs()
        print(f"  Contact pairs: {len(contact_pairs)}")
        
        # Demonstrate particle volumes
        particles = generator.getParticles()
        for i, particle in enumerate(particles[:3]):  # Show first 3 particles
            volume = generator.getParticleVolume(particle.getId())
            print(f"    Particle {particle.getId()}: volume = {volume}")
    
    # Test adding spheres via C++
    print("\nTesting sphere addition via C++...")
    initial_particles = generator.getParticleCount() if generator else 0
    
    # Create a simple model for testing
    model = GraphPPO(env)
    
    # Take a few steps
    for step in range(3):
        action, _ = model.predict(obs)  # FIXED: Use model.predict
        obs, reward, terminated, truncated, info = env.step(action)
        
        if generator:
            current_particles = generator.getParticleCount()
            print(f"  Step {step}: particles = {current_particles}, "
                  f"density = {generator.getCurrentDensity():.4f}, "
                  f"contacts = {generator.getContactCount()}")
        
        if terminated or truncated:
            break

def demo_sphere_connectivity():
    """
    Demonstrate sphere connectivity analysis.
    """
    print("\n" + "="*60)
    print("SPHERE CONNECTIVITY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    try:
        # Import with the fixed path approach
        sys.path.append('sphere_connectivity')
        from connectivity_checker import reliable_sphere_connectivity
        print("✓ Sphere connectivity checker loaded successfully")
    except ImportError:
        print("✗ Sphere connectivity checker not found")
        print("  Please run: python sphere_connectivity/generate_lookup.py")
        return
    
    # Test connectivity for different sphere pairs
    test_cases = [
        (10, 15, 24.0),  # Should be connected
        (10, 15, 26.0),  # Should not be connected
        (5, 7, 11.5),    # Edge case
        (20, 20, 39.8),  # Large spheres, edge case
    ]
    
    print("Testing sphere connectivity:")
    for r1, r2, distance in test_cases:
        is_connected = reliable_sphere_connectivity(r1, r2, distance)
        analytical = distance < (r1 + r2)
        
        print(f"  Spheres r1={r1}, r2={r2}, d={distance}: "
              f"connected={is_connected}, analytical={analytical}")
        
        if is_connected != analytical:
            print(f"    ⚠️  Discrepancy detected! Using voxel-level result.")

def run_short_vectorized_training():
    """
    Run a short vectorized training session to test everything works.
    """
    print("\n" + "="*60)
    print("SHORT VECTORIZED TRAINING SESSION")
    print("="*60)
    
    # Create training config for short run
    config = TrainingConfig(
        total_timesteps=1_000,  # Very short for demo
        n_envs=4,  # Small number of environments
        min_cores=15,
        curriculum_phases=[{  # FIXED: Proper curriculum config
            'name': 'demo_phase',
            'domain_size': 100,
            'timesteps': 1_000,
            'temperature': 1.0,
        }],
        initial_temperature=1.5,
        final_temperature=0.8,
        n_steps=32,  # Smaller rollouts
        experiment_name="short_vectorized_demo"
    )
    
    print(f"Running {config.total_timesteps} timesteps with {config.n_envs} environments...")
    print(f"Temperature will decay from {config.initial_temperature} to {config.final_temperature}")
    
    # Create vectorized environment
    env_configs = []
    for i in range(config.n_envs):
        packing_config = PackingConfig(
            domain_size=100,
            min_cores=config.min_cores
        )
        env_configs.append(packing_config)
    
    env = VectorizedPackingEnvironment(env_configs, config.n_envs)
    
    try:
        # Create model
        model = GraphPPO(
            env, # Since it's the vectorized version, essentially 4x policy networks are created
            policy_class=TemperatureControlledActorCritic,
            learning_rate=3e-4,
            n_steps=32,
            batch_size=32,
            n_epochs=4
        )
        
        # Configuration
        n_training_iterations = 5

        env.reset()

        # Training loop
        for iteration in range(n_training_iterations):
            print(f"\nIteration {iteration + 1}/{n_training_iterations}")
            
            # Phase 1: Collect rollout
            print("  Collecting rollout...")
            rollout_data = model._collect_rollout()
            
            for step in range(config.n_steps):
                print(f"\nStep {step}:")
                for i in range(config.n_envs):
                    print(f"Reward for env {i}: {rollout_data['rewards'][step][i]} \n")
                    print(f"action for env {i}: {rollout_data['actions'][step][i]} \n")

            # Phase 2: Update policy
            print("  Updating policy...")
            model._update_policy(rollout_data)
            
            # Phase 3: Synchronize updated policy to workers
            print("  Synchronizing policy...")
            env.set_core_attention_policy(model.policy)
            
            # Phase 4: Update temperature
            progress = iteration / n_training_iterations
            temperature = 1.5 - 1.0 * progress  # Decay from 1.5 to 0.5
            env.set_temperature(temperature)
            
            print(f"  ✓ Iteration complete (temperature: {temperature:.2f})")
    
    finally:

        env.close()
        print("\n✓ Training example completed successfully!")

def show_updated_usage_examples():
    """
    Show various usage examples for the updated implementation.
    """
    print("\n" + "="*60)
    print("UPDATED USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Basic Environment Usage (Updated):")
    print("-" * 30)
    print("""
# Create environment with any-sphere selection
config = PackingConfig(domain_size=300, min_cores=20)
env = TemperatureControlledPackingEnvironment(config, initial_temperature=1.0)

# Reset uses C++ core generation
obs, info = env.reset()
print(f"Generated {info['n_cores']} cores using C++")

# Step with sphere selection (any sphere, not just cores)
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f"Selected sphere ID: {info['selected_sphere_id']}")
""")
    
    print("\n2. Vectorized Environment Usage (NEW):")
    print("-" * 30)
    print("""
# Create vectorized environments
env_configs = [PackingConfig(domain_size=150+i*50, min_cores=20) for i in range(4)]
vectorized_env = VectorizedPackingEnvironment(env_configs, n_envs=4)

# Vectorized operations
obs, infos = vectorized_env.reset()  # Shape: (4, 1)
actions = np.random.randint(0, 10, (4, 2))  # 4 environments, 2 action dims
obs, rewards, terminated, truncated, infos = vectorized_env.step(actions)

# Batched graph observations for GNN
batch_graph = vectorized_env.get_batch_graph_observations()
""")
    
    print("\n3. C++ Integration (Enhanced):")
    print("-" * 30)
    print("""
# Access C++ generator directly
generator = env.generator
contact_pairs = generator.getContactPairs()  # All particle contacts
total_volume = generator.getTotalVolume()    # Voxel-accurate volume
particle_volume = generator.getParticleVolume(particle_id)

# Add spheres using C++ (automatic voxel tracking)
success = generator.insertSuppSpheres(x, y, z, radius, particle_id)
""")
    
    print("\n4. Sphere Connectivity Analysis (NEW):")
    print("-" * 30)
    print("""
sys.path.append('sphere_connectivity')
from connectivity_checker import reliable_sphere_connectivity

# 100% reliable connectivity check (analytical + lookup table)
is_connected = reliable_sphere_connectivity(radius1, radius2, distance)

# Works with all radii from the connectivity analysis
radii = [2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
""")
    
    print("\n5. Training with Vectorization (NEW):")
    print("-" * 30)
    print("""
# Configure vectorized training
config = TrainingConfig(
    total_timesteps=5_000_000,
    n_envs=32,  # 32 parallel environments
    curriculum_phases=[...],  # Proper curriculum config
    initial_temperature=1.5,
    final_temperature=0.1,
    experiment_name="vectorized_training"
)

# Training automatically uses all environments
python updated_complete_training.py
""")

def main_demo():
    """
    Main demonstration function for updated implementation.
    """
    print("Vectorized Sphere Selection Particle Packing RL")
    print("Updated Implementation Demo")
    print("="*60)
    
    try:
        # Check if C++ bindings are available
        import particle_packing as pp
        print("✓ C++ bindings found")
    except ImportError:
        print("✗ C++ bindings not found - build the C++ components first")
        print("  Run: mkdir build && cd build && cmake .. && cmake --build .")
        return
    
    # Check sphere connectivity
    try:
        sys.path.append('sphere_connectivity')
        from connectivity_checker import reliable_sphere_connectivity
        print("✓ Sphere connectivity checker found")
    except ImportError:
        print("⚠️  Sphere connectivity checker not found")
        print("  Run: python sphere_connectivity/generate_lookup.py")
        print("  (Will use analytical method as fallback)")
    
    # Show usage examples
    show_updated_usage_examples()
    
    # Demonstrate sphere selection effects
    demo_sphere_selection_effects()
    
    # Demonstrate vectorized environments
    demo_vectorized_environments()
    
    # Demonstrate C++ integration
    demo_cpp_integration()
    
    # Demonstrate sphere connectivity
    demo_sphere_connectivity()
    
    # Run short vectorized training
    run_short_vectorized_training()
    
    print("\n" + "="*60)
    print("UPDATED DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate connectivity lookup: 'python sphere_connectivity/generate_lookup.py'")
    print("2. Run quick test: 'python updated_complete_training.py test'")
    print("3. Run full vectorized training: 'python updated_complete_training.py'")
    print("4. Submit HPC job: 'sbatch updated_slurm_template.sh'")
    print("5. Evaluate trained model: 'python updated_evaluation.py model.pt --report --plots'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "sphere":
            demo_sphere_selection_effects()
        elif sys.argv[1] == "vectorized":
            demo_vectorized_environments()
        elif sys.argv[1] == "cpp":
            demo_cpp_integration()
        elif sys.argv[1] == "connectivity":
            demo_sphere_connectivity()
        elif sys.argv[1] == "train":
            run_short_vectorized_training()
        else:
            print("Options: quick, sphere, vectorized, cpp, connectivity, train")
    else:
        # main_demo()
        run_short_vectorized_training()
