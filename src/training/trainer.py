"""
updated_complete_training_parallel.py - REPLACE your complete_training.py with this
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import multiprocessing as mp

# Import parallel environment
from environment.packing_env import VectorizedPackingEnvironment, PackingConfig 
from policy.gnn_policy import GraphPPO, TemperatureControlledActorCritic

@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Parallel settings
    n_envs: int = 8  # Start with 8, increase if you have more cores
    
    # Environment parameters
    min_cores: int = 20
    
    # Temperature scheduling
    initial_temperature: float = 1.5
    final_temperature: float = 0.1
    
    # Curriculum learning
    curriculum_phases: List[Dict] = None
    
    # Logging
    log_interval: int = 10
    save_freq: int = 50_000
    eval_freq: int = 25_000
    
    experiment_name: str = "parallel_packing_training"
    save_dir: str = "./experiments"
    seed: int = 42
    
    def __post_init__(self):
        if self.curriculum_phases is None:
            self.curriculum_phases = [
                {
                    'name': 'phase_1',
                    'domain_size': 150,
                    'timesteps': 300_000,
                    'temperature': 1.5,
                },
                {
                    'name': 'phase_2',
                    'domain_size': 300,
                    'timesteps': 400_000,
                    'temperature': 1.0,
                },
                {
                    'name': 'phase_3',
                    'domain_size': 500,
                    'timesteps': 300_000,
                    'temperature': 0.5,
                }
            ]


def create_vectorized_environment(config: TrainingConfig, phase_config: Dict):
    """Create parallel vectorized environment"""
    env_configs = []
    for i in range(config.n_envs):
        packing_config = PackingConfig(
            domain_size=phase_config['domain_size'],
            min_cores=config.min_cores
        )
        env_configs.append(packing_config)
    
    return VectorizedPackingEnvironment(env_configs, config.n_envs)


def main():
    """Main training function"""
    
    config = TrainingConfig(
        total_timesteps=1_000_000,
        n_envs=min(8, mp.cpu_count()),  # Use available cores
        seed=42
    )
    
    print(f"Starting parallel training with {config.n_envs} environments")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.save_dir) / f"{config.experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.json", 'w') as f:
        json.dump({
            'total_timesteps': config.total_timesteps,
            'n_envs': config.n_envs,
            'learning_rate': config.learning_rate,
            'seed': config.seed
        }, f, indent=2)
    
    # Create environment
    current_phase = config.curriculum_phases[0]
    env = create_vectorized_environment(config, current_phase)
    
    try:
        # Create model
        ppo_kwargs = {
            'learning_rate': config.learning_rate,
            'n_steps': config.n_steps,
            'batch_size': config.batch_size,
            'n_epochs': config.n_epochs,
            'gamma': config.gamma,
            'gae_lambda': config.gae_lambda,
            'clip_range': config.clip_range,
            'ent_coef': config.ent_coef,
            'vf_coef': config.vf_coef,
            'max_grad_norm': config.max_grad_norm
        }
        
        model = GraphPPO(env, policy_class=TemperatureControlledActorCritic, **ppo_kwargs)
        
        print("Model created successfully")
        
        # Training loop
        timesteps_completed = 0
        phase_idx = 0
        
        while timesteps_completed < config.total_timesteps:
            
            # Check for phase advancement
            if (phase_idx < len(config.curriculum_phases) - 1 and 
                timesteps_completed >= sum(p['timesteps'] for p in config.curriculum_phases[:phase_idx+1])):
                
                phase_idx += 1
                new_phase = config.curriculum_phases[phase_idx]
                
                print(f"\nAdvancing to phase {phase_idx + 1}: {new_phase['name']}")
                print(f"Domain size: {new_phase['domain_size']}")
                
                # Create new environment for new phase
                env.close()
                env = create_vectorized_environment(config, new_phase)
                model.env = env
                env.set_core_attention_policy(model.policy)
            
            # Update temperature
            current_phase = config.curriculum_phases[phase_idx]
            progress = timesteps_completed / config.total_timesteps
            current_temp = config.initial_temperature - (config.initial_temperature - config.final_temperature) * progress
            env.set_temperature(current_temp)
            
            # Training step
            print(f"Training step at timestep {timesteps_completed}")
            start_time = time.time()
            
            # Collect rollout (now parallel!)
            rollout_data = model._collect_rollout()
            
            # Update policy
            model._update_policy(rollout_data)
            
            end_time = time.time()
            
            # Extract metrics
            avg_reward = rollout_data['rewards'].mean().item()
            max_reward = rollout_data['rewards'].max().item()
            
            # Log progress
            if timesteps_completed % config.log_interval == 0:
                steps_per_second = config.n_steps * config.n_envs / (end_time - start_time)
                print(f"Step {timesteps_completed}:")
                print(f"  Avg reward: {avg_reward:.3f}")
                print(f"  Max reward: {max_reward:.3f}")
                print(f"  Temperature: {current_temp:.3f}")
                print(f"  Phase: {current_phase['name']}")
                print(f"  Steps/sec: {steps_per_second:.1f}")
                print(f"  Training time: {end_time - start_time:.2f}s")
            
            # Save model
            if timesteps_completed % config.save_freq == 0 and timesteps_completed > 0:
                save_path = exp_dir / f"model_{timesteps_completed}.pt"
                model.save(save_path)
                print(f"Model saved to {save_path}")
            
            timesteps_completed += config.n_steps
        
        # Final save
        final_path = exp_dir / "final_model.pt"
        model.save(final_path)
        print(f"Training completed! Final model saved to {final_path}")
    
    finally:
        env.close()


def quick_test():
    """Quick test of parallel environments"""
    print("Running quick parallel test...")
    
    config = TrainingConfig(
        total_timesteps=1000,
        n_envs=4,
        n_steps=64
    )
    
    phase_config = {'domain_size': 150, 'temperature': 1.0}
    env = create_vectorized_environment(config, phase_config)
    
    try:
        # Test reset
        start_time = time.time()
        obs, infos = env.reset()
        reset_time = time.time() - start_time
        print(f"Reset {config.n_envs} environments in {reset_time:.2f}s")
        
        # Test steps
        for i in range(5):
            actions = np.random.randint(0, 10, size=(config.n_envs, 2))
            start_time = time.time()
            obs, rewards, terminated, truncated, infos = env.step(actions)
            step_time = time.time() - start_time
            print(f"Step {i}: {config.n_envs} envs in {step_time:.3f}s, avg_reward={np.mean(rewards):.3f}")
        
        print("✅ Quick test passed!")
    
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()
