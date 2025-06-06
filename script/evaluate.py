"""
Updated Model Evaluation Script for Vectorized Sphere Selection Implementation
Works with the new sphere selection environment and policy.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse
import sys

# Import updated implementation
from environment.packing_env import TemperatureControlledPackingEnvironment, PackingConfig
from policy.gnn_policy import GraphPPO, TemperatureControlledActorCritic
from training.trainer import TrainingConfig

class UpdatedModelEvaluator:
    """Evaluation for the updated sphere selection implementation."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_path: Path to saved model (.pt file)
            config_path: Path to training configuration (optional)
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.training_config = json.load(f)
        else:
            # Default configuration
            self.training_config = {
                'min_cores': 20,
                'n_envs': 1
            }
        
        # Create environment configuration
        self.env_config = PackingConfig(
            domain_size=300,  # Use medium size for evaluation
            min_cores=self.training_config.get('min_cores', 20)
        )
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self) -> GraphPPO:
        """Load the trained model."""
        try:
            # Create environment for model loading
            env = TemperatureControlledPackingEnvironment(
                self.env_config, 
                initial_temperature=0.1  # Low temperature for evaluation
            )
            
            # Create model with same architecture
            model = GraphPPO(env, policy_class=TemperatureControlledActorCritic)
            
            # Load saved weights
            model.load(self.model_path)
            
            print(f"✓ Model loaded from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Make sure the model was saved with the updated implementation")
            raise
    
    def evaluate_sphere_selection_performance(self, domain_size: int = 300, n_episodes: int = 20, 
                                            temperature: float = 0.1) -> Dict:
        """
        Evaluate model performance focusing on sphere selection strategies.
        
        Args:
            domain_size: Domain size for evaluation
            n_episodes: Number of episodes to evaluate
            temperature: Temperature for sphere selection (low for evaluation)
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        print(f"Evaluating sphere selection on {domain_size}³ domain over {n_episodes} episodes (T={temperature})...")
        
        # Create environment for this evaluation
        eval_config = PackingConfig(
            domain_size=domain_size,
            min_cores=self.env_config.min_cores
        )
        
        env = TemperatureControlledPackingEnvironment(
            eval_config, 
            initial_temperature=temperature
        )
        
        # Connect model to environment
        env.set_core_attention_policy(self.model.policy)
        
        episode_results = []
        
        for episode in range(n_episodes):
            # Reset environment
            obs, info = env.reset()
            
            episode_data = {
                'episode': episode,
                'initial_cores': info['n_cores'],
                'rewards': [],
                'avg_particle_sizes': [],
                'contact_counts': [],
                'steps': 0,
                'sphere_selections': [],
                'sphere_selection_diversity': 0.0,
                'final_metrics': {}
            }
            
            total_reward = 0
            step = 0
            terminated = False
            truncated = False
            selected_spheres = set()
            
            while not (terminated or truncated) and step < 800:  # Safety limit
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record data
                episode_data['rewards'].append(reward)
                episode_data['avg_particle_sizes'].append(info.get('avg_particle_size', 0))
                episode_data['contact_counts'].append(info.get('contact_count', 0))
                
                # Track sphere selection
                selected_sphere_id = info.get('selected_sphere_id', -1)
                episode_data['sphere_selections'].append(selected_sphere_id)
                selected_spheres.add(selected_sphere_id)
                
                total_reward += reward
                step += 1
            
            # Calculate final metrics
            episode_data['steps'] = step
            episode_data['total_reward'] = total_reward
            episode_data['final_avg_particle_size'] = episode_data['avg_particle_sizes'][-1] if episode_data['avg_particle_sizes'] else 0
            episode_data['final_contact_count'] = episode_data['contact_counts'][-1] if episode_data['contact_counts'] else 0
            
            # Calculate sphere selection diversity
            total_selections = len(episode_data['sphere_selections'])
            unique_spheres = len(selected_spheres)
            episode_data['sphere_selection_diversity'] = unique_spheres / max(1, total_selections)
            
            episode_results.append(episode_data)
            
            if episode % 5 == 0:
                print(f"  Episode {episode}: reward={total_reward:.3f}, "
                      f"particle_size={episode_data['final_avg_particle_size']:.3f}, "
                      f"contacts={episode_data['final_contact_count']}, "
                      f"sphere_diversity={episode_data['sphere_selection_diversity']:.3f}")
        
        # Aggregate results
        aggregated_results = self._aggregate_episode_results(episode_results, domain_size)
        
        return aggregated_results
    
    def _aggregate_episode_results(self, episode_results: List[Dict], domain_size: int) -> Dict:
        """Aggregate results across episodes."""
        
        # Extract key metrics
        total_rewards = [ep['total_reward'] for ep in episode_results]
        final_particle_sizes = [ep['final_avg_particle_size'] for ep in episode_results]
        final_contact_counts = [ep['final_contact_count'] for ep in episode_results]
        episode_lengths = [ep['steps'] for ep in episode_results]
        sphere_diversities = [ep['sphere_selection_diversity'] for ep in episode_results]
        
        results = {
            'domain_size': domain_size,
            'n_episodes': len(episode_results),
            
            # Reward metrics
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            
            # Performance metrics
            'mean_particle_size': np.mean(final_particle_sizes),
            'std_particle_size': np.std(final_particle_sizes),
            'mean_contact_count': np.mean(final_contact_counts),
            'std_contact_count': np.std(final_contact_counts),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            
            # Sphere selection metrics (NEW)
            'mean_sphere_diversity': np.mean(sphere_diversities),
            'std_sphere_diversity': np.std(sphere_diversities),
            
            # Success rate (particle size > 30.0)
            'success_rate': np.mean([s > 30.0 for s in final_particle_sizes]),
            
            # Raw data for further analysis
            'raw_rewards': total_rewards,
            'raw_particle_sizes': final_particle_sizes,
            'raw_contact_counts': final_contact_counts,
            'raw_episode_lengths': episode_lengths,
            'raw_sphere_diversities': sphere_diversities
        }
        
        return results
    
    def evaluate_temperature_effects(self, domain_size: int = 300, n_episodes: int = 10) -> Dict:
        """
        Evaluate how temperature affects performance and sphere selection.
        
        Args:
            domain_size: Domain size to evaluate
            n_episodes: Episodes per temperature
            
        Returns:
            temperature_results: Results for each temperature
        """
        print(f"Evaluating temperature effects on {domain_size}³ domain...")
        
        temperatures = [2.0, 1.5, 1.0, 0.5, 0.1]
        temperature_results = {}
        
        for temp in temperatures:
            print(f"  Testing temperature {temp}...")
            results = self.evaluate_sphere_selection_performance(domain_size, n_episodes, temp)
            temperature_results[temp] = results
        
        return temperature_results
    
    def evaluate_domain_scaling(self, n_episodes: int = 15) -> Dict:
        """Evaluate performance across different domain sizes."""
        
        print("Evaluating performance across domain sizes...")
        
        domain_sizes = [150, 300, 500]
        scaling_results = {}
        
        for domain_size in domain_sizes:
            print(f"  Testing domain size {domain_size}³...")
            results = self.evaluate_sphere_selection_performance(domain_size, n_episodes)
            scaling_results[domain_size] = results
        
        # Create scaling analysis
        scaling_analysis = {
            'domain_scaling': scaling_results,
            'scaling_trends': {
                'reward_scaling': [scaling_results[ds]['mean_reward'] for ds in domain_sizes],
                'particle_size_scaling': [scaling_results[ds]['mean_particle_size'] for ds in domain_sizes],
                'contact_scaling': [scaling_results[ds]['mean_contact_count'] for ds in domain_sizes],
                'diversity_scaling': [scaling_results[ds]['mean_sphere_diversity'] for ds in domain_sizes]
            }
        }
        
        return scaling_analysis
    
    def analyze_sphere_selection_patterns(self, domain_size: int = 300, n_episodes: int = 20) -> Dict:
        """Analyze sphere selection patterns and strategies."""
        
        print("Analyzing sphere selection patterns...")
        
        # Create environment
        eval_config = PackingConfig(domain_size=domain_size, min_cores=20)
        env = TemperatureControlledPackingEnvironment(eval_config, initial_temperature=0.1)
        env.set_core_attention_policy(self.model.policy)
        
        selection_data = []
        attention_weights_data = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            
            episode_selections = []
            episode_attention = []
            
            for step in range(100):  # Analyze first 100 steps
                # Get attention weights
                graph_obs = env.get_current_graph_observation()
                if graph_obs is not None:
                    with torch.no_grad():
                        attention_weights = self.model.policy.get_sphere_attention_weights(graph_obs)
                        if attention_weights.numel() > 0:
                            episode_attention.append(attention_weights.cpu().numpy())
                
                # Take action
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_selections.append(info.get('selected_sphere_id', -1))
                
                if terminated or truncated:
                    break
            
            selection_data.append(episode_selections)
            attention_weights_data.append(episode_attention)
        
        # Analyze patterns
        analysis = {
            'selection_entropy': self._calculate_selection_entropy(selection_data),
            'attention_consistency': self._calculate_attention_consistency(attention_weights_data),
            'selection_frequency': self._calculate_selection_frequency(selection_data),
            'exploration_vs_exploitation': self._analyze_exploration_exploitation(selection_data)
        }
        
        return analysis
    
    def _calculate_selection_entropy(self, selection_data: List[List[int]]) -> float:
        """Calculate entropy of sphere selections."""
        all_selections = [sel for episode in selection_data for sel in episode if sel >= 0]
        if not all_selections:
            return 0.0
        
        from collections import Counter
        counts = Counter(all_selections)
        total = len(all_selections)
        
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        return entropy
    
    def _calculate_attention_consistency(self, attention_data: List[List[np.ndarray]]) -> float:
        """Calculate consistency of attention weights."""
        if not attention_data or not attention_data[0]:
            return 0.0
        
        # Calculate variance in attention weights
        all_attention = []
        min_length = min(len(episode) for episode in attention_data if episode)
        
        for i in range(min_length):
            step_attention = []
            for episode in attention_data:
                if i < len(episode) and len(episode[i]) > 0:
                    step_attention.append(episode[i])
            
            if step_attention and all(len(att) == len(step_attention[0]) for att in step_attention):
                all_attention.append(np.std(step_attention, axis=0).mean())
        
        return np.mean(all_attention) if all_attention else 0.0
    
    def _calculate_selection_frequency(self, selection_data: List[List[int]]) -> Dict:
        """Calculate frequency of different selection patterns."""
        from collections import Counter
        
        all_selections = [sel for episode in selection_data for sel in episode if sel >= 0]
        
        if not all_selections:
            return {'most_common': [], 'selection_counts': {}}
        
        counts = Counter(all_selections)
        return {
            'most_common': counts.most_common(10),
            'selection_counts': dict(counts),
            'unique_selections': len(counts)
        }
    
    def _analyze_exploration_exploitation(self, selection_data: List[List[int]]) -> Dict:
        """Analyze exploration vs exploitation in sphere selection."""
        exploration_scores = []
        
        for episode in selection_data:
            if len(episode) <= 1:
                continue
            
            unique_selections = len(set(episode))
            total_selections = len(episode)
            exploration_score = unique_selections / total_selections
            exploration_scores.append(exploration_score)
        
        return {
            'mean_exploration': np.mean(exploration_scores) if exploration_scores else 0.0,
            'std_exploration': np.std(exploration_scores) if exploration_scores else 0.0,
            'exploration_trend': exploration_scores
        }
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        
        print("Generating comprehensive evaluation report...")
        
        # Run comprehensive evaluation
        performance_300 = self.evaluate_sphere_selection_performance(domain_size=300, n_episodes=20)
        temp_effects = self.evaluate_temperature_effects(domain_size=300, n_episodes=8)
        scaling_analysis = self.evaluate_domain_scaling(n_episodes=15)
        selection_patterns = self.analyze_sphere_selection_patterns(domain_size=300, n_episodes=15)
        
        # Generate report
        report = []
        report.append("="*60)
        report.append("UPDATED SPHERE SELECTION MODEL EVALUATION")
        report.append("="*60)
        report.append(f"Model: {self.model_path}")
        report.append(f"Environment: Any-sphere selection with temperature control")
        report.append("")
        
        # Performance results
        report.append("PERFORMANCE RESULTS (300³ domain)")
        report.append("-" * 40)
        report.append(f"Mean Reward: {performance_300['mean_reward']:.3f} ± {performance_300['std_reward']:.3f}")
        report.append(f"Mean Particle Size: {performance_300['mean_particle_size']:.3f} ± {performance_300['std_particle_size']:.3f}")
        report.append(f"Mean Contact Count: {performance_300['mean_contact_count']:.1f} ± {performance_300['std_contact_count']:.1f}")
        report.append(f"Success Rate: {performance_300['success_rate']:.1%}")
        report.append(f"Mean Episode Length: {performance_300['mean_episode_length']:.1f} steps")
        report.append(f"Sphere Selection Diversity: {performance_300['mean_sphere_diversity']:.3f} ± {performance_300['std_sphere_diversity']:.3f}")
        report.append("")
        
        # Temperature effects
        report.append("TEMPERATURE EFFECTS")
        report.append("-" * 40)
        for temp, results in temp_effects.items():
            report.append(f"T={temp}: Reward={results['mean_reward']:.3f}, "
                         f"ParticleSize={results['mean_particle_size']:.3f}, "
                         f"Diversity={results['mean_sphere_diversity']:.3f}")
        report.append("")
        
        # Domain scaling
        report.append("DOMAIN SCALING ANALYSIS")
        report.append("-" * 40)
        for domain_size, results in scaling_analysis['domain_scaling'].items():
            report.append(f"{domain_size}³: Reward={results['mean_reward']:.3f}, "
                         f"ParticleSize={results['mean_particle_size']:.3f}, "
                         f"Contacts={results['mean_contact_count']:.1f}")
        report.append("")
        
        # Sphere selection analysis
        report.append("SPHERE SELECTION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Selection Entropy: {selection_patterns['selection_entropy']:.3f}")
        report.append(f"Attention Consistency: {selection_patterns['attention_consistency']:.3f}")
        report.append(f"Mean Exploration Score: {selection_patterns['exploration_vs_exploitation']['mean_exploration']:.3f}")
        report.append(f"Unique Selections: {selection_patterns['selection_frequency']['unique_selections']}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Best domain size: {max(scaling_analysis['domain_scaling'].keys(), key=lambda x: scaling_analysis['domain_scaling'][x]['mean_reward'])}³")
        report.append(f"Optimal temperature: {min(temp_effects.keys(), key=lambda x: -temp_effects[x]['mean_reward'])}")
        report.append(f"Model demonstrates {'high' if performance_300['mean_sphere_diversity'] > 0.5 else 'low'} sphere selection diversity")
        report.append(f"Performance scales {'well' if scaling_analysis['scaling_trends']['reward_scaling'][-1] > scaling_analysis['scaling_trends']['reward_scaling'][0] else 'poorly'} with domain size")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
    
    def visualize_performance(self, save_dir: Optional[str] = None):
        """Create comprehensive performance visualization plots."""
        
        print("Creating performance visualizations...")
        
        # Run evaluations
        performance_300 = self.evaluate_sphere_selection_performance(domain_size=300, n_episodes=20)
        temp_effects = self.evaluate_temperature_effects(domain_size=300, n_episodes=8)
        scaling_analysis = self.evaluate_domain_scaling(n_episodes=15)
        selection_patterns = self.analyze_sphere_selection_patterns(domain_size=300, n_episodes=15)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Updated Sphere Selection Model Performance Analysis', fontsize=16)
        
        # Plot 1: Temperature effects on reward
        temps = list(temp_effects.keys())
        temp_rewards = [temp_effects[t]['mean_reward'] for t in temps]
        temp_reward_stds = [temp_effects[t]['std_reward'] for t in temps]
        
        axes[0, 0].errorbar(temps, temp_rewards, yerr=temp_reward_stds, marker='o', capsize=5)
        axes[0, 0].set_title('Temperature vs Reward')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Temperature effects on sphere diversity
        temp_diversities = [temp_effects[t]['mean_sphere_diversity'] for t in temps]
        temp_diversity_stds = [temp_effects[t]['std_sphere_diversity'] for t in temps]
        
        axes[0, 1].errorbar(temps, temp_diversities, yerr=temp_diversity_stds, marker='s', capsize=5, color='red')
        axes[0, 1].set_title('Temperature vs Sphere Diversity')
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Sphere Selection Diversity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Domain scaling
        domain_sizes = list(scaling_analysis['domain_scaling'].keys())
        scaling_rewards = [scaling_analysis['domain_scaling'][ds]['mean_reward'] for ds in domain_sizes]
        scaling_particle_sizes = [scaling_analysis['domain_scaling'][ds]['mean_particle_size'] for ds in domain_sizes]
        
        ax3_twin = axes[0, 2].twinx()
        line1 = axes[0, 2].plot(domain_sizes, scaling_rewards, 'b-o', label='Reward')
        line2 = ax3_twin.plot(domain_sizes, scaling_particle_sizes, 'r-s', label='Particle Size')
        
        axes[0, 2].set_xlabel('Domain Size')
        axes[0, 2].set_ylabel('Mean Reward', color='b')
        ax3_twin.set_ylabel('Mean Particle Size', color='r')
        axes[0, 2].set_title('Domain Size Scaling')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Reward vs Particle Size correlation
        rewards = performance_300['raw_rewards']
        particle_sizes = performance_300['raw_particle_sizes']
        
        axes[0, 3].scatter(rewards, particle_sizes, alpha=0.6, s=30)
        axes[0, 3].set_title('Reward vs Particle Size')
        axes[0, 3].set_xlabel('Episode Reward')
        axes[0, 3].set_ylabel('Final Particle Size')
        axes[0, 3].grid(True, alpha=0.3)
        
        # Plot 5: Reward distribution
        axes[1, 0].hist(performance_300['raw_rewards'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(performance_300['mean_reward'], color='red', linestyle='--', label='Mean')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Episode Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 6: Contact count distribution
        axes[1, 1].hist(performance_300['raw_contact_counts'], bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(performance_300['mean_contact_count'], color='red', linestyle='--', label='Mean')
        axes[1, 1].set_title('Contact Count Distribution')
        axes[1, 1].set_xlabel('Final Contact Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 7: Sphere diversity distribution
        axes[1, 2].hist(performance_300['raw_sphere_diversities'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].axvline(performance_300['mean_sphere_diversity'], color='red', linestyle='--', label='Mean')
        axes[1, 2].set_title('Sphere Selection Diversity')
        axes[1, 2].set_xlabel('Selection Diversity')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 8: Selection frequency (most common selections)
        if 'most_common' in selection_patterns['selection_frequency']:
            top_selections = selection_patterns['selection_frequency']['most_common'][:10]
            if top_selections:
                sphere_ids, counts = zip(*top_selections)
                axes[1, 3].bar(range(len(sphere_ids)), counts, alpha=0.7, color='orange')
                axes[1, 3].set_title('Most Selected Spheres')
                axes[1, 3].set_xlabel('Sphere ID Rank')
                axes[1, 3].set_ylabel('Selection Count')
                axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots if directory provided
        if save_dir:
            save_path = Path(save_dir) / 'updated_sphere_selection_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()

def main():
    """Main evaluation script."""
    
    parser = argparse.ArgumentParser(description='Evaluate trained sphere selection RL model')
    parser.add_argument('model_path', help='Path to trained model (.pt file)')
    parser.add_argument('--config', help='Path to training configuration file')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--temperature', action='store_true', help='Analyze temperature effects')
    parser.add_argument('--scaling', action='store_true', help='Analyze domain scaling')
    parser.add_argument('--patterns', action='store_true', help='Analyze sphere selection patterns')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = UpdatedModelEvaluator(args.model_path, args.config)
    
    # Generate report if requested
    if args.report:
        report_path = None
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            report_path = Path(args.output) / 'evaluation_report.txt'
        
        report = evaluator.generate_comprehensive_report(report_path)
        print(report)
    
    # Generate plots if requested  
    if args.plots:
        output_dir = args.output if args.output else '.'
        evaluator.visualize_performance(output_dir)
    
    # Analyze temperature effects if requested
    if args.temperature:
        temp_results = evaluator.evaluate_temperature_effects()
        print(f"Temperature analysis complete. Diversity ranges from "
              f"{min(r['mean_sphere_diversity'] for r in temp_results.values()):.3f} to "
              f"{max(r['mean_sphere_diversity'] for r in temp_results.values()):.3f}")
    
    # Analyze domain scaling if requested
    if args.scaling:
        scaling_results = evaluator.evaluate_domain_scaling()
        print("Domain scaling analysis complete:")
        for domain_size, results in scaling_results['domain_scaling'].items():
            print(f"  {domain_size}³: reward={results['mean_reward']:.3f}, "
                  f"particle_size={results['mean_particle_size']:.3f}")
    
    # Analyze selection patterns if requested
    if args.patterns:
        patterns = evaluator.analyze_sphere_selection_patterns()
        print("Sphere selection pattern analysis complete:")
        print(f"  Selection entropy: {patterns['selection_entropy']:.3f}")
        print(f"  Mean exploration score: {patterns['exploration_vs_exploitation']['mean_exploration']:.3f}")
    
    # Run basic evaluation if no specific outputs requested
    if not any([args.report, args.plots, args.temperature, args.scaling, args.patterns]):
        performance = evaluator.evaluate_sphere_selection_performance(300, args.episodes)
        print(f"Evaluation complete. Mean particle size: {performance['mean_particle_size']:.3f}, "
              f"Sphere diversity: {performance['mean_sphere_diversity']:.3f}")

if __name__ == "__main__":
    main()
