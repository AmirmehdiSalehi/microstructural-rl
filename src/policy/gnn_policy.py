"""
Complete GNN Policy with Temperature-Controlled Sphere Selection
Updated implementation with any-sphere selection and vectorization support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

class TemperatureControlledGraphExtractor(BaseFeaturesExtractor):
    """
    Graph Neural Network feature extractor with temperature-controlled sphere selection.
    Updated to select any sphere, not just cores.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Space,
        features_dim: int = 256,
        n_gnn_layers: int = 3,
        hidden_dim: int = 128,
        n_attention_heads: int = 4
    ):
        super().__init__(observation_space, features_dim)
        
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.n_attention_heads = n_attention_heads
        
        # Updated input dimensions based on new features
        self.node_input_dim = 15  # Reduced from 16, removed sphere_count
        self.edge_input_dim = 9   # Increased from 8, added connectivity feature
        self.global_input_dim = 8  # Reduced from 10
        
        # Node and edge embedding layers
        self.node_embedding = nn.Linear(self.node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_input_dim, hidden_dim)
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_gnn_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim, 
                    hidden_dim // n_attention_heads, 
                    heads=n_attention_heads,
                    dropout=0.1,
                    edge_dim=hidden_dim,
                    concat=True  # Concatenate attention heads
                )
            )
        
        # Global feature processing
        self.global_embedding = nn.Linear(self.global_input_dim, hidden_dim)
        
        # Final feature combination 
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, features_dim), # Graph + global features
            nn.ReLU(), 
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        # Sphere attention mechanism for action selection (any sphere, not just cores)
        self.sphere_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Store processed features for sphere selection
        self._last_node_features = None
        self._last_data = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        This method is required by BaseFeaturesExtractor.
        """
        # observations is a dummy tensor from the environment
        # We need to get the actual graph data from the environment
        
        # Create a dummy output for now - this will be overridden by the training loop
        # that provides the actual graph data via process_graph_observation
        batch_size = observations.shape[0] if observations.dim() > 1 else 1
        return torch.zeros(batch_size, self.features_dim, device=observations.device)
    
    def process_graph_observation(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Process actual graph observation and return features.
        This is called by the training loop with real graph data.
        Supports both single graphs and batched graphs.
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        batch = graph_data.batch if hasattr(graph_data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_attr = graph_data.global_attr
        
        # Process node features
        x = self.node_embedding(x)
        
        # Process edge features if available
        edge_attr_processed = None
        if edge_attr.size(0) > 0:
            edge_attr_processed = self.edge_embedding(edge_attr)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x_new = gat_layer(x, edge_index, edge_attr=edge_attr_processed)
            x = F.relu(x_new)
            x = F.dropout(x, training=self.training)
        
        # Store node features and data for sphere attention
        self._last_node_features = x
        self._last_data = graph_data
        
        # Global pooling for graph-level features
        if isinstance(graph_data, Batch):
            graph_features = global_mean_pool(x, batch)
        else:
            graph_features = x.mean(dim=0, keepdim=True)
        # Process global features
        if isinstance(graph_data, Batch):
            # For batched data, global features might be stacked
            n_graphs = graph_data.num_graphs

            if global_attr.dim() == 1:
                global_attr = global_attr.view(n_graphs, 8)
            global_features = self.global_embedding(global_attr)
        else:
            if global_attr.dim() == 1:
                global_attr = global_attr.unsqueeze(0)
            global_features = self.global_embedding(global_attr)
        # Combine graph and global features
        combined_features = torch.cat([graph_features, global_features], dim=-1)
        features = self.feature_combiner(combined_features)
        
        return features
    
    def get_sphere_attention_weights(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get attention weights for sphere selection (any sphere, not just cores).
        """
        # Ensure we have processed the graph
        if (self._last_node_features is None or 
            self._last_data is None or 
            not self._data_equal(self._last_data, graph_data)):
            # Process the graph to get node features
            _ = self.process_graph_observation(graph_data)
        
        # Get node features
        x = self._last_node_features
        
        if x.size(0) == 0:
            # No spheres found
            return torch.tensor([], device=x.device)
        
        # Calculate attention logits for all spheres
        attention_logits = self.sphere_attention(x).squeeze(-1)
        
        return attention_logits
    
    def _data_equal(self, data1: Union[Data, Batch], data2: Union[Data, Batch]) -> bool:
        """Check if two graph data objects are the same."""
        try:
            if type(data1) != type(data2):
                return False
            
            if isinstance(data1, Batch):
                return (torch.equal(data1.x, data2.x) and 
                       torch.equal(data1.edge_index, data2.edge_index) and
                       torch.equal(data1.batch, data2.batch))
            else:
                return (torch.equal(data1.x, data2.x) and 
                       torch.equal(data1.edge_index, data2.edge_index))
        except:
            return False

class TemperatureControlledActorCritic(BasePolicy):
    """
    Actor-Critic policy with temperature-controlled sphere selection.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = TemperatureControlledGraphExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        self.observation_space = observation_space
        # Initialize features extractor properly
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        self.features_extractor = self.make_features_extractor()

        self.features_dim = self.features_extractor.features_dim
        
        # Build policy and value networks
        self._build_networks()
        
        # Initialize optimizer with proper learning rate
        self.lr_schedule = lr_schedule
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(self.observation_space)
        

    def _build_networks(self):
        """Build actor and critic networks."""
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Action heads - separate heads for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(256, self.action_space.nvec[0]),  # Sphere size selection
            nn.Linear(256, self.action_space.nvec[1])   # Position sector selection  
        ])
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get parameters for reconstructing the policy."""
        data = super()._get_constructor_parameters()
        data.update(dict(
            lr_schedule=self.lr_schedule,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs
        ))
        return data
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.
        """
        # Extract features (dummy forward pass)
        features = self.extract_features(obs)
        
        # Get action distributions
        actions, log_probs = self._get_action_dist_from_latent(features, deterministic)
        
        # Get value estimate
        values = self.value_net(features)
        
        return actions, values, log_probs
    
    def forward_with_graph(self, graph_data: Union[Data, Batch], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with actual graph data.
        Supports both single graphs and batched graphs.
        """
        # Extract features from graph
        features = self.features_extractor.process_graph_observation(graph_data)
        
        # Get action distributions
        actions, log_probs = self._get_action_dist_from_latent(features, deterministic)
        
        # Get value estimate
        values = self.value_net(features)
        
        return actions, values, log_probs
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution from latent policy features.
        """
        # Process through policy network
        policy_features = self.policy_net(latent_pi)
        
        # Get logits for each action dimension
        action_logits = [head(policy_features) for head in self.action_heads]
        
        # Sample actions
        if deterministic:
            actions = [torch.argmax(logits, dim=-1) for logits in action_logits]
        else:
            actions = [torch.distributions.Categorical(logits=logits).sample() for logits in action_logits]
        
        # Calculate log probabilities
        log_probs = []
        for i, (action, logits) in enumerate(zip(actions, action_logits)):
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(action))
        
        # Combine actions and log probs
        combined_actions = torch.stack(actions, dim=-1)
        combined_log_probs = sum(log_probs)
        
        return combined_actions, combined_log_probs
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get value estimate
        values = self.value_net(features).squeeze(-1)
        
        # Process through policy network
        policy_features = self.policy_net(features)
        
        # Get logits for each action dimension
        action_logits = [head(policy_features) for head in self.action_heads]
        
        # Calculate log probabilities and entropy
        log_probs = []
        entropies = []
        
        for i, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            action_dim = actions[:, i] if actions.dim() > 1 else actions
            log_probs.append(dist.log_prob(action_dim))
            entropies.append(dist.entropy())
        
        # Combine log probs and entropy
        combined_log_probs = sum(log_probs)
        combined_entropy = sum(entropies)
        
        return values, combined_log_probs, combined_entropy
    
    def evaluate_actions_with_graph(self, graph_data: Union[Data, Batch], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions with actual graph data.
        Supports both single graphs and batched graphs.
        """
        # Extract features from graph
        features = self.features_extractor.process_graph_observation(graph_data)
        
        # Get value estimate
        values = self.value_net(features).squeeze(-1)
        
        # Process through policy network
        policy_features = self.policy_net(features)
        
        # Get logits for each action dimension
        action_logits = [head(policy_features) for head in self.action_heads]
        
        # Calculate log probabilities and entropy
        log_probs = []
        entropies = []
        
        for i, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            action_dim = actions[:, i] if actions.dim() > 1 else actions
            log_probs.append(dist.log_prob(action_dim))
            entropies.append(dist.entropy())
        
        # Combine log probs and entropy
        combined_log_probs = sum(log_probs)
        combined_entropy = sum(entropies)
        
        return values, combined_log_probs, combined_entropy
    
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict values for given observations.
        """
        features = self.extract_features(obs)
        return self.value_net(features).squeeze(-1)
    
    def predict_values_with_graph(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Predict values with actual graph data.
        """
        features = self.features_extractor.process_graph_observation(graph_data)
        return self.value_net(features).squeeze(-1)
    
    
    def get_sphere_attention_weights(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get attention weights for sphere selection (any sphere).
        """
        return self.features_extractor.get_sphere_attention_weights(graph_data)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Required abstract method for BasePolicy.
        Get the action according to the policy for a given observation.
        """
        # Extract features
        features = self.extract_features(observation)
        
        # Get action distributions
        actions, _ = self._get_action_dist_from_latent(features, deterministic)
        
        return actions
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes state information which is not used here.
        """
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation, dtype=torch.float32)
        
        # Get actions from the policy
        with torch.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        
        return actions.cpu().numpy(), state


# Custom PPO wrapper for graph observations with vectorization support

class GraphPPO:
    """
    Fixed PPO implementation that caches processed features during rollout.
    """
    
    def __init__(self, env, policy_class=TemperatureControlledActorCritic, **ppo_kwargs):
        self.env = env
        self.is_vectorized = hasattr(env, 'n_envs')
        
        self.policy = policy_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda x: ppo_kwargs.get('learning_rate', 3e-4)
        )
        
        # PPO hyperparameters
        self.n_steps = ppo_kwargs.get('n_steps', 512)
        self.batch_size = ppo_kwargs.get('batch_size', 64)
        self.n_epochs = ppo_kwargs.get('n_epochs', 10)
        self.gamma = ppo_kwargs.get('gamma', 0.99)
        self.gae_lambda = ppo_kwargs.get('gae_lambda', 0.95)
        self.clip_range = ppo_kwargs.get('clip_range', 0.2)
        self.ent_coef = ppo_kwargs.get('ent_coef', 0.01)
        self.vf_coef = ppo_kwargs.get('vf_coef', 0.5)
        self.max_grad_norm = ppo_kwargs.get('max_grad_norm', 0.5)
        
        # Connect policy to environment for sphere selection
        if self.is_vectorized:
            self.env.set_core_attention_policy(self.policy)
        else:
            self.env.set_core_attention_policy(self.policy)
        
        self.total_timesteps = 0
    
    def _collect_rollout(self):
        """
        FIXED: Collect rollout and cache processed features to avoid reprocessing.
        """
        # Storage for rollout data
        observations = []
        processed_features = []  # Cache processed features instead of raw graphs
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
            
        for step in range(self.n_steps):
            # Get graph observation
            if self.is_vectorized:
                batch_graph_obs = self.env.get_batch_graph_observations()
            else:
                batch_graph_obs = self.env.get_current_graph_observation()
            
            # Process graph observation ONCE and cache the features
            with torch.no_grad():
                features = self.policy.features_extractor.process_graph_observation(batch_graph_obs)
                
                # Get action from processed features (avoiding reprocessing)
                policy_features = self.policy.policy_net(features)
                action_logits = [head(policy_features) for head in self.policy.action_heads]
                
                # Sample actions
                actions_list = [torch.distributions.Categorical(logits=logits).sample() 
                               for logits in action_logits]
                action = torch.stack(actions_list, dim=-1)
                
                # Calculate log probabilities
                log_probs_list = []
                for i, (act, logits) in enumerate(zip(actions_list, action_logits)):
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs_list.append(dist.log_prob(act))
                log_prob = sum(log_probs_list)
                
                # Get value from processed features
                value = self.policy.value_net(features).squeeze(-1)
            
            # Take step in environment
            if self.is_vectorized:
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
            else:
                obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
                done = terminated or truncated
            
            # Store processed features instead of raw graphs
            observations.append(obs)
            processed_features.append(features)  # Cache processed features!
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)
            
            # Handle environment resets
            if self.is_vectorized:
                if np.any(done):
                    obs, _ = self.env.reset()
            else:
                if done:
                    obs, _ = self.env.reset()
        
        return {
            'observations': observations,
            'processed_features': processed_features,  # Use cached features
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'dones': torch.tensor(dones, dtype=torch.float32),
            'values': torch.stack(values),
            'log_probs': torch.stack(log_probs)
        }
    
    def _update_policy(self, rollout_data):
        """
        FIXED: Update policy using cached processed features.
        """
        # Calculate advantages using GAE
        advantages = self._calculate_gae(rollout_data)
        returns = advantages + rollout_data['values']
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Stack processed features for efficient batching
        all_features = torch.stack(rollout_data['processed_features'])  # Shape: [n_steps, batch_size, features_dim]
        
        # Flatten for mini-batch processing
        if self.is_vectorized:
            # For vectorized environments, flatten across both steps and environments
            n_steps, n_envs = all_features.shape[:2]
            all_features = all_features.view(n_steps * n_envs, -1)
            all_actions = rollout_data['actions'].view(n_steps * n_envs, -1)
            all_advantages = advantages.view(-1)
            all_returns = returns.view(-1)
            all_old_log_probs = rollout_data['log_probs'].view(-1)
        else:
            # For single environment, just flatten steps
            n_steps = all_features.shape[0]
            all_features = all_features.view(n_steps, -1)
            all_actions = rollout_data['actions']
            all_advantages = advantages
            all_returns = returns
            all_old_log_probs = rollout_data['log_probs']
        
        dataset_size = len(all_actions)
        indices = np.arange(dataset_size)
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch data using cached features
                batch_features = all_features[batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_old_log_probs = all_old_log_probs[batch_indices]
                
                # Calculate losses using cached features
                policy_loss = self._calculate_policy_loss_from_features(
                    batch_features, batch_actions, batch_advantages, batch_old_log_probs
                )
                value_loss = self._calculate_value_loss_from_features(batch_features, batch_returns)
                
                total_loss = policy_loss + self.vf_coef * value_loss
                
                # Update parameters
                self.policy.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
    
    def _calculate_policy_loss_from_features(self, features, actions, advantages, old_log_probs):
        """
        FIXED: Calculate policy loss from pre-processed features.
        """
        # Get policy outputs from features
        policy_features = self.policy.policy_net(features)
        action_logits = [head(policy_features) for head in self.policy.action_heads]
        
        # Calculate log probabilities and entropy
        log_probs = []
        entropies = []
        
        for i, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            action_dim = actions[:, i] if actions.dim() > 1 else actions
            log_probs.append(dist.log_prob(action_dim))
            entropies.append(dist.entropy())
        
        combined_log_probs = sum(log_probs)
        combined_entropy = sum(entropies)
        
        # PPO loss calculation
        ratio = torch.exp(combined_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.ent_coef * combined_entropy.mean()
        
        return policy_loss + entropy_loss
    
    def _calculate_value_loss_from_features(self, features, returns):
        """
        FIXED: Calculate value loss from pre-processed features.
        """
        values = self.policy.value_net(features).squeeze(-1)
        return F.mse_loss(values, returns)
    
    def _calculate_gae(self, rollout_data):
        """Calculate Generalized Advantage Estimation."""
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        if self.is_vectorized:
            # For vectorized environments, GAE calculation is more complex
            n_steps, n_envs = rewards.shape
            for env_idx in range(n_envs):
                last_gae = 0
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:
                        next_value = 0
                    else:
                        next_value = values[t + 1, env_idx]
                    
                    delta = (rewards[t, env_idx] + 
                            self.gamma * next_value * (1 - dones[t, env_idx]) - 
                            values[t, env_idx])
                    advantages[t, env_idx] = last_gae = (
                        delta + self.gamma * self.gae_lambda * (1 - dones[t, env_idx]) * last_gae
                    )
        else:
            # Single environment GAE
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                advantages[t] = last_gae = (
                    delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
                )
        
        return advantages
    
    def predict(self, observation, deterministic=False):
        """Predict action for given observation."""
        if self.is_vectorized:
            batch_graph_obs = self.env.get_batch_graph_observations()
        else:
            batch_graph_obs = self.env.get_current_graph_observation()
        
        with torch.no_grad():
            actions, values, log_probs = self.policy.forward_with_graph(batch_graph_obs, deterministic)
            
        return actions.cpu().numpy(), None
    
    def learn(self, total_timesteps, callback=None):
        """
        Learn using PPO algorithm with vectorized environments.
        """
        steps_completed = 0
        
        while steps_completed < total_timesteps:
            # Collect rollout
            rollout_data = self._collect_rollout()
            
            # Update policy
            self._update_policy(rollout_data)
            
            steps_completed += self.n_steps
            self.total_timesteps += self.n_steps
            
            # Call callback if provided
            if callback:
                callback.on_step()
                
            if steps_completed % 10000 == 0:
                print(f"Steps completed: {steps_completed}/{total_timesteps}")

    
    def save(self, path):
        """Save the model."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        """Load the model."""
        self.policy.load_state_dict(torch.load(path))
    
    
