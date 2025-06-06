"""
Enhanced Shared Memory Vectorized Environment with Policy Synchronization
Adds efficient policy sharing for sphere selection across worker processes.
"""

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Union, Any
import math
from dataclasses import dataclass
import multiprocessing as mp
import queue
import time

# Import the C++ packing generator and connectivity checker
import sys
import os

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils/connectivity/sphere_connectivity'))
from utils.connectivity.connectivity_checker import reliable_sphere_connectivity

# sys.path.append("/Volumes/Miscellanous/Deep Learning/Particle Packing project/particle-packing-generator/build/python")
import particle_packing as pp

@dataclass
class PackingConfig:
    domain_size: int = 300
    min_cores: int = 20
    
    # Core sphere parameters
    core_radius_min: int = 10
    core_radius_max: int = 20
    
    # Supplementary sphere sizes (matching the connectivity analysis radii)
    sphere_sizes: List[int] = None
    
    # Placement parameters
    n_position_sectors: int = 26  # Icosahedral + poles arrangement
    compactness_factor: float = 0.5
    
    # Task configuration
    task_config: Dict = None
    
    def __post_init__(self):
        if self.sphere_sizes is None:
            # Radii that match the connectivity analysis
            self.sphere_sizes = [2, 3, 5, 7, 9, 12]
        
        if self.task_config is None:
            self.task_config = {
                'name': 'size_contacts',
                'weights': {'target_size': 0.6, 'contacts': 0.4},
                'target_size': 35.0,  # Target average particle size
                'target_avg_contacts': 4,
                'max_steps': 800
            }

class TemperatureControlledPackingEnvironment(gym.Env):
    """Enhanced environment with local policy support for sphere selection."""
    
    def __init__(self, config: PackingConfig, initial_temperature: float = 1.0):
        super().__init__()
        self.config = config
        self.temperature = initial_temperature
        
        # Action space: [sphere_size_idx, position_sector]
        self.action_space = spaces.MultiDiscrete([
            len(config.sphere_sizes),
            config.n_position_sectors
        ])
        
        # Observation space (dummy for SB3 compatibility)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        
        # Environment state
        self.episode_step = 0
        self.previous_size_difference = 0.0
        self.initial_avg_particle_size = 0.0
        self.initial_avg_contacts = 0.0
        self.particle_count = 0
        
        # C++ PackingGenerator instance
        self.generator = None
        
        # Local policy for sphere selection (loaded from state_dict)
        self.local_policy = None
        self.local_policy_version = -1
        
        # Position sectors for sphere placement (icosahedral + poles)
        self.position_sectors = self._generate_position_sectors()
        
        # Storage for graph observations
        self._current_graph_obs = None
    
    def load_policy_from_state_dict(self, policy_state_dict: Dict, policy_config: Dict, version: int):
        """
        Load policy from state_dict for local sphere selection.
        This creates a minimal policy copy just for attention computation.
        """
        try:
            # Import policy class locally to avoid circular imports
            from updated_clean_policy import TemperatureControlledActorCritic
            
            # Create minimal policy for sphere selection only
            if self.local_policy is None:
                
                self.local_policy = TemperatureControlledActorCritic(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    lr_schedule=lambda x: 3e-4,  # Dummy lr_schedule
                    **policy_config
                )
            
            # Load the state dict
            self.local_policy.load_state_dict(policy_state_dict)
            self.local_policy.eval()  # Set to evaluation mode
            self.local_policy_version = version
            
            # Move to CPU to save memory (workers don't need GPU)
            self.local_policy.cpu()
            
        except Exception as e:
            print(f"Warning: Failed to load policy in worker: {e}")
            self.local_policy = None
    
    def set_core_attention_policy(self, policy):
        """
        Set the policy for attention-based sphere selection.
        In workers, this will be called with policy_state_dict instead of policy object.
        """
        if isinstance(policy, dict):
            # Policy state dict format: {'state_dict': {}, 'config': {}, 'version': int}
            self.load_policy_from_state_dict(
                policy['state_dict'], 
                policy.get('config', {}), 
                policy.get('version', 0)
            )
        else:
            # Direct policy object (for non-vectorized environments)
            self.local_policy = policy
    
    def set_temperature(self, temperature: float):
        """Update temperature for sphere selection."""
        self.temperature = temperature
    
    def _generate_position_sectors(self, n_sectors: int = 26) -> List[Tuple[float, float]]:
        """
        Generate n_sectors position sectors using Fibonacci spiral for optimal uniform distribution.
        
        This algorithm uses the golden ratio to create a spiral pattern that distributes points
        very uniformly across the sphere surface and is chosen because:
        - Works for ANY number of points (not just specific geometric numbers)
        - Provides better uniformity metrics
        - Much simpler implementation
        - Deterministic and reproducible
        
        Args:
            n_sectors: Number of sectors to generate (recommended: 20-40)
            
        Returns:
            List of (theta, phi) tuples in spherical coordinates
            theta: polar angle [0, π] (0 = north pole, π = south pole)
            phi: azimuthal angle [-π, π] (longitude)
        """
        sectors = []
        golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618
        
        for i in range(n_sectors):
            # Map i to y-coordinate: uniformly distributed from +1 to -1
            y = 1 - (i / (n_sectors - 1)) * 2  # Linear progression from 1 to -1
            
            # Calculate radius at this y level (from sphere equation x² + y² + z² = 1)
            radius = np.sqrt(1 - y * y)
            
            # Golden angle increment: 2π/φ ≈ 137.5° for optimal spiral distribution
            theta_spiral = 2 * np.pi * i / golden_ratio
            
            # Convert spiral to Cartesian coordinates on the circle at this y level
            x = np.cos(theta_spiral) * radius
            z = np.sin(theta_spiral) * radius
            
            # Convert Cartesian to spherical coordinates
            # Note: we already have y, and r = 1 (unit sphere)
            theta = np.arccos(y)          # Polar angle [0, π]
            phi = np.arctan2(z, x)        # Azimuthal angle [-π, π]
            
            sectors.append((theta, phi))
        
        return sectors
    
    def _create_cpp_generator(self):
        """Create C++ PackingGenerator instance."""
        self.generator = pp.PackingGenerator(
            size=self.config.domain_size,
            coreRadiusMin=self.config.core_radius_min,
            coreRadiusMax=self.config.core_radius_max,
            secondaryRadiusMin=min(self.config.sphere_sizes),
            secondaryRadiusMax=max(self.config.sphere_sizes),
            tertiaryRadiusMin=min(self.config.sphere_sizes),
            tertiaryRadiusMax=max(self.config.sphere_sizes),
            targetDensity=0.6,
            compactnessFactor=self.config.compactness_factor
        )

    def _create_graph_state(self) -> Data:
        """
        Create initial PyTorch Geometric graph representation.
        Called only once at episode reset when only core spheres are present.
        At this stage: sphereID = particleID = enumerate index = node index.
        """
        if self.generator is None:
            # Create minimal empty graph
            node_features = torch.zeros((1, 15))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, 9))
            global_features = torch.zeros(8)
            
            return Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                global_attr=global_features,
                batch=torch.zeros(1, dtype=torch.long)
            )
        
        # Get particles from C++ generator (only core spheres exist at this stage)
        particles = self.generator.getParticles()
        
        if not particles:
            return self._create_graph_state()  # Return empty graph
        
        # Build node features - sphere_id = node_idx = enumerate index
        node_features = []
        
        for sphere_id, particle in enumerate(particles):
            # Get the core sphere (only sphere present at this stage)
            core_sphere = particle.getCoreSphere()
            
            # Get particle volume from C++ generator
            particle_volume = self.generator.getParticleVolume(sphere_id)
            
            # Normalize coordinates
            center = core_sphere.getCenter()
            norm_x = center.x / self.config.domain_size
            norm_y = center.y / self.config.domain_size
            norm_z = center.z / self.config.domain_size
            norm_radius = core_sphere.getRadius() / 30.0
            
            features = [
                norm_x, norm_y, norm_z, norm_radius,
                sphere_id / len(particles),  # Normalized particle ID
                1.0,  # is_core flag (always True at this stage)
                particle_volume / 100000.0,  # Normalized particle volume
                len(particles) / 100.0,  # Normalized total particles
                self.episode_step / self.config.task_config['max_steps'],  # Episode progress
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding for future features
            ]
            
            node_features.append(features)
        
        # Convert to tensor
        node_features = torch.tensor(node_features, dtype=torch.float32)
        n_nodes = len(particles)
        
        # Create complete graph - connect all core spheres to each other
        edge_index = []
        edge_features = []
        
        # Get actual voxel-level contacts from C++ generator
        contact_pairs = self.generator.getContactPairs()
        connected_pairs = set(contact_pairs)  # Set of (sphereID, sphereID) tuples
        
        # Create edges between all pairs of nodes (complete graph)
        for sphere_id_i in range(n_nodes):
            for sphere_id_j in range(n_nodes):
                if sphere_id_i != sphere_id_j:  # No self-loops
                    edge_index.extend([[sphere_id_i, sphere_id_j], [sphere_id_j, sphere_id_i]])
                    
                    # Check if these spheres are actually voxel-wise connected
                    is_actually_connected = (
                        (sphere_id_i, sphere_id_j) in connected_pairs or 
                        (sphere_id_j, sphere_id_i) in connected_pairs
                    )
                    
                    # Get sphere objects for distance calculation
                    sphere_i = particles[sphere_id_i].getCoreSphere()
                    sphere_j = particles[sphere_id_j].getCoreSphere()
                    
                    center_i = sphere_i.getCenter()
                    center_j = sphere_j.getCenter()
                    
                    distance = math.sqrt(
                        (center_i.x - center_j.x)**2 +
                        (center_i.y - center_j.y)**2 +
                        (center_i.z - center_j.z)**2
                    )
                    
                    # Edge features: [distance, overlap, relative_position..., is_connected]
                    overlap = max(0, sphere_i.getRadius() + sphere_j.getRadius() - distance) if is_actually_connected else 0.0
                    rel_x = (center_j.x - center_i.x) / self.config.domain_size
                    rel_y = (center_j.y - center_i.y) / self.config.domain_size
                    rel_z = (center_j.z - center_i.z) / self.config.domain_size
                    
                    edge_feat = [
                        distance / self.config.domain_size,
                        overlap / 30.0,  # Normalized overlap
                        rel_x, rel_y, rel_z,
                        1.0 if is_actually_connected else 0.0,  # is_connected flag
                        0.0, 0.0, 0.0  # Padding
                    ]
                    
                    edge_features.extend([edge_feat, edge_feat])  # Bidirectional
        
        # Convert to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, 9), dtype=torch.float32)
        
        # Global features using C++ methods
        total_volume = self.generator.getTotalVolume()
        domain_volume = self.config.domain_size ** 3
        density = total_volume / domain_volume
        
        avg_particle_size = self.generator.getAverageParticleRadius()
        contact_count = self.generator.getContactCount()
        
        global_features = torch.tensor([
            density,
            avg_particle_size / 50.0,  # Normalized
            contact_count / 100.0,  # Normalized
            self.particle_count / 100.0,
            self.episode_step / self.config.task_config['max_steps'],
            0.0, 0.0, 0.0  # Padding
        ], dtype=torch.float32)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            global_attr=global_features,
            batch=torch.zeros(len(node_features), dtype=torch.long)
        )


    def _update_graph_state(self, new_sphere_id: int, particle_id: int):
        """
        Update graph state after adding a new sphere.
        Uses direct sphere_id = node_index relationship.
        """
        if self._current_graph_obs is None:
            return
        
        # Update episode progress for all nodes
        progress = self.episode_step / self.config.task_config['max_steps']
        self._current_graph_obs.x[:, 8] = progress
        
        # Update particle volumes for affected particle
        updated_volume = self.generator.getParticleVolume(particle_id)
        norm_volume = updated_volume / 100000.0
        
        # Update all nodes belonging to this particle
        particle = self.generator.getParticle(particle_id)
        for sphere in particle.getSpheres():
            # sphere_id = node_index directly
            if (sphere.getSphereId() != new_sphere_id):
                self._current_graph_obs.x[sphere.getSphereId(), 6] = norm_volume
        
        # Add new node for the newly added sphere
        # new_sphere_id IS the new node index
        new_sphere = self.generator.getSphere(new_sphere_id)

        # Create features for new sphere
        center = new_sphere.getCenter()
        norm_x = center.x / self.config.domain_size
        norm_y = center.y / self.config.domain_size
        norm_z = center.z / self.config.domain_size
        norm_radius = new_sphere.getRadius() / 30.0
        
        new_features = torch.tensor([[
            norm_x, norm_y, norm_z, norm_radius,
            particle_id / self.particle_count,
            0.0, # is_core flag (always False at this stage)
            norm_volume,
            self.particle_count / 100.0,
            progress,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]], dtype=torch.float32)
        
        # Add new node - new_sphere_id should be the next index
        self._current_graph_obs.x = torch.cat([self._current_graph_obs.x, new_features], dim=0)
        
        # Find neighbors and add edges using direct sphere_id indexing
        neighbors = self.generator.getSphereNeighbors(
            center.x, center.y, center.z, new_sphere.getRadius()
        )
        
        new_edges = []
        new_edge_features = []
        
        for neighbor_sphere_id in neighbors:
            if neighbor_sphere_id != new_sphere_id:
                # Get neighbor sphere
                neighbor_sphere = self.generator.getSphere(neighbor_sphere_id)
                
                if neighbor_sphere is not None:
                    neighbor_center = neighbor_sphere.getCenter()
                    distance = math.sqrt(
                        (center.x - neighbor_center.x)**2 +
                        (center.y - neighbor_center.y)**2 +
                        (center.z - neighbor_center.z)**2
                    )
                    
                    # Check if spheres are actually voxel-wise connected
                    is_actually_connected = reliable_sphere_connectivity(
                        new_sphere.getRadius(), 
                        neighbor_sphere.getRadius(), 
                        distance
                    )            

                    # Create edges to ALL neighbors (regardless of actual connectivity)
                    # This enables information flow in the GNN
                    new_edges.extend([
                        [new_sphere_id, neighbor_sphere_id], 
                        [neighbor_sphere_id, new_sphere_id]
                    ])
                    
                    # Edge features
                    overlap = max(0, new_sphere.getRadius() + neighbor_sphere.getRadius() - distance) if is_actually_connected else 0.0
                    rel_x = (neighbor_center.x - center.x) / self.config.domain_size
                    rel_y = (neighbor_center.y - center.y) / self.config.domain_size
                    rel_z = (neighbor_center.z - center.z) / self.config.domain_size
                    
                    edge_feat = [
                        distance / self.config.domain_size,
                        overlap / 30.0,
                        rel_x, rel_y, rel_z,
                        1.0 if is_actually_connected else 0.0, # is_connected flag
                        0.0, 0.0, 0.0
                    ]
                    
                    new_edge_features.extend([edge_feat, edge_feat])
        
        # Add new edges
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            new_edge_attr = torch.tensor(new_edge_features, dtype=torch.float32)
            
            self._current_graph_obs.edge_index = torch.cat([self._current_graph_obs.edge_index, new_edge_index], dim=1)
            self._current_graph_obs.edge_attr = torch.cat([self._current_graph_obs.edge_attr, new_edge_attr], dim=0)
        
        # Update batch tensor
        self._current_graph_obs.batch = torch.zeros(len(self._current_graph_obs.x), dtype=torch.long)

    def _select_sphere_with_attention(self, graph_obs: Data) -> int:
        """Select sphere using local policy attention mechanism with temperature control."""
        if self.local_policy is None:
            return self._select_sphere_heuristic()
        
        try:
            # Get sphere attention weights from local policy
            with torch.no_grad():
                attention_weights = self.local_policy.get_sphere_attention_weights(graph_obs)
                
                if attention_weights.numel() == 0:
                    return self._select_sphere_heuristic()
                
                # Apply temperature
                if self.temperature == 0.0:
                    # Deterministic
                    selected_sphere_id = torch.argmax(attention_weights).item()
                else:
                    # Probabilistic with temperature
                    scaled_logits = attention_weights / self.temperature
                    probs = F.softmax(scaled_logits, dim=0)
                    selected_sphere_id = torch.multinomial(probs, 1).item()
                
                return selected_sphere_id

        except Exception as e:
            print(f"Attention-based sphere selection failed: {e}")
            return self._select_sphere_heuristic()
    
    def _select_sphere_heuristic(self) -> int:
        """Fallback heuristic sphere selection."""
        if self.generator is None:
            return 0
        
        particles = self.generator.getParticles()
        if not particles:
            return 0
        
        # Select sphere from largest particle with fewest spheres
        target_sphere_id = 0
        min_sphere_count = float('inf')
        max_particle_volume = 0
        
        for sphere_id, particle in enumerate(particles):
            sphere_count = len(particle.getSpheres())
            particle_volume = self.generator.getParticleVolume(particle.getId())
            
            if (sphere_count < min_sphere_count or 
                (sphere_count == min_sphere_count and particle_volume > max_particle_volume)):
                min_sphere_count = sphere_count
                max_particle_volume = particle_volume
                # Select the core sphere of this particle
                target_sphere_id = sphere_id  # sphere_id = enumerate index
        
        return target_sphere_id
    
    def _calculate_sphere_position(self, base_sphere: pp.Sphere, sector_idx: int, 
                                 new_radius: int) -> Tuple[int, int, int]:
        """Calculate 3D position for new sphere around base sphere."""
        theta, phi = self.position_sectors[sector_idx]
        
        base_center = base_sphere.getCenter()
        base_radius = base_sphere.getRadius()
        
        # Calculate distance using compactness factor
        r_max = max(base_radius, new_radius)
        r_min = min(base_radius, new_radius)
        distance = r_max - self.config.compactness_factor * r_min
        
        # Convert spherical to Cartesian
        x = base_center.x + int(distance * math.sin(theta) * math.cos(phi))
        y = base_center.y + int(distance * math.sin(theta) * math.sin(phi))
        z = base_center.z + int(distance * math.cos(theta))
        
        return x, y, z
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state and task."""
        if self.generator is None:
            return 0.0

        particles = self.generator.getParticles()
        if not particles:
            return 0.0

        # Calculate metrics using C++ methods
        n_particles = len(particles)
        current_avg_particle_size = self.generator.getAverageParticleRadius()
        current_avg_contacts = self.generator.getContactCount() / n_particles
        
        # Task configuration
        task_config = self.config.task_config
        weights = task_config['weights']
        target_size = task_config['target_size']
        
        reward = 0.0
        
        # Size reward - encourage reaching target average particle size
        current_size_difference = abs(target_size - current_avg_particle_size) / abs(target_size - self.initial_avg_particle_size)
        size_improvement = self.previous_size_difference - current_size_difference
        size_reward = (
            10.0 * size_improvement +
            0.1 * max(0, 1.0 - current_size_difference) 
        )
        reward += weights['target_size'] * size_reward

        # Update previous values
        self.previous_size_difference = current_size_difference
        
        # Contact reward - encourage reaching target average inter-particle contacts
        target_avg_contacts = task_config['target_avg_contacts']
        contact_difference = abs(target_avg_contacts - current_avg_contacts) / abs(target_avg_contacts - self.initial_avg_contacts)

        contact_reward = 15.0 * max(0, 1.0 - contact_difference)  # Large reward for new contacts
        reward += weights['contacts'] * contact_reward
        
        # Step penalty to encourage efficiency
        step_penalty = 0.05 * (self.episode_step / task_config['max_steps'])
        reward -= step_penalty
        
        return reward
    
    def reset(self, **kwargs):
        """Reset environment for new episode."""
        self.episode_step = 0
        self.previous_size_difference = 0.0
        self.initial_avg_particle_size = 0.0
        self.initial_avg_contacts = 0.0
        
        # Create new C++ generator
        self._create_cpp_generator()
        
        # Generate core spheres using C++ method
        self.particle_count = self.generator.insertCoreSpheres()
        
        # Calculate initial average particle size
        if self.particle_count > 0:
            self.initial_avg_particle_size = self.generator.getAverageParticleRadius()
        else:
            self.initial_avg_particle_size = 0.0
        
        # Create initial graph state
        self._current_graph_obs = self._create_graph_state()
        
        info = {
            'n_cores': self.particle_count,
            'episode_step': self.episode_step,
            'initial_avg_radius': self.initial_avg_particle_size
        }
        
        # Return dummy observation for SB3 compatibility
        return np.array([0.0], dtype=np.float32), info
    
    def step(self, action):
        """Execute one step in the environment."""
        if isinstance(action, (int, np.integer)):
            sphere_size_idx = action % len(self.config.sphere_sizes)
            position_sector = action // len(self.config.sphere_sizes)
        else:
            sphere_size_idx, position_sector = action
        
        sphere_radius = self.config.sphere_sizes[sphere_size_idx]
        position_sector = position_sector % self.config.n_position_sectors
        
        self.episode_step += 1
        
        if self.generator is None:
            return np.array([0.0], dtype=np.float32), 0.0, True, False, {}
        
        particles = self.generator.getParticles()
        if not particles:
            return np.array([0.0], dtype=np.float32), 0.0, True, False, {}
        
        # Select target sphere using attention or heuristic
        target_sphere_id = self._select_sphere_with_attention(self._current_graph_obs)
        
        # Find the target sphere using direct sphere_id
        target_sphere = self.generator.getSphere(target_sphere_id)
        target_particle_id = target_sphere.getParticleId()
        
        if target_sphere is None:
            # Fallback: select first available sphere
            for particle in particles:
                spheres = particle.getSpheres()
                if spheres:
                    target_sphere = spheres[0]
                    target_particle_id = particle.getId()
                    break
        
        if target_sphere is None:
            return np.array([0.0], dtype=np.float32), -0.1, True, False, {}
        
        # Calculate new sphere position
        x, y, z = self._calculate_sphere_position(target_sphere, position_sector, sphere_radius)
        
        # Check bounds
        if (x - sphere_radius < 0 or x + sphere_radius >= self.config.domain_size or
            y - sphere_radius < 0 or y + sphere_radius >= self.config.domain_size or
            z - sphere_radius < 0 or z + sphere_radius >= self.config.domain_size):
            # Invalid placement - penalty
            reward = -0.1
            terminated = False
        else:
            # Try to add sphere using C++ method
            success = self.generator.insertSuppSpheres(x, y, z, sphere_radius, target_particle_id)
            
            if success:
                # Get the new sphere ID (will be the latest one added)
                new_sphere_id = self.generator.getSphereCount() - 1
                
                # Update graph state
                if new_sphere_id is not None:
                    self._update_graph_state(new_sphere_id, target_particle_id)
                
                # Calculate reward
                reward = self._calculate_reward()
                
                # Check termination
                current_avg_size = self.generator.getAverageParticleRadius()
                terminated = current_avg_size >= self.config.task_config['target_size']
            else:
                # Failed to add sphere
                reward = -0.1
                terminated = False
        
        truncated = self.episode_step >= self.config.task_config['max_steps']
        
        info = {
            'episode_step': self.episode_step,
            'n_particles': len(self.generator.getParticles()) if self.generator else 0,
            'avg_particle_size': self.generator.getAverageParticleRadius() if self.generator else 0,
            'contact_count': self.generator.getContactCount() if self.generator else 0,
            'selected_sphere_id': target_sphere_id,
            'temperature': self.temperature,
            'policy_version': self.local_policy_version
        }
        
        # Return dummy observation for SB3 compatibility
        return np.array([0.0], dtype=np.float32), reward, terminated, truncated, info
    
    def get_current_graph_observation(self) -> Data:
        """Get the current graph observation."""
        return self._current_graph_obs

"""
Shared Memory Vectorized Environment
Combines persistent workers with shared memory for graph data.
"""
class SharedMemoryManager:
    """Manages shared memory tensors for graph data"""
    
    def __init__(self, n_envs: int, max_nodes_per_env: int = 200, max_edges_per_env: int = 100000):
        self.n_envs = n_envs
        self.max_nodes_per_env = max_nodes_per_env
        self.max_edges_per_env = max_edges_per_env
        
        # Create shared tensors for each environment
        self.shared_data = {}
        
        for env_id in range(n_envs):
            self.shared_data[env_id] = {
                # Node features: (max_nodes, 15)
                'node_features': torch.zeros(max_nodes_per_env, 15).share_memory_(),
                'node_count': torch.zeros(1, dtype=torch.long).share_memory_(),
                
                # Edge data: (2, max_edges) and (max_edges, 9)
                'edge_index': torch.zeros(2, max_edges_per_env, dtype=torch.long).share_memory_(),
                'edge_features': torch.zeros(max_edges_per_env, 9).share_memory_(),
                'edge_count': torch.zeros(1, dtype=torch.long).share_memory_(),
                
                # Global features: (8,)
                'global_features': torch.zeros(8).share_memory_(),
                
                # Flags for synchronization
                'data_ready': torch.zeros(1, dtype=torch.bool).share_memory_(),
                'data_version': torch.zeros(1, dtype=torch.long).share_memory_()
            }
    
    def get_shared_tensors(self, env_id: int):
        """Get shared tensors for a specific environment"""
        return self.shared_data[env_id]
    
    def read_graph_data(self, env_id: int) -> Data:
        """Read graph data from shared memory and create PyTorch Geometric Data object"""
        shared = self.shared_data[env_id]
        
        # Wait for data to be ready (simple spinlock)
        timeout = 1.0
        start_time = time.time()
        while not shared['data_ready'].item() and (time.time() - start_time) < timeout:
            time.sleep(0.001)
        
        if not shared['data_ready'].item():
            # Return empty graph if timeout
            return Data(
                x=torch.zeros(1, 15),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 9),
                global_attr=torch.zeros(8)
            )
        
        # Extract actual data (slice to real sizes)
        n_nodes = shared['node_count'].item()
        n_edges = shared['edge_count'].item()
        
        node_features = shared['node_features'][:n_nodes].clone()
        edge_index = shared['edge_index'][:, :n_edges].clone()
        edge_features = shared['edge_features'][:n_edges].clone()
        global_features = shared['global_features'].clone()
        
        # Reset ready flag
        shared['data_ready'].fill_(False)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            global_attr=global_features
        )

class SharedMemoryWorker:
    """Enhanced worker with policy synchronization support"""
    
    def __init__(self, env_config: PackingConfig, worker_id: int, 
                 command_queue: mp.Queue, result_queue: mp.Queue,
                 shared_tensors: Dict[str, torch.Tensor]):
        self.env_config = env_config
        self.worker_id = worker_id
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.shared_tensors = shared_tensors
        self.env = None
        self.current_policy_version = -1
    
    def run(self):
        """Main worker loop with policy synchronization"""
        try:
            # Create environment once
            self.env = TemperatureControlledPackingEnvironment(
                self.env_config, initial_temperature=0.1
            )
            
            # Signal ready
            self.result_queue.put(('ready', self.worker_id))
            
            while True:
                try:
                    command, data = self.command_queue.get(timeout=1.0)
                    
                    if command == 'reset':
                        obs, info = self.env.reset()
                        self._write_graph_to_shared_memory()
                        self.result_queue.put(('success', (obs, info)))
                        
                    elif command == 'step':
                        action = data
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self._write_graph_to_shared_memory()
                        self.result_queue.put(('success', (obs, reward, terminated, truncated, info)))
                        
                    elif command == 'set_temperature':
                        temperature = data
                        self.env.set_temperature(temperature)
                        self.result_queue.put(('success', None))
                        
                    elif command == 'set_policy':
                        policy_data = data
                        self.env.set_core_attention_policy(policy_data)
                        self.current_policy_version = policy_data.get('version', 0)
                        self.result_queue.put(('success', self.current_policy_version))
                        
                    elif command == 'get_policy_version':
                        self.result_queue.put(('success', self.current_policy_version))
                        
                    elif command == 'get_spaces':
                        action_space = self.env.action_space
                        observation_space = self.env.observation_space
                        self.result_queue.put(('success', (action_space, observation_space)))
                        
                    elif command == 'shutdown':
                        break
                        
                except queue.Empty:
                    continue
                    
        except Exception as e:
            self.result_queue.put(('error', str(e)))
    
    def _write_graph_to_shared_memory(self):
        """Write current graph state to shared memory"""
        try:
            # Get graph observation from environment
            graph_obs = self.env.get_current_graph_observation()
            if graph_obs is None:
                return
            
            # Extract tensor data
            node_features = graph_obs.x  # Shape: (n_nodes, 15)
            edge_index = graph_obs.edge_index  # Shape: (2, n_edges)
            edge_features = graph_obs.edge_attr  # Shape: (n_edges, 9)
            global_features = graph_obs.global_attr  # Shape: (8,)
            
            # Get sizes
            n_nodes = node_features.shape[0]
            n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            
            # Check bounds
            max_nodes = self.shared_tensors['node_features'].shape[0]
            max_edges = self.shared_tensors['edge_features'].shape[0]
            
            if n_nodes > max_nodes or n_edges > max_edges:
                print(f"Warning: Graph too large (nodes: {n_nodes}/{max_nodes}, edges: {n_edges}/{max_edges})")
                n_nodes = min(n_nodes, max_nodes)
                n_edges = min(n_edges, max_edges)
            
            # Write to shared memory (direct tensor copy - very fast!)
            self.shared_tensors['node_features'][:n_nodes] = node_features[:n_nodes]
            self.shared_tensors['node_count'].fill_(n_nodes)
            
            if n_edges > 0:
                self.shared_tensors['edge_index'][:, :n_edges] = edge_index[:, :n_edges]
                self.shared_tensors['edge_features'][:n_edges] = edge_features[:n_edges]
            self.shared_tensors['edge_count'].fill_(n_edges)
            
            self.shared_tensors['global_features'][:] = global_features
            
            # Increment version and mark ready
            self.shared_tensors['data_version'] += 1
            self.shared_tensors['data_ready'].fill_(True)
            
        except Exception as e:
            print(f"Error writing to shared memory: {e}")

def start_worker(env_config, worker_id, command_queue, result_queue, shared_tensors):
    """Start function for policy-enabled shared memory worker"""
    worker = SharedMemoryWorker(env_config, worker_id, command_queue, result_queue, shared_tensors)
    worker.run()

class VectorizedPackingEnvironment:
    """
    Enhanced vectorized environment with policy synchronization.
    Now supports efficient policy sharing across worker processes.
    """
    
    def __init__(self, env_configs: List[PackingConfig], n_envs: int = None,
                 max_nodes_per_env: int = 200, max_edges_per_env: int = 100000):
        if n_envs is None:
            n_envs = len(env_configs)
        
        self.n_envs = n_envs
        self.env_configs = env_configs[:n_envs]
        
        # Set up shared memory manager
        self.shared_memory = SharedMemoryManager(n_envs, max_nodes_per_env, max_edges_per_env)
        
        # Create communication queues
        self.command_queues = [mp.Queue() for _ in range(n_envs)]
        self.result_queues = [mp.Queue() for _ in range(n_envs)]
        
        # Start worker processes with shared memory access
        self.workers = []
        for i in range(n_envs):
            shared_tensors = self.shared_memory.get_shared_tensors(i)
            worker = mp.Process(
                target=start_worker,
                args=(self.env_configs[i], i, self.command_queues[i], 
                      self.result_queues[i], shared_tensors)
            )
            worker.start()
            self.workers.append(worker)
        
        # Wait for workers to be ready
        self._wait_for_workers_ready()
        
        # Get action/observation spaces
        self.command_queues[0].put(('get_spaces', None))
        status, (action_space, observation_space) = self.result_queues[0].get(timeout=10.0)
        if status != 'success':
            raise RuntimeError("Failed to get action/observation spaces")
        
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Policy synchronization state
        self.current_policy_version = 0
        self.policy_config = {}
        
        print(f"Policy-enabled vectorized environment: {n_envs} workers ready")
    
    def set_core_attention_policy(self, policy):
        """
        Set the policy for all workers by sending state_dict through queues.
        This is called when the policy is updated during training.
        """
        try:
            # Extract policy state dict and configuration
            policy_state_dict = policy.state_dict()
            
            # Get policy configuration for reconstruction in workers
            self.policy_config = {
                'features_extractor_class': policy.features_extractor_class,
                'features_extractor_kwargs': policy.features_extractor_kwargs,
            }
            
            # Increment version
            self.current_policy_version += 1
            
            # Package policy data for workers
            policy_data = {
                'state_dict': policy_state_dict,
                'config': self.policy_config,
                'version': self.current_policy_version
            }
            
            # Send to all workers
            commands = [('set_policy', policy_data) for _ in range(self.n_envs)]
            results = self._send_commands_and_collect_results(commands, timeout=30.0)
            
            # Verify all workers updated successfully
            for i, version in enumerate(results):
                if version != self.current_policy_version:
                    print(f"Warning: Worker {i} failed to update policy (version {version} != {self.current_policy_version})") #Amir: Does it really check the policy update was susccessful?
            
            print(f"Policy synchronized to all workers (version {self.current_policy_version})")
            
        except Exception as e:
            print(f"Error synchronizing policy: {e}")
            raise
    
    def set_temperature(self, temperature: float):
        """Set temperature for all environments"""
        commands = [('set_temperature', temperature) for _ in range(self.n_envs)]
        self._send_commands_and_collect_results(commands)
    
    def get_policy_versions(self) -> List[int]:
        """Get current policy versions from all workers"""
        commands = [('get_policy_version', None) for _ in range(self.n_envs)]
        return self._send_commands_and_collect_results(commands)
    
    def _wait_for_workers_ready(self, timeout: float = 30.0):
        """Wait for all workers to initialize"""
        print("Waiting for policy-enabled workers to initialize...")
        
        ready_workers = set()
        start_time = time.time()
        
        while len(ready_workers) < self.n_envs:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Workers failed to initialize within {timeout}s")
            
            for i in range(self.n_envs):
                if i in ready_workers:
                    continue
                    
                try:
                    status, worker_id = self.result_queues[i].get_nowait()
                    if status == 'ready':
                        ready_workers.add(i)
                        print(f"Worker {i} ready ({len(ready_workers)}/{self.n_envs})")
                    elif status == 'error':
                        raise RuntimeError(f"Worker {i} failed: {worker_id}")
                        
                except queue.Empty:
                    continue
                    
            time.sleep(0.1)
        
        print("All policy-enabled workers ready!")
    
    def _send_commands_and_collect_results(self, commands_data: List[tuple], timeout: float = 60.0):
        """Send commands and collect results"""
        # Send commands
        for i, (command, data) in enumerate(commands_data):
            self.command_queues[i].put((command, data))
        
        # Collect results
        results = []
        for i in range(self.n_envs):
            try:
                status, result = self.result_queues[i].get(timeout=timeout)
                if status == 'error':
                    raise RuntimeError(f"Worker {i} error: {result}")
                results.append(result)
            except queue.Empty:
                raise TimeoutError(f"Worker {i} timeout after {timeout}s")
        
        return results
    
    def reset(self, **kwargs):
        """Reset all environments"""
        commands = [('reset', None) for _ in range(self.n_envs)]
        results = self._send_commands_and_collect_results(commands)
        
        observations = []
        infos = []
        
        for i, (obs, info) in enumerate(results):
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos
    
    def step(self, actions):
        """Step all environments"""
        commands = [('step', actions[i]) for i in range(self.n_envs)]
        results = self._send_commands_and_collect_results(commands)
        
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (np.array(observations), np.array(rewards), 
                np.array(terminateds), np.array(truncateds), infos)
    
    def get_batch_graph_observations(self) -> Batch:
        """Get batched graph observations from shared memory (FAST!)"""
        graphs = []
        
        for i in range(self.n_envs):
            graph_data = self.shared_memory.read_graph_data(i)
            graphs.append(graph_data)
        
        return Batch.from_data_list(graphs)
    
    def close(self):
        """Clean up workers and shared memory"""
        # Send shutdown commands
        for queue in self.command_queues:
            queue.put(('shutdown', None))
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        print("Policy-enabled vectorized environment closed")
    
    def __del__(self):
        self.close()