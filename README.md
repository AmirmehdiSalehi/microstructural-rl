# Vectorized Sphere Selection Particle Packing RL

A sophisticated reinforcement learning framework for generating optimized microstructures through intelligent particle assembly and arrangement. While specifically implemented for non-spherical particle packing using temperature-controlled sphere selection and vectorized training environments, this system represents a powerful foundation for **microstructural design in materials science**. The framework's graph-based representation and reinforcement learning approach enable the design of complex material microstructures with targeted properties - from porous media and composite materials to cellular structures and metamaterials. By learning optimal spatial arrangements and connectivity patterns, this tool can guide the creation of materials with specific mechanical, thermal, electrical, or transport properties, making it invaluable for applications ranging from battery electrode design and catalyst optimization to structural composites and biomaterial scaffolds. This project combines high-performance C++ simulation with modern Python RL frameworks and Graph Neural Networks to create a versatile platform for computational materials design.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Key Features

- **Any-Sphere Selection**: Agent selects from ANY sphere (core or supplementary) rather than just cores
- **Temperature-Controlled Exploration**: Attention-based sphere selection with temperature scheduling
- **Vectorized Training**: 32+ parallel environments for 5-10x training speedup
- **Reliable Connectivity**: 100% accurate sphere connectivity using analytical + lookup table approach
- **Graph Neural Networks**: PyTorch Geometric-based spatial relationship modeling
- **C++ Integration**: High-performance voxel-level particle tracking and collision detection
- **HPC Ready**: Optimized for cluster deployment with SLURM support

## 📊 Performance

- **Training Speed**: 5-10x faster with vectorized environments
- **Accuracy**: 100% reliable connectivity detection
- **Scalability**: Tested on 200+ core HPC systems
- **Memory Efficiency**: Chunk-based voxel storage with efficient graph updates

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Python RL Framework                      │
├─────────────────────────────────────────────────────────────┤
│  VectorizedPackingEnvironment (32+ parallel environments)  │
│  ├─ TemperatureControlledPackingEnvironment               │
│  ├─ Graph State Management                                │
│  └─ Temperature Scheduling                                │
├─────────────────────────────────────────────────────────────┤
│  TemperatureControlledActorCritic Policy                   │
│  ├─ GNN Feature Extractor (GAT layers)                    │
│  ├─ Sphere Attention Mechanism                           │
│  └─ Action Heads (radius + position)                     │
├─────────────────────────────────────────────────────────────┤
│  Sphere Connectivity Analysis Module                       │
│  ├─ Analytical Connectivity (fast path)                   │
│  └─ Lookup Table (edge cases)                            │
└─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Python Bindings    │
                    │     (pybind11)       │
                    └───────────┬───────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   C++ Packing Engine                       │
├─────────────────────────────────────────────────────────────┤
│  PackingGenerator (Main orchestrator)                      │
│  ├─ 3-Stage RSA Algorithm                                 │
│  ├─ Spatial R-tree Indexing                              │
│  └─ Contact Detection                                     │
├─────────────────────────────────────────────────────────────┤
│  VoxelGrid (Memory management)                             │
│  ├─ Chunk-based Storage                                   │
│  ├─ 16-bit Voxel Encoding                                │
│  └─ Surface/Bulk Classification                          │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- C++ compiler (GCC 11+ or Clang 12+)
- CMake 3.18+
- CUDA (optional, for GPU acceleration)


## 🚦 Quick Start

### Basic Usage

```python
from src.environment import TemperatureControlledPackingEnvironment, PackingConfig
from src.policy import GraphPPO

# Create environment
config = PackingConfig(domain_size=300, min_cores=20)
env = TemperatureControlledPackingEnvironment(config, initial_temperature=1.0)

# Create and train model
model = GraphPPO(env)

# Reset and step
obs, info = env.reset()
action, _ = model.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)

print(f"Selected sphere: {info['selected_sphere_id']}")
print(f"Reward: {reward:.3f}")
```

### Vectorized Training

```python
from src.environment import VectorizedPackingEnvironment
from src.training import TrainingConfig

# Configure vectorized training
config = TrainingConfig(
    total_timesteps=5_000_000,
    n_envs=32,  # 32 parallel environments
    learning_rate=3e-4
)

# Run training
python scripts/train.py
```

### Temperature Effects Demo

```python
# Test different exploration strategies
temperatures = [2.0, 1.0, 0.5, 0.1]
for temp in temperatures:
    env.set_temperature(temp)
    # Higher temperature = more exploration
    # Lower temperature = more exploitation
```

## 📚 Usage Examples

### Training

```bash
# Quick test (small domain, few timesteps)
python scripts/train.py --mode test

# Full training with vectorization
python scripts/train.py --config configs/default.yaml

# HPC cluster training
sbatch deployment/slurm_template.sh
```

### Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py model.pt

# Comprehensive analysis with plots
python scripts/evaluate.py model.pt --report --plots --temperature --scaling --patterns
```

### Development and Testing

```bash
# Run demos
python scripts/demo.py quick          # Quick vectorized test
python scripts/demo.py sphere         # Sphere selection demo
python scripts/demo.py vectorized     # Vectorized environment demo

# Run tests
python -m pytest tests/
```


## 🧠 Key Concepts

### Sphere Selection Strategy

The agent can select **any sphere** (core or supplementary) for placement, not just core spheres. This dramatically expands the action space and enables more flexible particle growth strategies.

### Temperature-Controlled Exploration

- **High Temperature (1.5-2.0)**: More exploration, diverse sphere selection
- **Low Temperature (0.1-0.5)**: More exploitation, focused sphere selection
- **Temperature Scheduling**: Linear decay from high to low during training

### Graph Neural Network Architecture

- **Node Features**: 15 dimensions (position, radius, particle info, etc.)
- **Edge Features**: 9 dimensions (distance, overlap, connectivity, etc.)
- **Global Features**: 8 dimensions (density, contacts, progress, etc.)
- **GNN Layers**: 3 GAT layers with 4 attention heads each

### Connectivity Analysis

Hybrid approach for 100% reliable sphere connectivity:
- **Analytical Method**: Fast `distance < sum_of_radii` check
- **Lookup Table**: Voxel-level verification for edge cases
- **Supports All Radii**: [2,3,5,7,9,10-20] from connectivity analysis

## 🚀 Performance Optimization

### Vectorized Environments

- **Single Environment**: ~50-100 episodes/hour
- **32 Environments**: ~500-1000 episodes/hour (5-10x speedup)
- **Memory Usage**: ~1-2 GB per environment
- **Recommended**: 64+ CPU cores, 128+ GB RAM for large scale

### HPC Deployment

```bash
# Example SLURM submission
sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=256G deployment/slurm_template.sh
```

### Memory Management

- **Chunk-based Voxel Storage**: Efficient memory usage
- **Graph Batching**: PyTorch Geometric automatic batching
- **Shared Memory**: Zero-copy data transfer between processes

## 📈 Evaluation and Analysis

### Performance Metrics

- **Reward**: Multi-objective optimization (particle size + contacts)
- **Particle Size**: Target average particle size (35.0)
- **Contact Count**: Inter-particle contact optimization
- **Sphere Diversity**: Selection strategy analysis
- **Episode Length**: Efficiency measurement

### Analysis Tools

```python
from scripts.evaluate import ModelEvaluator

evaluator = ModelEvaluator('path/to/model.pt')

# Performance analysis
performance = evaluator.evaluate_sphere_selection_performance()

# Temperature effects
temp_effects = evaluator.evaluate_temperature_effects()

# Domain scaling
scaling = evaluator.evaluate_domain_scaling()

# Selection patterns
patterns = evaluator.analyze_sphere_selection_patterns()
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

# Format code
black src/ scripts/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


