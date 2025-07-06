# SAGIN - Space-Air-Ground Integrated Network

A comprehensive implementation of a heterogeneous Space-Air-Ground Integrated Network (SAGIN) system with hierarchical reinforcement learning for dynamic UAV allocation and task offloading.

## Overview

SAGIN represents a three-layer heterogeneous network architecture that integrates:

- **🚗 Ground Layer**: Vehicles with communication modules generating computational tasks
- **🚁 Air Layer**: Static UAVs (regional coverage) and Dynamic UAVs (repositionable on-demand)  
- **🛰️ Space Layer**: LEO satellite constellation providing overflow processing capacity

The system implements realistic models for mobility, communication, energy consumption, and task processing, with support for hierarchical reinforcement learning optimization.

## System Overview

The SAGIN system partitions the deployment area into R non-overlapping sub-regions and operates in discrete time epochs. It consists of:

- **Vehicles**: Ground vehicles with communication modules (no computation capability)
- **Static UAVs**: One per region, providing continuous coverage and first-hop offloading
- **Dynamic UAVs**: Centrally managed, reassignable based on task demand
- **Satellites**: LEO satellites with computing capacity for overflow tasks

## Key Features

1. **Dynamic UAV Repositioning**: Real-time allocation of dynamic UAVs based on regional task demand
2. **Hierarchical Task Offloading**: Multi-level decision making (local, dynamic UAV, satellite)
3. **Energy-Aware Operations**: UAV energy consumption modeling for flight, communication, and computation
4. **Latency Optimization**: End-to-end latency minimization with deadline constraints
5. **Load Balancing**: Fair workload distribution across the network
6. **Reinforcement Learning**: Hierarchical RL for joint optimization

## Project Structure

```
SAGIN/
├── src/
│   ├── core/                    # Core system components
│   │   ├── network.py          # Network topology and elements
│   │   ├── vehicles.py         # Vehicle models and mobility
│   │   ├── uavs.py             # UAV models (static and dynamic)
│   │   ├── satellites.py       # Satellite models
│   │   └── tasks.py            # Task generation and management
│   ├── models/                  # Communication and computation models
│   │   ├── communication.py    # Channel models and data rates
│   │   ├── computation.py      # Computing and processing models
│   │   ├── energy.py           # Energy consumption models
│   │   └── latency.py          # Latency calculation models
│   ├── rl/                     # Reinforcement Learning components
│   │   ├── agents/             # RL agents
│   │   ├── environments/       # RL environments
│   │   └── policies/           # Policy implementations
│   ├── optimization/           # Optimization algorithms
│   │   ├── allocation.py       # Dynamic UAV allocation
│   │   ├── offloading.py       # Task offloading decisions
│   │   └── constraints.py      # System constraints
│   └── utils/                  # Utility functions
├── tests/                      # Unit and integration tests
├── examples/                   # Example usage and scenarios
├── docs/                       # Documentation
└── requirements.txt            # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.core.network import SAGINNetwork
from src.optimization.allocation import DynamicUAVAllocator
from src.rl.environments.sagin_env import SAGINEnvironment

# Initialize SAGIN network
network = SAGINNetwork(num_regions=5, num_vehicles=100, num_dynamic_uavs=10)

# Create RL environment
env = SAGINEnvironment(network)

# Train hierarchical RL agents
# ... (training code)
```

## Documentation

📖 **[Complete Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Detailed technical documentation with examples and API reference

Key documentation sections:
- Architecture overview and component design
- Mathematical model implementation details
- Usage examples and configuration options
- Performance metrics and monitoring
- Extension points for custom components

## Configuration

The system can be configured through JSON files or environment variables. Key parameters include:

- Network topology (regions, distances)
- Vehicle mobility patterns
- UAV specifications (energy, processing capacity)
- Task generation parameters
- Communication parameters
- RL hyperparameters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic test
python examples/test_basic.py

# Run demonstration
python examples/demo_simple.py

# Run full simulation (requires matplotlib)
python examples/basic_simulation.py
```

## Project Status

✅ **Core Implementation Complete**
- Three-layer heterogeneous network architecture
- Realistic mobility models and task generation
- Comprehensive communication and latency models
- Energy-aware UAV operations
- Performance monitoring and metrics

🔄 **Ready for RL Integration**
- Hierarchical decision making framework
- State/action space definitions
- Reward function implementation
- Multi-agent coordination support

## License

MIT License
