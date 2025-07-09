# SAGIN Network Simulation

A comprehensive Space-Air-Ground Integrated Network (SAGIN) simulation framework with hierarchical reinforcement learning capabilities.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd SAGIN

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python examples/sagin_demo.py
```

## 📋 Overview

This simulation framework provides:

- **Multi-layer network architecture**: Vehicles, UAVs, and satellites
- **Configurable scenarios**: Pre-defined and custom configurations
- **Hierarchical RL integration**: Dynamic UAV allocation and task offloading
- **Comprehensive metrics**: Performance analysis and visualization
- **Grid-based topology**: Flexible network layouts

## 🏗️ Architecture

```
SAGIN Network
├── Ground Layer: Vehicles (random movement, bus routes)
├── Air Layer: UAVs (static per region, dynamic repositioning)
└── Space Layer: Satellites (constellation with orbital mechanics)
```

## 📊 Available Configurations

| Configuration | Grid Size | Vehicles | UAVs | Satellites | Use Case |
|---------------|-----------|----------|------|------------|----------|
| `small_test` | 2×3 | 13 | 2 | 4 | Quick testing |
| `medium_demo` | 4×4 | 40 | 5 | 8 | Standard demo |
| `large_simulation` | 5×10 | 100 | 15 | 12 | Performance testing |
| `highway_scenario` | 1×20 | 75 | 8 | 6 | Highway simulation |
| `city_scenario` | 8×12 | 180 | 20 | 16 | Dense urban |
| `sparse_rural` | 6×8 | 30 | 6 | 10 | Rural coverage |

## 🎯 Key Features

### Configuration System
- **No code changes needed** for different scenarios
- **Grid-based topology** with configurable dimensions
- **Vehicle distribution** (random walkers, bus routes)
- **UAV management** (static per region + dynamic reallocation)
- **Satellite constellations** with orbital mechanics
- **Task generation** with burst events

### Performance Models
- **Shannon capacity** for data rate calculations
- **Latency modeling** with propagation, transmission, queuing delays
- **Load balancing** metrics and optimization
- **Energy consumption** tracking

### Hierarchical RL Module
- **Central agent** for dynamic UAV allocation
- **Local agents** for task offloading decisions
- **MDP formulation** based on research paper
- **Reward structure** optimizing success rate and load balance

## 🔧 Configuration

Edit `config/grid_config.py` to customize:

```python
# Example custom configuration
custom_config = SAGINConfig(
    name="my_scenario",
    grid=GridConfig(
        grid_rows=3,
        grid_cols=5,
        area_bounds=(0.0, 15000.0, 0.0, 9000.0)
    ),
    vehicles=VehicleConfig(
        random_vehicles=50,
        bus_vehicles=10
    ),
    uavs=UAVConfig(
        dynamic_uavs=8
    ),
    # ... other settings
)
```

## 📈 Usage Examples

### Basic Simulation
```python
from examples.sagin_demo import SAGINDemo

demo = SAGINDemo()
network = demo.run_simulation("medium_demo")
demo.print_summary(network)
```

### Custom Configuration
```python
from config.grid_config import get_sagin_config

config = get_sagin_config("highway_scenario")
network = demo.create_network("highway_scenario")
```

### RL Training
```python
from src.rl.trainers import HierarchicalRLTrainer

trainer = HierarchicalRLTrainer(network)
trainer.train(num_episodes=1000)
```

## 📊 Metrics and Analysis

The simulation provides comprehensive metrics:

- **Task Success Rate**: Percentage of tasks completed within deadline
- **Average Latency**: End-to-end task completion time
- **Load Imbalance**: Distribution of workload across UAVs
- **Resource Utilization**: UAV and satellite usage efficiency
- **Energy Consumption**: Power usage tracking
- **Coverage**: Network coverage percentage

## 🧪 Testing

## 📁 Project Structure

```
SAGIN/
├── config/                 # Configuration files
│   └── grid_config.py     # Main configuration system
├── src/
│   ├── core/              # Core simulation components
│   │   ├── network.py     # Main network orchestrator
│   │   ├── vehicles.py    # Vehicle management
│   │   ├── uavs.py        # UAV models
│   │   ├── satellites.py  # Satellite constellation
│   │   └── tasks.py       # Task generation and management
│   ├── models/            # Mathematical models
│   │   ├── communication.py # Shannon capacity, path loss
│   │   └── latency.py     # Latency components
│   └── rl/                # Reinforcement learning
│       ├── environment.py # RL environment
│       ├── agents.py      # Central and local agents
│       └── trainers.py    # Training algorithms
├── examples/
│   └── sagin_demo.py      # Main demonstration script
├── docs/                  # Documentation
└── tests/                 # Test files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙋 Support

For questions and support:
- Check the documentation in `docs/`
- Run tests to verify installation
- Review configuration examples in `config/`
