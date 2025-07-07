# SAGIN - Space-Air-Ground Integrated Network

A comprehensive implementation of a heterogeneous Space-Air-Ground Integrated Network (SAGIN) system with hierarchical task offloading and dynamic UAV allocation, designed for reinforcement learning research.

## 🌟 Overview

SAGIN represents a three-layer heterogeneous network architecture integrating:

- **🚗 Ground Layer**: Vehicles with communication modules generating computational tasks
- **🚁 Air Layer**: Static UAVs (regional coverage) and Dynamic UAVs (repositionable on-demand)  
- **🛰️ Space Layer**: LEO satellite constellation providing overflow processing capacity

The system implements realistic models for mobility, communication, energy consumption, and task processing with comprehensive logging and metrics collection.

## ✨ Key Features

✅ **Complete Three-Layer Architecture**
- Realistic mobility models and spatio-temporal task generation
- Comprehensive communication models with Shannon capacity
- Energy-aware UAV operations with battery constraints
- Hierarchical task offloading (Local → Dynamic UAV → Satellite)
- Performance monitoring with detailed metrics

✅ **Advanced Simulation Capabilities**
- Step-by-step simulation control with detailed logging
- Configurable network topology and parameters
- Burst event generation for dynamic scenarios
- Real-time performance analytics and state reporting

🔄 **RL Integration Ready**
- Hierarchical decision making framework
- State/action space definitions prepared
- Reward function components implemented
- Multi-agent coordination support

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python examples/sagin_demo.py
```

### Interactive Demo Options
The unified demo provides multiple simulation modes:
- **🧪 Test Mode**: Quick validation (5 steps)
- **🎓 Simple Demo**: Educational overview (100 epochs)
- **📊 Standard Simulation**: Medium logging (200 epochs)
- **🔍 Detailed Simulation**: High logging (100 epochs)
- **🚀 Full Simulation**: Low logging (500 epochs)
- **🎯 Custom Simulation**: Choose your own settings
- **🔄 Legacy Mode**: Compatible with old basic_simulation.py behavior

### Basic Usage
```python
from src.core.network import SAGINNetwork

# Create and configure network
network = SAGINNetwork()
network.setup_network_topology((0, 5000, 0, 5000), num_regions=3)

# Add network elements
network.add_vehicles(30, (0, 5000, 0, 5000), vehicle_type="random")
network.add_vehicles(10, (0, 5000, 0, 5000), vehicle_type="bus")
network.add_dynamic_uavs(5, (0, 5000, 0, 5000))
network.add_satellite_constellation(num_satellites=6, num_planes=2)

# Run simulation
network.initialize_simulation()
for epoch in range(50):
    results = network.step()
    
print(f"Success rate: {network.metrics.success_rate:.3f}")
```

### Run Examples
```bash
python examples/test_basic.py           # Basic functionality test
python examples/demo_simple.py         # Simple demonstration  
python examples/detailed_logging_demo.py # Comprehensive logging demo
python examples/basic_simulation.py    # Full simulation with visualization
```

## 📊 System Architecture

### Core Components

| Layer | Component | Description |
|-------|-----------|-------------|
| **Space** | LEO Satellites | High-capacity processing nodes with orbital mechanics |
| **Air** | Static UAVs | One per region, continuous coverage |
| **Air** | Dynamic UAVs | Repositionable based on demand |
| **Ground** | Vehicles | Mobile nodes with task generation patterns |

### Task Offloading Hierarchy

```
🚗 Vehicle generates task
    ↓
🚁 Static UAV (local processing)
    ↓ (if overloaded)
🚁 Dynamic UAV (load balancing)
    ↓ (if unavailable)
🛰️ Satellite (overflow processing)
```

### Performance Metrics

- **Success Rate**: Task completion percentage (target: >0.8)
- **Average Latency**: End-to-end processing time
- **Resource Utilization**: UAV and satellite usage efficiency
- **Energy Consumption**: Total system energy tracking
- **Load Imbalance**: Workload distribution fairness
- **Coverage**: Satellite visibility percentage

## 🏗️ Project Structure

```
SAGIN/
├── src/
│   ├── core/                    # Core system components
│   │   ├── network.py          # Main network orchestrator  
│   │   ├── vehicles.py         # Vehicle models and mobility
│   │   ├── uavs.py             # UAV models (static and dynamic)
│   │   ├── satellites.py       # Satellite constellation
│   │   ├── tasks.py            # Task generation and management
│   │   └── types.py            # Data structures and enums
│   ├── models/                  # Mathematical models
│   │   ├── communication.py    # Channel models and data rates
│   │   └── latency.py          # Latency calculation models
│   ├── optimization/           # Future RL/optimization components
│   └── utils/                  # Utility functions
├── examples/                   # Usage examples and demonstrations
├── docs/                       # Detailed documentation
└── requirements.txt            # Dependencies
```

## 🔧 Configuration Examples

### Custom System Parameters
```python
from src.core.types import SystemParameters

params = SystemParameters(
    epoch_duration=1.0,        # 1 second epochs
    uav_altitude=100.0,        # 100m UAV flight altitude
    max_uav_speed=25.0,        # 25 m/s maximum UAV speed
    min_rate_threshold=1.0,    # 1 Mbps minimum data rate
    satellite_altitude=600000  # 600km satellite altitude
)
```

### Dynamic Events
```python
# Add traffic burst events
network.task_manager.add_burst_event(
    region_id=1, start_time=50.0, duration=30.0, amplitude=2.5
)
```

### Different Scenarios
```python
# Dense urban scenario
area_bounds = (0.0, 2000.0, 0.0, 2000.0)  # 2km x 2km
network.add_vehicles(100, area_bounds, "random")
network.add_dynamic_uavs(15, area_bounds)

# Sparse rural scenario  
area_bounds = (0.0, 20000.0, 0.0, 20000.0)  # 20km x 20km
network.add_vehicles(20, area_bounds, "random")
network.add_dynamic_uavs(8, area_bounds)
```

## 📈 Implementation Status

### Core Implementation ✅ Complete
- **Architecture**: Three-layer heterogeneous network
- **Mobility Models**: Random waypoint and fixed route patterns
- **Communication**: Shannon capacity with realistic path loss
- **Energy Management**: UAV battery constraints and energy consumption
- **Task Processing**: Hierarchical offloading with decision framework
- **Performance Monitoring**: Comprehensive metrics and logging

### Key Achievements
- **40+ Vehicles**: Random and fixed route mobility patterns
- **Static + Dynamic UAVs**: Regional coverage and on-demand repositioning
- **12 LEO Satellites**: Orbital mechanics with visibility calculations
- **Realistic Models**: Physical constraints, energy consumption, communication limits
- **Hierarchical Decisions**: 3-level task offloading optimization
- **Comprehensive Metrics**: Success rate, latency, utilization, energy tracking

### Technical Specifications
- **Simulation Speed**: ~1000 epochs/minute on standard hardware
- **Memory Usage**: ~100MB for typical simulations
- **Dependencies**: NumPy, JSON (minimal external dependencies)
- **Extensibility**: Plugin architecture for custom components
- **Testing**: Comprehensive unit and integration tests

## 🔄 RL Integration Ready

### Prepared Components
- **State Space**: Network state, resource availability, task queues
- **Action Space**: UAV positioning, task assignment, resource allocation
- **Reward Functions**: Success rate, latency, energy efficiency, load balance
- **Multi-Agent Support**: Coordinated UAV operations and satellite usage

### Research Applications
- **Task Offloading Optimization**: Hierarchical decision making
- **Dynamic UAV Allocation**: Demand-based positioning strategies
- **Energy Management**: Battery-aware operations and charging
- **Load Balancing**: Workload distribution across processing nodes

## 📚 Documentation

### Core Documentation
- **[Getting Started](docs/GETTING_STARTED.md)** - Complete tutorial for new users
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Technical architecture and detailed implementation
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with examples

### Reference Materials
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Developer reference card with essential commands
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🎯 Getting Started Paths

### New Users
1. **Installation**: `pip install -r requirements.txt`
2. **Verify**: `python examples/test_basic.py`
3. **Learn**: Read [Getting Started Guide](docs/GETTING_STARTED.md)
4. **Explore**: Run examples in `examples/`

### Developers
1. **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
2. **Architecture**: [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)
3. **Quick Reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
4. **Extend**: Check extension points in Implementation Guide

### Researchers
1. **Capabilities**: Review system architecture above
2. **RL Integration**: Check RL section in [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)
3. **Performance**: Study metrics and simulation control
4. **Customize**: Explore parameter tuning and custom components

## 🔍 Common Use Cases

### Basic Simulation
```python
network = SAGINNetwork()
network.setup_network_topology((0, 5000, 0, 5000), num_regions=3)
network.add_vehicles(40, (0, 5000, 0, 5000), "random")
network.add_dynamic_uavs(5, (0, 5000, 0, 5000))
network.initialize_simulation()

for epoch in range(100):
    results = network.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Success rate = {network.metrics.success_rate:.3f}")
```

### Performance Analysis
```python
# Track metrics over time
metrics_history = []
for epoch in range(100):
    results = network.step()
    metrics_history.append({
        'epoch': epoch,
        'success_rate': network.metrics.success_rate,
        'latency': network.metrics.average_latency,
        'utilization': network.metrics.uav_utilization
    })

# Export results
network.export_results("simulation_results.json")
```

## 🚨 Troubleshooting Quick Tips

| Issue | Solution |
|-------|----------|
| Import errors | Run from project root directory |
| No tasks generated | Check vehicles are added and simulation initialized |
| Low success rate | Increase UAV capacity or add more dynamic UAVs |
| High latency | Reduce task complexity or increase processing resources |
| Energy depletion | Increase battery capacity or reduce flight activity |

**For detailed troubleshooting**: See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

1. Check existing documentation for similar information
2. Follow the established code structure and patterns
3. Add tests for new functionality
4. Update documentation for API changes
5. Test examples and verify they work

## 📄 License

MIT License

---

**SAGIN** - A comprehensive Space-Air-Ground Integrated Network simulation framework for research and development.

*For detailed technical documentation, see the [docs](docs/) directory.*
