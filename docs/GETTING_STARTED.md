# Getting Started with SAGIN Simulation

This guide will help you quickly get up and running with the SAGIN network simulation.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd SAGIN

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Running Your First Simulation

### Option 1: Interactive Menu
```bash
python examples/sagin_demo.py
```

This will present you with an interactive menu:
```
Select simulation mode:
1. 📋 List available configurations
2. ℹ️  Show configuration details  
3. 🚀 Run simulation with configuration
4. 🧪 Quick test (small_test config)
0. ❌ Exit
```

### Option 2: Direct Configuration
```python
from examples.sagin_demo import SAGINDemo

demo = SAGINDemo()
network = demo.run_simulation("medium_demo")
demo.print_summary(network)
```

## 📊 Understanding Configurations

Each configuration defines:
- **Grid topology**: Number of regions and area coverage
- **Network elements**: Vehicles, UAVs, satellites
- **Task parameters**: Generation rates, burst events
- **Simulation settings**: Duration, logging level

### Available Configurations

| Name | Description | Best For |
|------|-------------|----------|
| `small_test` | Quick validation (2×3 grid) | Testing and debugging |
| `medium_demo` | Standard demonstration (4×4 grid) | Learning the system |
| `large_simulation` | Comprehensive analysis (5×10 grid) | Performance evaluation |
| `highway_scenario` | Linear topology (1×20 grid) | Highway/corridor scenarios |
| `city_scenario` | Dense urban (8×12 grid) | High-density environments |
| `sparse_rural` | Rural coverage (6×8 grid) | Low-density scenarios |

## 🔧 Customizing Configurations

### Method 1: Modify Existing Configuration
Edit `config/grid_config.py` and modify the `SAGIN_CONFIGS` dictionary:

```python
"my_scenario": SAGINConfig(
    name="my_scenario",
    description="My custom scenario",
    grid=GridConfig(
        grid_rows=3,
        grid_cols=4,
        area_bounds=(0.0, 12000.0, 0.0, 9000.0)
    ),
    vehicles=VehicleConfig(
        random_vehicles=30,
        bus_vehicles=8
    ),
    # ... other parameters
)
```

### Method 2: Create Custom Configuration Programmatically
```python
from config.grid_config import create_custom_sagin_config

config = create_custom_sagin_config(
    name="test_config",
    grid_rows=2,
    grid_cols=3,
    random_vehicles=20,
    dynamic_uavs=3
)
```

## 📈 Understanding Output

### Console Output
During simulation, you'll see:
```
🚀 SAGIN SIMULATION: MEDIUM_DEMO
============================================================
Created medium_demo SAGIN network:
  - 4x4 grid (16 regions)
  - Area: 8.0km x 8.0km
  - 30 random vehicles
  - 10 bus vehicles
  - 16 static UAVs (1 per region)
  - 5 dynamic UAVs
  - 8 satellites

Running 100-epoch simulation with medium logging...
Epoch   0: Success: 0.000, Latency: 0.000s, Load: 0.000
Epoch  25: Success: 0.850, Latency: 2.340s, Load: 0.234
...
```

### Key Metrics
- **Success Rate**: Percentage of tasks completed within deadline
- **Latency**: Average time from task creation to completion
- **Load**: Load imbalance across UAVs (lower is better)

## 🧪 Testing Different Scenarios

### 1. Highway Scenario
```python
# Good for testing linear topology
demo.run_simulation("highway_scenario")
```

### 2. Dense City
```python
# Test high-load scenarios
demo.run_simulation("city_scenario") 
```

### 3. Rural Coverage
```python
# Test sparse networks
demo.run_simulation("sparse_rural")
```

## 📊 Visualization and Analysis

After running a simulation:

### View Results
```python
demo.print_summary(network)
```

### Plot Performance
```python
demo.plot_results(network)
```

### Export Data
```python
demo.export_results(network)
```

## 🔍 Debugging and Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the SAGIN directory
cd SAGIN
python examples/sagin_demo.py
```

**2. No Tasks Generated**
- Check vehicle distribution
- Verify task generation parameters
- Increase simulation duration

**3. Low Success Rates**
- Increase UAV capacity
- Add more dynamic UAVs  
- Reduce task generation rate

### Verbose Logging
For detailed debugging, use high logging level:
```python
# Edit config in grid_config.py
simulation=SimulationConfig(
    logging_level="high",
    log_decisions=True,
    log_resource_usage=True
)
```

## 🎯 Next Steps

1. **Explore Configurations**: Try different scenarios
2. **Customize Parameters**: Modify configurations for your needs
3. **Add RL Training**: Use the hierarchical RL module
4. **Analyze Performance**: Use visualization tools
5. **Create New Scenarios**: Design custom configurations

## 📚 Further Reading

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) - Technical details
- [Quick Reference](QUICK_REFERENCE.md) - Command reference

## 🆘 Getting Help

If you encounter issues:
1. Check this guide for common solutions
2. Review configuration examples in `config/grid_config.py`
3. Check the console output for error messages
