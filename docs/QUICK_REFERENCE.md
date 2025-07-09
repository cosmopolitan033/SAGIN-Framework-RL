# SAGIN Quick Reference

## 🚀 Quick Commands

### Run Simulation
```bash
# Interactive menu
python examples/sagin_demo.py

# Quick test
python examples/sagin_demo.py
# Then select option 4
```

#### 🎯 Quick Start Workflow

1. **Install**: `pip install -r requirements.txt`
2. **Run**: `python examples/sagin_demo.py`
3. **Select**: Option 4 (Quick test)
4. **Analyze**: Review console output
5. **Experiment**: Try different configurations
6. **Customize**: Edit `config/grid_config.py`ion Reference

### Available Configurations
| Name | Grid | Vehicles | UAVs | Description |
|------|------|----------|------|-------------|
| `small_test` | 2×3 | 13 | 2 | Quick testing |
| `medium_demo` | 4×4 | 40 | 5 | Standard demo |
| `large_simulation` | 5×10 | 100 | 15 | Performance test |
| `highway_scenario` | 1×20 | 75 | 8 | Highway/linear |
| `city_scenario` | 8×12 | 180 | 20 | Dense urban |
| `sparse_rural` | 6×8 | 30 | 6 | Rural/sparse |

### Configuration Parameters

#### Grid Configuration
```python
GridConfig(
    grid_rows=5,                    # Number of rows
    grid_cols=10,                   # Number of columns  
    area_bounds=(0, 10000, 0, 5000) # (min_x, max_x, min_y, max_y)
)
```

#### Vehicle Configuration
```python
VehicleConfig(
    random_vehicles=50,             # Random movement vehicles
    bus_vehicles=15,                # Bus route vehicles
    vehicle_speed_range=(5.0, 20.0) # Min/max speed (m/s)
)
```

#### UAV Configuration
```python
UAVConfig(
    dynamic_uavs=10,                # Number of dynamic UAVs
    uav_max_speed=20.0,             # Max UAV speed (m/s)
    uav_altitude=100.0,             # Flight altitude (m)
    uav_cpu_capacity=1e9            # Processing capacity
)
```

#### Simulation Configuration
```python
SimulationConfig(
    total_epochs=100,               # Simulation duration
    logging_level="medium",         # low/medium/high
    progress_interval=25            # Progress reporting interval
)
```

## 🎮 Interactive Menu Options

```
1. 📋 List available configurations    # Show all configs
2. ℹ️  Show configuration details     # Config info
3. 🚀 Run simulation with configuration # Main simulation
4. 🧪 Quick test (small_test config)   # Fast test
0. ❌ Exit                            # Quit
```

### Post-Simulation Options
```
p. 📈 Plot results      # Show performance graphs
e. 💾 Export results    # Save to JSON file
c. 🔄 Continue         # Return to main menu
```

## 💻 Python API Reference

### Basic Usage
```python
from examples.sagin_demo import SAGINDemo

demo = SAGINDemo()
network = demo.run_simulation("medium_demo")
demo.print_summary(network)
demo.plot_results(network)
```

### Configuration Management
```python
from config.grid_config import (
    get_sagin_config,
    list_available_configs,
    print_config_summary
)

# Get configuration
config = get_sagin_config("medium_demo")

# List all configurations
configs = list_available_configs()

# Show configuration details
print_config_summary("highway_scenario")
```

### Network Creation
```python
from src.core.network import SAGINNetwork

# Create network with custom parameters
system_params = config.get_system_parameters()
network = SAGINNetwork(system_params)
network.setup_network_topology_with_grid(config.grid)
```

## 📊 Key Metrics

### Performance Indicators
- **Success Rate**: `metrics.success_rate` (0.0 - 1.0)
- **Average Latency**: `metrics.average_latency` (seconds)
- **Load Imbalance**: `metrics.load_imbalance` (lower is better)
- **UAV Utilization**: `metrics.uav_utilization` (0.0 - 1.0)
- **Satellite Utilization**: `metrics.satellite_utilization` (0.0 - 1.0)

### Task Statistics
- **Total Generated**: `metrics.total_tasks_generated`
- **Total Completed**: `metrics.total_tasks_completed`
- **Total Failed**: `metrics.total_tasks_failed`

## 🔧 Common Customizations

### Create Small Test Environment
```python
test_config = SAGINConfig(
    grid=GridConfig(grid_rows=2, grid_cols=2),
    vehicles=VehicleConfig(random_vehicles=5, bus_vehicles=2),
    simulation=SimulationConfig(total_epochs=20)
)
```

### Highway Scenario
```python
highway_config = SAGINConfig(
    grid=GridConfig(grid_rows=1, grid_cols=10),
    vehicles=VehicleConfig(random_vehicles=30, vehicle_speed_range=(15, 30))
)
```

### Dense Urban Environment
```python
city_config = SAGINConfig(
    grid=GridConfig(grid_rows=8, grid_cols=8),
    vehicles=VehicleConfig(random_vehicles=100, bus_vehicles=20),
    uavs=UAVConfig(dynamic_uavs=15)
)
```

## 🐛 Troubleshooting

### Common Issues
| Problem | Solution |
|---------|----------|
| Import errors | Run from SAGIN directory |
| No tasks generated | Check vehicle distribution |
| Low success rates | Increase UAV capacity |
| Memory issues | Use smaller configurations |

### Debug Commands
```bash
# Verbose logging
# Edit config: logging_level="high"

```

## 📁 File Locations

### Configuration
- **Main config**: `config/grid_config.py`
- **Predefined scenarios**: `SAGIN_CONFIGS` dictionary

### Core Code
- **Network**: `src/core/network.py`
- **Vehicles**: `src/core/vehicles.py`
- **UAVs**: `src/core/uavs.py`
- **Tasks**: `src/core/tasks.py`

### Examples
- **Main demo**: `examples/sagin_demo.py`
- **RL example**: `examples/rl_sagin_optimization.py`

### Tests
- **Core tests**: `test_implementation.py`
- **Communication**: `test_integrated_communication.py`
- **RL tests**: `test_rl.py`

## 🎯 Quick Start Workflow

1. **Install**: `pip install -r requirements.txt`
2. **Test**: `python test_implementation.py`
3. **Run**: `python examples/sagin_demo.py`
4. **Select**: Option 4 (Quick test)
5. **Analyze**: Review console output
6. **Experiment**: Try different configurations
7. **Customize**: Edit `config/grid_config.py`
