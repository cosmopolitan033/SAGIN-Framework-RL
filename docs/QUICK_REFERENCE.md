# SAGIN Quick Reference

## 🚀 Quick Setup

```bash
pip install -r requirements.txt
python examples/test_basic.py  # Verify installation
```

## 📖 Essential Imports

```python
from src.core.network import SAGINNetwork
from src.core.types import SystemParameters, Position, TaskType
```

## 🏗️ Basic Network Creation

```python
# 1. Create parameters
params = SystemParameters(epoch_duration=1.0, total_epochs=100)

# 2. Initialize network
network = SAGINNetwork(params)

# 3. Setup topology
area_bounds = (0.0, 5000.0, 0.0, 5000.0)
network.setup_network_topology(area_bounds, num_regions=3)

# 4. Add elements
vehicles = network.add_vehicles(30, area_bounds, "random")
uavs = network.add_dynamic_uavs(5, area_bounds)
satellites = network.add_satellite_constellation(6, 2)

# 5. Initialize and run
network.initialize_simulation()
for epoch in range(100):
    results = network.step()
```

## 📊 Key Metrics

```python
metrics = network.metrics
print(f"Success Rate: {metrics.success_rate:.3f}")
print(f"Avg Latency: {metrics.average_latency:.3f}s")
print(f"UAV Utilization: {metrics.uav_utilization:.3f}")
```

## 🔧 Common Parameters

```python
SystemParameters(
    epoch_duration=1.0,          # Simulation step size
    min_rate_threshold=1.0,      # Min data rate (Mbps)
    uav_altitude=100.0,          # UAV flight altitude (m)
    uav_max_speed=20.0,          # Max UAV speed (m/s)
    uav_battery_capacity=100000.0, # UAV battery (Joules)
    satellite_altitude=600000.0,  # Satellite altitude (m)
    max_load_imbalance=0.3       # Load balance threshold
)
```

## 🎯 Task Generation

```python
# Add burst events
network.task_manager.add_burst_event(
    region_id=1,
    start_time=50.0,
    duration=30.0,
    amplitude=2.5
)

# Task types
TaskType.NORMAL               # Standard tasks
TaskType.COMPUTATION_INTENSIVE # High CPU
TaskType.DATA_INTENSIVE       # Large data
TaskType.LATENCY_SENSITIVE    # Tight deadlines
```

## 🚗 Vehicle Types

```python
# Random mobility vehicles
random_vehicles = network.add_vehicles(50, area_bounds, "random")

# Fixed route vehicles (buses)
bus_vehicles = network.add_vehicles(20, area_bounds, "bus")
```

## 🚁 UAV Management

```python
# Static UAVs (automatically created with regions)
static_uav = network.uav_manager.get_static_uav_by_region(region_id)

# Dynamic UAVs (repositionable)
dynamic_uavs = network.add_dynamic_uavs(10, area_bounds)
available_uavs = network.uav_manager.get_available_dynamic_uavs_in_region(region_id)
```

## 🛰️ Satellite Operations

```python
# Create constellation
satellites = network.add_satellite_constellation(
    num_satellites=12,
    num_planes=3
)

# Check visibility
region_center = network.regions[1].center
visible_sats = network.satellite_constellation.find_visible_satellites(region_center)
```

## 📈 Performance Monitoring

```python
# Get network state
state = network.get_network_state()

# Get performance summary
performance = network.get_performance_summary()

# Export results
network.export_results("results.json")
```

## 🔍 Debug Mode

```python
# Verbose simulation step
results = network.step(verbose=True)

# Check for errors
from src.core.types import UAVStatus
for uav in network.uav_manager.dynamic_uavs.values():
    if uav.status == UAVStatus.INACTIVE:
        print(f"UAV {uav.id} is inactive (low energy)")
```

## 🎮 Simulation Control

```python
# Step-by-step execution
network.initialize_simulation()
for epoch in range(100):
    results = network.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {len(results.get('new_tasks', []))} new tasks")

# Continuous execution
network.run_simulation(num_epochs=1000)
```

## 📋 Results Analysis

```python
# Track metrics over time
metrics_history = []
for epoch in range(100):
    results = network.step()
    metrics_history.append({
        'epoch': epoch,
        'success_rate': network.metrics.success_rate,
        'latency': network.metrics.average_latency,
        'tasks_generated': network.metrics.total_tasks_generated,
        'tasks_completed': network.metrics.total_tasks_completed
    })
```

## 🔧 Common Patterns

### Dense Urban Scenario
```python
area_bounds = (0.0, 2000.0, 0.0, 2000.0)  # 2km x 2km
network.setup_network_topology(area_bounds, num_regions=4)
network.add_vehicles(100, area_bounds, "random")
network.add_dynamic_uavs(15, area_bounds)
```

### Sparse Rural Scenario
```python
area_bounds = (0.0, 20000.0, 0.0, 20000.0)  # 20km x 20km
network.setup_network_topology(area_bounds, num_regions=3)
network.add_vehicles(20, area_bounds, "random")
network.add_dynamic_uavs(8, area_bounds)
```

### Emergency Response
```python
# Multiple burst events
for region_id in [1, 2, 3]:
    network.task_manager.add_burst_event(
        region_id=region_id,
        start_time=50.0 + region_id * 10,
        duration=30.0,
        amplitude=4.0
    )
```

## 📊 Performance Targets

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| Success Rate | >0.7 | >0.9 | Task completion rate |
| Average Latency | <5.0s | <2.0s | End-to-end delay |
| UAV Utilization | 0.6-0.8 | 0.7-0.9 | Resource efficiency |
| Load Imbalance | <0.3 | <0.1 | Distribution fairness |
| Coverage | >80% | >95% | Satellite visibility |

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| No tasks generated | Check vehicles are added and simulation initialized |
| Low success rate | Increase UAV capacity or add more dynamic UAVs |
| High latency | Reduce task complexity or processing delays |
| Energy depletion | Increase battery capacity or reduce flight activity |
| Import errors | Run from project root directory |

## 📂 File Structure

```
src/
├── core/
│   ├── network.py      # Main orchestrator
│   ├── vehicles.py     # Vehicle management
│   ├── uavs.py         # UAV management
│   ├── satellites.py   # Satellite constellation
│   ├── tasks.py        # Task generation
│   └── types.py        # Data structures
├── models/
│   ├── communication.py # Channel models
│   └── latency.py      # Delay calculations
└── utils/              # Utility functions
```

## 🔗 Quick Links

- **[Getting Started](GETTING_STARTED.md)** - Complete tutorial
- **[API Reference](API_REFERENCE.md)** - Full API documentation
- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Technical details
- **[Troubleshooting](TROUBLESHOOTING.md)** - Problem solving

---

*Keep this reference handy while working with SAGIN! 📌*
