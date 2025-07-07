# Getting Started with SAGIN

This tutorial will guide you through setting up and running your first SAGIN simulation.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- Basic understanding of networks and distributed systems

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SAGIN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python examples/test_basic.py
```

If you see "🎉 All tests passed successfully!", your installation is working correctly.

## Your First Simulation

### Step 1: Understanding the Architecture

SAGIN consists of three layers:

```
🛰️  SPACE LAYER    │ LEO Satellites (overflow processing)
🚁  AIR LAYER      │ UAVs (Static per region + Dynamic repositionable)
🚗  GROUND LAYER   │ Vehicles (task generation and mobility)
```

### Step 2: Simple Example

Create a file called `my_first_simulation.py`:

```python
from src.core.network import SAGINNetwork
from src.core.types import SystemParameters

# Create system parameters
params = SystemParameters(
    epoch_duration=1.0,      # 1 second per simulation step
    total_epochs=50,         # Run for 50 steps
    min_rate_threshold=1.0,  # Minimum 1 Mbps data rate
    uav_altitude=100.0,      # UAVs fly at 100m altitude
    uav_max_speed=20.0       # Maximum UAV speed 20 m/s
)

# Initialize the network
network = SAGINNetwork(params)

# Define a 5km x 5km simulation area
area_bounds = (0.0, 5000.0, 0.0, 5000.0)

# Setup network topology with 3 regions
network.setup_network_topology(area_bounds, num_regions=3)

# Add network elements
vehicles = network.add_vehicles(30, area_bounds, vehicle_type="random")
buses = network.add_vehicles(10, area_bounds, vehicle_type="bus")
dynamic_uavs = network.add_dynamic_uavs(5, area_bounds)
satellites = network.add_satellite_constellation(num_satellites=6, num_planes=2)

print(f"Created network with:")
print(f"  - 3 regions with static UAVs")
print(f"  - {len(vehicles)} random vehicles")
print(f"  - {len(buses)} bus vehicles")
print(f"  - {len(dynamic_uavs)} dynamic UAVs")
print(f"  - {len(satellites)} satellites")

# Initialize and run simulation
network.initialize_simulation()

print("\nRunning simulation...")
for epoch in range(50):
    results = network.step()
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        metrics = network.metrics
        print(f"Epoch {epoch}: "
              f"Generated: {metrics.total_tasks_generated}, "
              f"Completed: {metrics.total_tasks_completed}, "
              f"Success Rate: {metrics.success_rate:.3f}")

# Final results
print(f"\nFinal Results:")
print(f"Success Rate: {network.metrics.success_rate:.3f}")
print(f"Average Latency: {network.metrics.average_latency:.3f}s")
print(f"UAV Utilization: {network.metrics.uav_utilization:.3f}")
```

### Step 3: Run Your Simulation

```bash
python my_first_simulation.py
```

You should see output showing the network creation, simulation progress, and final results.

## Understanding the Output

### Network Creation
- **Regions**: Geographic areas that group vehicles and have dedicated static UAVs
- **Vehicles**: Mobile nodes that generate computational tasks
- **UAVs**: Processing nodes that handle tasks locally or forward to satellites
- **Satellites**: High-capacity processing nodes for overflow tasks

### Simulation Progress
- **Generated**: Total tasks created by vehicles
- **Completed**: Tasks successfully processed
- **Success Rate**: Percentage of tasks completed before deadline
- **Latency**: Average time from task generation to completion

### Key Metrics
- **Success Rate**: Higher is better (target: >0.8)
- **Average Latency**: Lower is better (depends on task deadlines)
- **UAV Utilization**: Efficiency of UAV usage (target: 0.6-0.8)
- **Load Imbalance**: Distribution of work (lower is better)

## Next Steps

### 1. Explore Examples

```bash
# Run comprehensive examples
python examples/demo_simple.py           # Simple demonstration
python examples/detailed_logging_demo.py # With detailed logging
python examples/basic_simulation.py      # Full simulation with plots
```

### 2. Customize Your Simulation

#### Add Burst Events
```python
# Add a traffic burst in region 1
network.task_manager.add_burst_event(
    region_id=1,
    start_time=20.0,    # Start at 20 seconds
    duration=15.0,      # Last for 15 seconds
    amplitude=3.0       # 3x normal task generation
)
```

#### Custom System Parameters
```python
params = SystemParameters(
    epoch_duration=0.5,        # Faster simulation
    min_rate_threshold=2.0,    # Higher data rate requirement
    uav_battery_capacity=150000.0,  # Longer UAV operation
    max_load_imbalance=0.2,    # Stricter load balancing
    energy_threshold=0.1       # Earlier energy warnings
)
```

#### Different Vehicle Types
```python
# Create vehicles with different mobility patterns
cars = network.add_vehicles(50, area_bounds, vehicle_type="random")
buses = network.add_vehicles(20, area_bounds, vehicle_type="bus")
```

### 3. Monitor Performance

```python
# Get detailed network state
network_state = network.get_network_state()
print(f"Current time: {network_state['current_time']}")
print(f"Active regions: {len(network_state['regions'])}")

# Get performance summary
performance = network.get_performance_summary()
print(f"Total epochs: {performance['total_epochs']}")
print(f"Network elements: {performance['network_elements']}")
```

### 4. Export Results

```python
# Export simulation results to JSON
network.export_results("my_simulation_results.json")
```

## Common Patterns

### 1. Dense Urban Scenario
```python
# High density, many vehicles, frequent tasks
area_bounds = (0.0, 2000.0, 0.0, 2000.0)  # Smaller area
network.setup_network_topology(area_bounds, num_regions=4)
network.add_vehicles(100, area_bounds, vehicle_type="random")
network.add_vehicles(30, area_bounds, vehicle_type="bus")
network.add_dynamic_uavs(15, area_bounds)
```

### 2. Sparse Rural Scenario
```python
# Low density, fewer vehicles, less frequent tasks
area_bounds = (0.0, 20000.0, 0.0, 20000.0)  # Larger area
network.setup_network_topology(area_bounds, num_regions=3)
network.add_vehicles(20, area_bounds, vehicle_type="random")
network.add_dynamic_uavs(8, area_bounds)
```

### 3. Emergency Response Scenario
```python
# Add multiple burst events to simulate emergency
for region_id in [1, 2, 3]:
    network.task_manager.add_burst_event(
        region_id=region_id,
        start_time=50.0 + region_id * 10,
        duration=30.0,
        amplitude=4.0
    )
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the project root directory
2. **No Tasks Generated**: Check if vehicles are added and simulation is initialized
3. **Low Success Rate**: Increase UAV capacity or add more dynamic UAVs
4. **High Latency**: Reduce task complexity or increase processing resources

### Debug Mode

```python
# Enable verbose logging
for epoch in range(10):
    results = network.step(verbose=True)  # Detailed output
```

### Performance Monitoring

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
```

## Advanced Topics

After mastering the basics, explore:

1. **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Deep technical details
2. **[API Reference](API_REFERENCE.md)** - Complete API documentation
3. **Custom Extensions** - Add your own mobility models, task types, or communication models
4. **Reinforcement Learning** - Use the system for RL research

## Getting Help

- Check the **[Troubleshooting Guide](TROUBLESHOOTING.md)** for common issues
- Look at the `examples/` directory for more complex scenarios
- Review the **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** for technical details

## What's Next?

Now that you have a basic understanding of SAGIN, you can:

1. **Experiment** with different network configurations
2. **Analyze** performance under various conditions
3. **Extend** the system with custom components
4. **Research** task offloading strategies and UAV coordination
5. **Develop** reinforcement learning algorithms for dynamic optimization

Welcome to the SAGIN ecosystem! 🚀
