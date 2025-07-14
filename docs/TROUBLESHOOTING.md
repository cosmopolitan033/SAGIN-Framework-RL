# SAGIN Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from the project root directory:
```bash
cd /path/to/SAGIN
python examples/test_basic.py  # Not python src/examples/test_basic.py
```

#### 2. Missing Dependencies
```bash
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

### Simulation Runtime Issues

#### 1. AttributeError: Position/Velocity
```bash
AttributeError: 'Velocity' object has no attribute 'x'
```
**Solution**: Use correct attributes:
- Position: `.x`, `.y`, `.z`
- Velocity: `.vx`, `.vy`, `.vz`

#### 2. UAV Energy Depletion
```bash
Warning: UAV X inactive due to low energy
```
**Causes & Solutions**:
- **High flight activity**: Reduce dynamic UAV movement frequency
- **Heavy processing load**: Balance tasks across UAVs
- **Long simulation**: Increase battery capacity or add recharging
```python
params = SystemParameters(
    uav_battery_capacity=200000.0,  # Double battery capacity
    energy_threshold=0.05           # Lower energy threshold
)
```

#### 3. Task Allocation Failures
```bash
Warning: Failed to allocate task X
```
**Common causes**:
- All UAVs at energy threshold
- Communication link failures
- Resource overload

**Solutions**:
```python
# Add more dynamic UAVs
network.add_dynamic_uavs(5, area_bounds)

# Increase UAV processing capacity
params.uav_cpu_capacity = 2e9

# Adjust task generation rate
params.task_arrival_rate = 0.3  # Reduce from default 0.5
```

#### 4. Poor Success Rate
```bash
Final success rate: 0.234
```
**Optimization strategies**:
```python
# Increase processing capacity
params.uav_cpu_capacity = 2e9
params.satellite_cpu_capacity = 5e9

# Add more resources
network.add_dynamic_uavs(10, area_bounds)
network.add_satellite_constellation(12, 3)

# Reduce task complexity
params.cpu_cycles_mean = 1e8  # Reduce from 5e8
params.deadline_mean = 15.0   # Increase from 10.0
```

### Performance Issues

#### 1. Slow Simulation
**Symptoms**: Taking >10 seconds per epoch
**Solutions**:
```python
# Reduce logging verbosity
results = network.step(verbose=False)

# Reduce simulation complexity
params.epoch_duration = 2.0    # Increase epoch duration
network.setup_network_topology(area_bounds, num_regions=3)  # Fewer regions

# Optimize task generation
params.task_arrival_rate = 0.2  # Reduce task generation
```

#### 2. High Memory Usage
**Symptoms**: Memory usage >1GB
**Solutions**:
```python
# Clear metrics history periodically
if epoch % 100 == 0:
    network.metrics_history = network.metrics_history[-50:]  # Keep last 50
    
# Reduce task tracking
network.task_manager.cleanup_expired_tasks(network.current_time)
```

### Configuration Issues

#### 1. Invalid Area Bounds
```bash
ValueError: Invalid area bounds
```
**Solution**: Ensure proper format:
```python
area_bounds = (min_x, max_x, min_y, max_y)  # e.g., (0, 5000, 0, 5000)
```

#### 2. Network Topology Errors
```bash
NetworkTopologyError: No regions defined
```
**Solution**: Setup topology before adding vehicles:
```python
network.setup_network_topology(area_bounds, num_regions=3)
network.add_vehicles(30, area_bounds)  # After topology setup
```

#### 3. Unrealistic Parameter Values
**Common mistakes**:
```python
# DON'T DO THIS
params.uav_altitude = 10000.0      # Too high for air layer
params.max_uav_speed = 200.0       # Unrealistic speed
params.satellite_altitude = 100.0   # Satellites should be in space

# CORRECT VALUES
params.uav_altitude = 100.0        # 100m altitude
params.max_uav_speed = 25.0        # 25 m/s (90 km/h)
params.satellite_altitude = 600000.0 # 600km altitude
```

### Communication Issues

#### 1. No Satellite Visibility
```bash
Warning: No satellites visible from region X
```
**Solutions**:
```python
# Increase constellation size
network.add_satellite_constellation(num_satellites=18)

# Lower satellite altitude (but keep realistic)
params.satellite_altitude = 500000.0  # 500km instead of 600km

# Check orbital period matches simulation time
# Satellites should complete orbits during simulation
```

#### 2. Poor Data Rates
```bash
Average data rate: 0.1 Mbps
```
**Solutions**:
```python
# Increase transmission power
params.tx_power_dbm = 30.0  # Increase from 23 dBm

# Use higher frequency with caution
params.frequency_ghz = 5.0  # But consider path loss increase

# Increase bandwidth
params.bandwidth_mhz = 40.0  # Increase from 20 MHz

# Reduce distances (smaller deployment area)
area_bounds = (0, 3000, 0, 3000)  # 3km x 3km instead of 10km x 10km
```

### Debugging Techniques

#### 1. Enable Detailed Logging
```python
# Use detailed logging demo for troubleshooting
python examples/detailed_logging_demo.py

# Or enable verbose mode
for epoch in range(100):
    results = network.step(verbose=True)  # Shows detailed information
```

#### 2. Monitor Key Metrics
```python
def monitor_health(network):
    metrics = network.metrics
    print(f"Success rate: {metrics.success_rate:.3f}")
    print(f"UAV utilization: {metrics.uav_utilization:.3f}")
    print(f"Energy consumption: {metrics.energy_consumption:.1f}J")
    print(f"Load imbalance: {metrics.load_imbalance:.3f}")
    
    # Check UAV energy levels
    for uav_id, uav in network.uav_manager.static_uavs.items():
        energy_pct = (uav.current_energy / uav.battery_capacity) * 100
        if energy_pct < 20:
            print(f"WARNING: UAV {uav_id} low energy: {energy_pct:.1f}%")

# Use during simulation
for epoch in range(200):
    network.step(verbose=False)
    if epoch % 50 == 0:
        monitor_health(network)
```

#### 3. Export and Analyze Data
```python
# Export detailed results
network.export_results("debug_results.json")

# Analyze task completion patterns
import json
with open("debug_results.json") as f:
    data = json.load(f)
    
# Check decision distribution
decisions = data.get('decision_history', [])
local_count = sum(1 for d in decisions if d['decision'] == 'local')
failed_count = sum(1 for d in decisions if not d['success'])
print(f"Local decisions: {local_count}, Failed: {failed_count}")
```

### Common Warning Messages

#### 1. Energy Warnings
```bash
WARNING: UAV X energy below threshold
```
**Normal**: UAVs naturally consume energy
**Action needed**: If many UAVs become inactive

#### 2. Task Deadline Warnings  
```bash
WARNING: Task X approaching deadline
```
**Normal**: Some tasks have tight deadlines
**Action needed**: If >50% of tasks are failing deadlines

#### 3. Load Imbalance Warnings
```bash
WARNING: High load imbalance: 2.5
```
**Normal**: Some imbalance expected
**Action needed**: If consistently >3.0

### Performance Optimization Tips

#### 1. Simulation Speed
- Use `verbose=False` for production runs
- Reduce logging frequency
- Optimize epoch duration vs. accuracy trade-off
- Use fewer regions for initial testing

#### 2. Success Rate Optimization
- Balance processing capacity vs. task generation
- Ensure adequate dynamic UAV coverage
- Monitor energy levels and add charging stations
- Adjust task deadlines for realistic scenarios

#### 3. Resource Utilization
- Monitor load imbalance metrics
- Adjust dynamic UAV allocation strategies
- Balance satellite vs. UAV processing
- Consider communication link quality

### Best Practices

#### 1. Configuration
- Start with default parameters and adjust incrementally
- Test with small scenarios before scaling up
- Validate parameter ranges for realism
- Document configuration changes

#### 2. Development
- Use the test examples to verify functionality
- Enable detailed logging during development
- Export results for analysis
- Monitor performance metrics continuously

#### 3. Research Applications
- Establish baseline performance before optimization
- Compare different configurations systematically
- Use consistent random seeds for reproducibility
- Document experimental setups thoroughly

## Getting Help

If you encounter issues not covered here:

1. **Check the examples**: `examples/` directory has working configurations
2. **Review the API reference**: `docs/API_REFERENCE.md` for detailed usage
3. **Enable detailed logging**: Use `detailed_logging_demo.py` for insights
4. **Validate configuration**: Ensure parameters are within realistic ranges
5. **Monitor metrics**: Track success rate, energy, and utilization trends

The system is designed to be robust, but complex simulations may require parameter tuning for optimal performance.
