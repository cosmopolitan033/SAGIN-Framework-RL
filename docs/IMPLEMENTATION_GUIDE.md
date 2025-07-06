# SAGIN System Implementation Guide

## Overview

This implementation provides a comprehensive Space-Air-Ground Integrated Network (SAGIN) simulation system based on the mathematical model described in the research paper. The system implements a hierarchical network architecture with three layers:

1. **Ground Layer**: Vehicles with different mobility patterns
2. **Air Layer**: Static and dynamic UAVs for coverage and processing
3. **Space Layer**: Satellite constellation for overflow processing

## Architecture

### Core Components

#### 1. Network Elements (`src/core/`)

- **`types.py`**: Fundamental data structures and enums
  - `Position`, `Velocity`: 3D spatial representations
  - `Task`: Computational task with timing and resource requirements
  - `Region`: Geographic regions with task generation parameters
  - `SystemParameters`: Global system configuration

- **`vehicles.py`**: Vehicle models and mobility patterns
  - `RandomWaypointMobility`: Random movement within bounded area
  - `FixedRouteMobility`: Predictable routes (buses, trains)
  - `Vehicle`: Base vehicle class with task generation
  - `VehicleManager`: Manages all vehicles in the system

- **`uavs.py`**: UAV models and management
  - `StaticUAV`: One per region, continuous coverage
  - `DynamicUAV`: Repositionable based on demand
  - `UAVManager`: Handles UAV allocation and operations

- **`satellites.py`**: Satellite constellation
  - `Satellite`: LEO satellite with orbital mechanics
  - `SatelliteConstellation`: Manages satellite network

- **`tasks.py`**: Task generation and management
  - `TaskGenerator`: Spatio-temporal task generation
  - `TaskQueue`: Priority-based task queuing
  - `TaskManager`: System-wide task coordination

- **`network.py`**: Main orchestrator
  - `SAGINNetwork`: Central coordinator for all components
  - Simulation execution and metrics collection

#### 2. Models (`src/models/`)

- **`communication.py`**: Communication models
  - Shannon capacity calculations
  - Path loss models for different link types
  - Link quality assessment

- **`latency.py`**: End-to-end latency calculations
  - Propagation, transmission, queuing, and processing delays
  - Multi-hop latency estimation

## Key Features Implemented

### 1. Heterogeneous Network Architecture

The system implements the three-layer SAGIN architecture:

```python
# Ground Layer
vehicles = network.add_vehicles(100, area_bounds, vehicle_type="random")
buses = network.add_vehicles(20, area_bounds, vehicle_type="bus")

# Air Layer  
static_uavs = network.setup_network_topology(area_bounds, num_regions=5)
dynamic_uavs = network.add_dynamic_uavs(10, area_bounds)

# Space Layer
satellites = network.add_satellite_constellation(num_satellites=12, num_planes=3)
```

### 2. Dynamic Task Generation

Tasks are generated based on spatio-temporal patterns:

```python
# Regional task intensity with burst events
network.task_manager.add_burst_event(
    region_id=1, start_time=100.0, duration=50.0, amplitude=3.0
)

# Different task types with varying characteristics
TaskType.COMPUTATION_INTENSIVE  # High CPU cycles
TaskType.DATA_INTENSIVE        # Large data transfers
TaskType.LATENCY_SENSITIVE     # Strict deadlines
TaskType.NORMAL               # Standard tasks
```

### 3. Vehicle Mobility Models

Two primary mobility patterns:

```python
# Random waypoint for cars, pedestrians
RandomWaypointMobility(area_bounds, min_speed=5.0, max_speed=15.0)

# Fixed routes for buses, trains
FixedRouteMobility(route_points, speed=10.0, loop=True)
```

### 4. UAV Energy Management

UAVs have comprehensive energy modeling:

```python
# Energy consumption components
E_flight = P_hover * dt + xi * distance_flown
E_comm = eta_comm * data_transmitted  
E_comp = kappa * cpu_cycles_processed

# Energy constraint
E_uav(t+1) = E_uav(t) - (E_flight + E_comm + E_comp)
```

### 5. Communication Models

Implements Shannon capacity with realistic path loss:

```python
# Vehicle-to-UAV (air-to-ground)
path_loss = air_to_ground_path_loss(distance, height, frequency)

# UAV-to-UAV (line-of-sight)
path_loss = free_space_path_loss(distance, frequency)

# UAV-to-Satellite (space communications)
path_loss = satellite_path_loss(distance, frequency, atmospheric_loss)

# Data rate calculation
data_rate = bandwidth * log2(1 + P_tx * G / N_0)
```

### 6. Latency Calculation

End-to-end latency with four components:

```python
T_total = T_prop + T_trans + T_queue + T_comp

# Propagation delay
T_prop = distance / propagation_speed

# Transmission delay  
T_trans = data_size / data_rate

# Queuing delay
T_queue = workload_ahead / cpu_capacity

# Processing delay
T_comp = cpu_cycles / cpu_capacity
```

### 7. Task Offloading Decisions

Three-level decision hierarchy:

```python
class TaskDecision(Enum):
    LOCAL = "local"      # Process at static UAV
    DYNAMIC = "dynamic"  # Forward to dynamic UAV
    SATELLITE = "satellite"  # Escalate to satellite
```

## Usage Examples

### Basic Simulation

```python
from core.network import SAGINNetwork
from core.types import SystemParameters

# Create network
params = SystemParameters(epoch_duration=1.0, total_epochs=1000)
network = SAGINNetwork(params)

# Setup topology
area_bounds = (0.0, 10000.0, 0.0, 10000.0)
network.setup_network_topology(area_bounds, num_regions=5)

# Add network elements
network.add_vehicles(100, area_bounds, vehicle_type="random")
network.add_dynamic_uavs(10, area_bounds)
network.add_satellite_constellation(num_satellites=12, num_planes=3)

# Initialize and run
network.initialize_simulation()
network.run_simulation(num_epochs=1000)
```

### Advanced Configuration

```python
# Custom system parameters
params = SystemParameters(
    epoch_duration=0.5,     # 0.5 second epochs
    min_rate_threshold=2.0,  # 2 Mbps minimum rate
    uav_max_speed=25.0,     # 25 m/s UAV speed
    max_load_imbalance=0.2,  # Stricter load balancing
    alpha1=0.15,            # Load imbalance penalty
    alpha2=0.8              # Energy violation penalty
)

# Custom vehicle mobility
from core.vehicles import RandomWaypointMobility
mobility = RandomWaypointMobility(
    area_bounds=(0, 5000, 0, 5000),
    min_speed=10.0,
    max_speed=30.0,
    pause_time=2.0
)

# Custom task characteristics
from core.tasks import TaskCharacteristics, TaskType
high_compute_tasks = TaskCharacteristics(
    cpu_cycles_mean=1e10,      # 10 billion cycles
    cpu_cycles_std=2e9,        # High variance
    data_size_in_mean=0.1,     # Small input
    data_size_out_mean=0.05,   # Small output
    deadline_mean=5.0,         # 5 second deadline
    deadline_std=1.0,          # Tight deadline
    priority=1.5               # High priority
)
```

## Performance Metrics

The system tracks comprehensive performance metrics:

```python
metrics = network.get_performance_summary()

# Task performance
success_rate = metrics['final_metrics'].success_rate
average_latency = metrics['final_metrics'].average_latency
deadline_violations = metrics['final_metrics'].total_tasks_failed

# Resource utilization
uav_utilization = metrics['final_metrics'].uav_utilization
satellite_utilization = metrics['final_metrics'].satellite_utilization
load_imbalance = metrics['final_metrics'].load_imbalance

# Network coverage
coverage_percentage = metrics['final_metrics'].coverage_percentage
```

## Simulation Control

```python
# Step-by-step simulation
network.initialize_simulation()
for epoch in range(1000):
    step_results = network.step()
    
    # Process results
    new_tasks = step_results['new_tasks']
    completed_tasks = step_results['uav_completed']
    
    # Add dynamic events
    if epoch == 500:
        network.task_manager.add_burst_event(
            region_id=1, start_time=network.current_time, 
            duration=100.0, amplitude=2.0
        )

# Export results
network.export_results("simulation_results.json")
```

## Extension Points

### 1. Custom Mobility Models

```python
from core.vehicles import MobilityModel

class CustomMobility(MobilityModel):
    def update_position(self, current_pos, current_time, dt):
        # Custom movement logic
        return new_position
    
    def get_velocity(self, current_time):
        # Custom velocity calculation
        return velocity
```

### 2. Custom Task Generation

```python
from core.tasks import TaskGenerator

class CustomTaskGenerator(TaskGenerator):
    def generate_task(self, vehicle_id, region_id, current_time, task_type):
        # Custom task generation logic
        return custom_task
```

### 3. Custom Communication Models

```python
from models.communication import CommunicationModel

class CustomCommModel(CommunicationModel):
    def calculate_data_rate(self, tx_pos, rx_pos, tx_type, rx_type):
        # Custom channel model
        return custom_data_rate
```

## Next Steps: Reinforcement Learning Integration

The system is designed to support RL-based optimization:

1. **Hierarchical RL**: Central controller for UAV allocation, local agents for task offloading
2. **State Representation**: Network state, queue lengths, energy levels
3. **Action Spaces**: UAV assignments, offloading decisions
4. **Reward Function**: Task completion rate, latency, energy efficiency

Future implementations will include:
- Deep Q-Network (DQN) for discrete action spaces
- Actor-Critic methods for continuous control
- Multi-agent RL for distributed decision making
- Transfer learning for different scenarios

## Running the Examples

```bash
# Basic functionality test
python examples/test_basic.py

# Simple demonstration
python examples/demo_simple.py

# Full simulation with visualization (requires matplotlib)
python examples/basic_simulation.py
```

The SAGIN system provides a solid foundation for research in heterogeneous network optimization, task offloading, and UAV coordination strategies.
