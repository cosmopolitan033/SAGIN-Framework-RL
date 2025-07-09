# SAGIN API Reference

This document provides comprehensive API documentation for the SAGIN simulation framework.

## 🏗️ Core Components

### SAGINNetwork
Main orchestrator for the entire SAGIN network simulation.

```python
from src.core.network import SAGINNetwork

network = SAGINNetwork(system_params)
```

#### Key Methods
- `setup_network_topology_with_grid(grid_config)` - Setup grid-based topology
- `add_vehicles(count, area_bounds, vehicle_type)` - Add vehicles to network
- `add_dynamic_uavs(count, area_bounds)` - Add dynamic UAVs
- `add_satellite_constellation(num_satellites, num_planes)` - Add satellites
- `initialize_simulation()` - Initialize simulation state
- `step(dt, verbose)` - Execute one simulation step
- `run_simulation(num_epochs, callback)` - Run complete simulation
- `get_performance_summary()` - Get performance metrics
- `get_network_state()` - Get current network state

### Configuration System

#### SAGINConfig
Complete configuration for SAGIN simulation.

```python
from config.grid_config import SAGINConfig, get_sagin_config

config = get_sagin_config("medium_demo")
system_params = config.get_system_parameters()
```

#### GridConfig
Grid topology configuration.

```python
from config.grid_config import GridConfig

grid = GridConfig(
    grid_rows=5,
    grid_cols=10,
    area_bounds=(0.0, 10000.0, 0.0, 5000.0)
)
```

**Properties:**
- `total_regions` - Total number of regions
- `region_width` - Width of each region
- `region_height` - Height of each region

**Methods:**
- `get_region_center(row, col)` - Get region center coordinates
- `get_region_id(row, col)` - Get region ID from grid position
- `get_grid_position(region_id)` - Get grid position from region ID

#### VehicleConfig
Vehicle configuration parameters.

```python
from config.grid_config import VehicleConfig

vehicles = VehicleConfig(
    random_vehicles=50,
    bus_vehicles=15,
    vehicle_speed_range=(5.0, 20.0)
)
```

#### UAVConfig
UAV configuration parameters.

```python
from config.grid_config import UAVConfig

uavs = UAVConfig(
    dynamic_uavs=10,
    uav_max_speed=20.0,
    uav_altitude=100.0,
    uav_cpu_capacity=1e9
)
```

#### SatelliteConfig
Satellite constellation configuration.

```python
from config.grid_config import SatelliteConfig

satellites = SatelliteConfig(
    num_satellites=12,
    num_planes=3,
    altitude=600000.0
)
```

#### SimulationConfig
Simulation parameters and settings.

```python
from config.grid_config import SimulationConfig

simulation = SimulationConfig(
    total_epochs=100,
    logging_level="medium",
    log_decisions=True,
    export_results=True
)
```

## 🎮 Demo Interface

### SAGINDemo
High-level interface for running simulations.

```python
from examples.sagin_demo import SAGINDemo

demo = SAGINDemo()
```

#### Methods
- `create_network(config_name)` - Create network from configuration
- `run_simulation(config_name)` - Run complete simulation
- `print_summary(network)` - Print simulation results
- `plot_results(network)` - Plot performance graphs
- `export_results(network)` - Export results to file

## 🚗 Vehicle Management

### VehicleManager
Manages all vehicles in the network.

```python
from src.core.vehicles import VehicleManager

vm = VehicleManager(system_params)
```

#### Methods
- `create_random_waypoint_vehicles(count, area_bounds)` - Create random vehicles
- `create_bus_route_vehicles(count, route_points)` - Create bus route vehicles
- `update_all_vehicles(current_time, dt)` - Update vehicle positions
- `assign_vehicles_to_regions(regions)` - Assign vehicles to regions
- `get_vehicles_in_region(region_id)` - Get vehicles in specific region

## 🚁 UAV Management

### UAVManager
Manages static and dynamic UAVs.

```python
from src.core.uavs import UAVManager

uav_mgr = UAVManager(system_params)
```

#### Methods
- `create_static_uav(region_id, position, cpu_capacity)` - Create static UAV
- `create_dynamic_uav(initial_position, cpu_capacity)` - Create dynamic UAV
- `get_static_uav_by_region(region_id)` - Get static UAV for region
- `get_available_dynamic_uavs_in_region(region_id)` - Get available dynamic UAVs
- `assign_dynamic_uav(uav_id, region_id, region_center, current_time)` - Assign dynamic UAV
- `update_all_uavs(current_time, dt)` - Update all UAVs

### UAV Classes

#### StaticUAV
Static UAV providing continuous coverage.

```python
from src.core.uavs import StaticUAV

static_uav = StaticUAV(uav_id, region_id, position, cpu_capacity)
```

**Methods:**
- `add_task(task)` - Add task to UAV queue
- `process_tasks(current_time, dt)` - Process queued tasks
- `make_offloading_decision(task, available_dynamic_uavs, satellite_available, current_time)` - Make offloading decision

#### DynamicUAV
Dynamic UAV that can be reassigned.

```python
from src.core.uavs import DynamicUAV

dynamic_uav = DynamicUAV(uav_id, initial_position, cpu_capacity)
```

**Methods:**
- `assign_to_region(region_id, region_center, current_time)` - Assign to new region
- `update_flight(current_time, dt)` - Update flight progress

## 🛰️ Satellite Management

### SatelliteConstellation
Manages satellite constellation.

```python
from src.core.satellites import SatelliteConstellation

sat_constellation = SatelliteConstellation(system_params)
```

#### Methods
- `create_constellation(num_satellites, num_planes)` - Create satellite constellation
- `update_all_satellites(current_time, dt)` - Update satellite positions
- `find_visible_satellites(ground_position)` - Find visible satellites
- `assign_task_to_satellite(task, ground_position)` - Assign task to satellite

## 📝 Task Management

### TaskManager
Central task management system.

```python
from src.core.tasks import TaskManager

task_mgr = TaskManager(system_params)
```

#### Methods
- `initialize_region_queues(region_ids, queue_size)` - Initialize task queues
- `generate_tasks(regions, vehicles_by_region, current_time, dt)` - Generate new tasks
- `get_tasks_for_region(region_id, max_tasks)` - Get pending tasks for region
- `mark_task_completed(task)` - Mark task as completed
- `mark_task_failed(task, reason)` - Mark task as failed
- `cleanup_expired_tasks(current_time)` - Remove expired tasks

### Task
Individual computational task.

```python
from src.core.types import Task

task = Task(
    id=1,
    source_vehicle_id=1,
    region_id=1,
    data_size_in=1.0,
    data_size_out=0.5,
    cpu_cycles=1e9,
    deadline=current_time + 10.0,
    creation_time=current_time
)
```

## 📡 Communication Models

### CommunicationModel
Advanced communication modeling.

```python
from src.models.communication import CommunicationModel

comm_model = CommunicationModel(system_params)
```

#### Methods
- `calculate_data_rate(tx_pos, rx_pos, tx_type, rx_type)` - Calculate data rate
- `calculate_transmission_delay(data_size, tx_pos, rx_pos, tx_type, rx_type)` - Calculate transmission delay
- `is_link_available(tx_pos, rx_pos, tx_type, rx_type)` - Check link availability

### ShannonCapacityModel
Shannon capacity calculations.

```python
from src.models.communication import ShannonCapacityModel

shannon = ShannonCapacityModel(system_params)
```

#### Methods
- `calculate_shannon_capacity(bandwidth_hz, transmit_power, channel_gain, noise_power)` - Basic Shannon capacity
- `calculate_practical_capacity(bandwidth_mhz, snr_db)` - Practical capacity with coding
- `calculate_adaptive_capacity(channel_conditions)` - Adaptive capacity based on conditions

## ⏱️ Latency Models

### LatencyModel
Comprehensive latency modeling.

```python
from src.models.latency import LatencyModel

latency_model = LatencyModel(system_params)
```

#### Methods
- `calculate_propagation_delay(source_pos, dest_pos)` - Calculate propagation delay
- `calculate_transmission_delay(data_size, source_pos, dest_pos, source_type, dest_type)` - Calculate transmission delay
- `calculate_end_to_end_latency(task, communication_path, processing_node_info)` - Calculate total latency

## 🤖 Reinforcement Learning

### SAGINRLEnvironment
RL environment for SAGIN optimization.

```python
from src.rl.environment import SAGINRLEnvironment

env = SAGINRLEnvironment(network, env_config)
```

#### Methods
- `reset()` - Reset environment
- `step(central_action, local_actions)` - Take action step
- `get_global_state()` - Get global state for central agent
- `get_local_state(region_id)` - Get local state for region

### CentralAgent
Central agent for dynamic UAV allocation.

```python
from src.rl.agents import CentralAgent

central_agent = CentralAgent(state_dim, action_dim, agent_config)
```

#### Methods
- `select_action(state, epsilon)` - Select action using epsilon-greedy
- `update(state, action, reward, next_state, done)` - Update agent

### LocalAgent
Local agent for task offloading decisions.

```python
from src.rl.agents import LocalAgent

local_agent = LocalAgent(state_dim, action_dim, agent_config)
```

## 📊 Utility Functions

### Configuration Utilities
```python
from config.grid_config import (
    get_sagin_config,
    list_available_configs,
    print_config_summary,
    create_custom_sagin_config
)

# Get predefined configuration
config = get_sagin_config("medium_demo")

# List all available configurations
configs = list_available_configs()

# Print configuration details
print_config_summary("highway_scenario")

# Create custom configuration
custom_config = create_custom_sagin_config(
    name="test_config",
    grid_rows=3,
    grid_cols=4,
    random_vehicles=25
)
```

## 🔢 Data Types

### Core Types
```python
from src.core.types import (
    Position,
    Velocity,
    SystemParameters,
    TaskStatus,
    TaskDecision,
    NodeType
)
```

### Enums
- `TaskStatus`: PENDING, IN_PROGRESS, COMPLETED, FAILED, DEADLINE_MISSED
- `TaskDecision`: LOCAL, DYNAMIC, SATELLITE
- `NodeType`: VEHICLE, STATIC_UAV, DYNAMIC_UAV, SATELLITE

### SystemParameters
```python
system_params = SystemParameters(
    epoch_duration=1.0,
    total_epochs=100,
    min_rate_threshold=1.0,
    uav_max_speed=20.0,
    uav_altitude=100.0,
    max_load_imbalance=0.3,
    propagation_speed=3e8,
    min_energy_threshold=1000.0
)
```

## ⚡ Performance Tips

1. **Use appropriate configurations** for your use case
2. **Enable low logging level** for large simulations
3. **Batch process results** for analysis
4. **Monitor memory usage** with large networks
5. **Use progress callbacks** for long simulations

## 🐛 Error Handling

Common exceptions and how to handle them:

```python
try:
    network = demo.run_simulation("invalid_config")
except KeyError:
    print("Configuration not found")

try:
    network.step()
except Exception as e:
    print(f"Simulation error: {e}")
```

##### State and Metrics
```python
def get_network_state(self) -> Dict[str, Any]
    """Get complete network state for RL agents."""

def get_performance_summary(self) -> Dict[str, Any]
    """Get performance summary."""

def export_results(self, filename: str) -> None
    """Export simulation results to JSON file."""
```

### Vehicle Classes

#### Vehicle
```python
class Vehicle:
    def __init__(self, vehicle_id: int, initial_position: Position, 
                 mobility_model: MobilityModel, task_characteristics: TaskCharacteristics)
    
    # Properties
    position: Position           # Current 3D position
    velocity: Velocity          # Current 3D velocity  
    mobility_model: MobilityModel # Movement pattern
    current_region_id: Optional[int] # Assigned region
```

#### MobilityModel Subclasses
```python
class RandomWaypointMobility(MobilityModel):
    def __init__(self, area_bounds: Tuple[float, float, float, float],
                 min_speed: float = 5.0, max_speed: float = 15.0,
                 pause_time: float = 1.0)

class FixedRouteMobility(MobilityModel):
    def __init__(self, route_points: List[Position], speed: float = 10.0, 
                 loop: bool = True)
```

### UAV Classes

#### StaticUAV
```python
class StaticUAV:
    def __init__(self, uav_id: int, assigned_region_id: int, position: Position,
                 cpu_capacity: float = 1e9, battery_capacity: float = 100000.0)
    
    # Key Methods
    def add_task(self, task: Task) -> bool
    def process_tasks(self, dt: float) -> List[Task]
    def make_offloading_decision(self, task: Task, dynamic_uav_count: int, 
                               satellite_available: bool, current_time: float) -> TaskDecision
    
    # Properties
    total_workload: float        # Current CPU workload
    queue_length: int           # Number of queued tasks
    current_energy: float       # Remaining battery energy
    is_available: bool          # Energy > threshold
```

#### DynamicUAV
```python
class DynamicUAV:
    def __init__(self, uav_id: int, initial_position: Position,
                 cpu_capacity: float = 1e9, battery_capacity: float = 100000.0,
                 max_speed: float = 25.0)
    
    # Key Methods
    def move_to_region(self, target_region_center: Position, current_time: float)
    def update_position(self, current_time: float, dt: float) -> Position
    
    # Properties
    status: UAVStatus           # IDLE, ACTIVE, FLYING, INACTIVE
    current_region_id: Optional[int]
    target_region_id: Optional[int]
```

### Task Classes

#### Task
```python
class Task:
    def __init__(self, task_id: int, vehicle_id: int, region_id: int,
                 cpu_cycles: float, data_size_in: float, data_size_out: float,
                 deadline: float, priority: float = 1.0)
    
    # Properties
    task_type: TaskType         # NORMAL, COMPUTATION_INTENSIVE, etc.
    status: TaskStatus          # PENDING, PROCESSING, COMPLETED, FAILED
    decision: Optional[TaskDecision] # LOCAL, DYNAMIC, SATELLITE
    completion_time: Optional[float]
    total_latency: Optional[float]
```

#### TaskGenerator
```python
class TaskGenerator:
    def generate_tasks_for_region(self, region: Region, vehicle_ids: List[int],
                                 current_time: float, dt: float) -> List[Task]
```

### Satellite Classes

#### Satellite
```python
class Satellite:
    def __init__(self, satellite_id: int, orbital_plane: int, position_in_plane: int,
                 altitude: float = 600000.0, cpu_capacity: float = 2e9)
    
    # Key Methods
    def update_position(self, current_time: float, dt: float)
    def is_visible_from(self, ground_position: Position) -> bool
    
    # Properties
    orbital_period: float       # Orbital period in seconds
    current_position: Position  # Current 3D position
```

#### SatelliteConstellation
```python
class SatelliteConstellation:
    def create_constellation(self, num_satellites: int, num_planes: int) -> List[int]
    def find_visible_satellites(self, ground_position: Position) -> List[Satellite]
    def assign_task_to_satellite(self, task: Task, ground_position: Position) -> bool
```

## Data Structures

### Core Types
```python
@dataclass
class Position:
    x: float
    y: float  
    z: float

@dataclass
class Velocity:
    vx: float
    vy: float
    vz: float

@dataclass
class Region:
    id: int
    name: str
    center: Position
    radius: float
    base_intensity: float
    current_intensity: float = 1.0
    vehicle_ids: List[int] = field(default_factory=list)
    static_uav_id: Optional[int] = None
    dynamic_uav_ids: List[int] = field(default_factory=list)
```

### Enums
```python
class TaskType(Enum):
    NORMAL = "normal"
    COMPUTATION_INTENSIVE = "computation_intensive"
    DATA_INTENSIVE = "data_intensive"
    LATENCY_SENSITIVE = "latency_sensitive"

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskDecision(Enum):
    LOCAL = "local"
    DYNAMIC = "dynamic"
    SATELLITE = "satellite"

class UAVStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FLYING = "flying"
    INACTIVE = "inactive"
```

### System Configuration
```python
@dataclass
class SystemParameters:
    # Simulation parameters
    epoch_duration: float = 2.0
    total_epochs: int = 1000
    
    # Physical parameters
    area_width: float = 10000.0
    area_height: float = 10000.0
    uav_altitude: float = 100.0
    satellite_altitude: float = 600000.0
    
    # Performance parameters
    max_uav_speed: float = 25.0
    min_rate_threshold: float = 1.0
    uav_cpu_capacity: float = 1e9
    satellite_cpu_capacity: float = 2e9
    
    # Energy parameters
    uav_battery_capacity: float = 100000.0
    uav_hover_power: float = 500.0
    uav_flight_efficiency: float = 1000.0
    energy_threshold: float = 0.1
    
    # Communication parameters
    frequency_ghz: float = 2.4
    bandwidth_mhz: float = 20.0
    tx_power_dbm: float = 23.0
    noise_power_dbm: float = -174.0
    
    # Task parameters
    task_arrival_rate: float = 0.5
    cpu_cycles_mean: float = 5e8
    data_size_mean: float = 1.0
    deadline_mean: float = 10.0
    
    # Optimization parameters
    max_load_imbalance: float = 0.3
    alpha1: float = 0.1  # Load imbalance penalty
    alpha2: float = 0.9  # Energy violation penalty
```

## Communication Models

### CommunicationModel
```python
class CommunicationModel:
    def calculate_data_rate(self, tx_pos: Position, rx_pos: Position,
                          tx_type: str, rx_type: str) -> float
        """Calculate Shannon capacity between two nodes."""
    
    def calculate_path_loss(self, distance: float, frequency: float,
                          link_type: str) -> float
        """Calculate path loss for different link types."""
```

### LatencyModel  
```python
class LatencyModel:
    def calculate_end_to_end_latency(self, task: Task, source_pos: Position,
                                   dest_pos: Position, data_rate: float,
                                   processing_queue: float, cpu_capacity: float) -> float
        """Calculate total latency with 4 components."""
    
    def calculate_propagation_delay(self, distance: float) -> float
    def calculate_transmission_delay(self, data_size: float, data_rate: float) -> float
    def calculate_queuing_delay(self, workload: float, cpu_capacity: float) -> float
    def calculate_processing_delay(self, cpu_cycles: float, cpu_capacity: float) -> float
```

## Usage Examples

### Basic Network Setup
```python
from src.core.network import SAGINNetwork
from src.core.types import SystemParameters

# Custom configuration
params = SystemParameters(
    epoch_duration=1.0,
    uav_altitude=150.0,
    max_uav_speed=30.0
)

# Create network
network = SAGINNetwork(params)
network.setup_network_topology((0, 5000, 0, 5000), num_regions=3)

# Add elements
vehicle_ids = network.add_vehicles(30, (0, 5000, 0, 5000), "random")
bus_ids = network.add_vehicles(10, (0, 5000, 0, 5000), "bus")
uav_ids = network.add_dynamic_uavs(5, (0, 5000, 0, 5000))
sat_ids = network.add_satellite_constellation(6, 2)
```

### Event-Driven Simulation
```python
# Add burst events
network.task_manager.add_burst_event(
    region_id=1, start_time=100.0, duration=50.0, amplitude=2.5
)

# Step-by-step control
network.initialize_simulation()
for epoch in range(200):
    results = network.step(verbose=False)
    
    # Monitor performance
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Success rate = {network.metrics.success_rate:.3f}")
    
    # Dynamic adjustments
    if epoch == 150:
        network.add_dynamic_uavs(2, (0, 5000, 0, 5000))

# Export results
network.export_results("simulation_results.json")
```

### Custom Components
```python
from src.core.vehicles import MobilityModel
from src.core.tasks import TaskGenerator

class CustomMobility(MobilityModel):
    def update_position(self, current_pos, current_time, dt):
        # Custom movement logic
        return new_position

class CustomTaskGen(TaskGenerator):
    def generate_task(self, vehicle_id, region_id, current_time, task_type):
        # Custom task characteristics
        return custom_task
```

## Error Handling

### Common Exceptions
```python
# Configuration errors
SystemParameterError: Invalid system parameters
NetworkTopologyError: Invalid network configuration

# Runtime errors  
EnergyConstraintViolation: UAV energy below threshold
CommunicationError: Unable to establish communication link
TaskAllocationError: Cannot allocate task to any resource

# Data errors
InvalidPositionError: Invalid 3D coordinates
InvalidTaskError: Task parameters out of bounds
```

### Best Practices
- Always check UAV energy levels before task assignment
- Validate network connectivity before simulation start
- Monitor load imbalance and adjust dynamic UAV allocation
- Use appropriate epoch duration for system dynamics
- Export results regularly for long simulations
