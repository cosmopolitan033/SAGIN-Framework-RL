# SAGIN API Reference

## Core Classes

### SAGINNetwork

Main orchestrator class for the SAGIN simulation system.

#### Constructor
```python
SAGINNetwork(system_params: Optional[SystemParameters] = None)
```

#### Key Methods

##### Network Setup
```python
def setup_network_topology(self, area_bounds: Tuple[float, float, float, float], 
                          num_regions: int = 5) -> None
    """Setup network topology with regions."""

def create_region(self, name: str, center: Position, radius: float,
                 base_intensity: float = 1.0) -> int
    """Create a new region in the network."""
```

##### Network Elements
```python
def add_vehicles(self, count: int, area_bounds: Tuple[float, float, float, float],
                vehicle_type: str = "random") -> List[int]
    """Add vehicles to the network. vehicle_type: 'random' or 'bus'"""

def add_dynamic_uavs(self, count: int, area_bounds: Tuple[float, float, float, float]) -> List[int]
    """Add dynamic UAVs to the network."""

def add_satellite_constellation(self, num_satellites: int = 12, 
                              num_planes: int = 3) -> List[int]
    """Add satellite constellation to the network."""
```

##### Simulation Control
```python
def initialize_simulation(self) -> None
    """Initialize the simulation."""

def step(self, dt: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]
    """Execute one simulation step."""

def run_simulation(self, num_epochs: int, progress_callback: Optional[callable] = None) -> None
    """Run the simulation for a specified number of epochs."""
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
