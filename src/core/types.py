"""
Core data structures and types for the SAGIN system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class NodeType(Enum):
    """Types of network nodes."""
    VEHICLE = "vehicle"
    STATIC_UAV = "static_uav"
    DYNAMIC_UAV = "dynamic_uav"
    SATELLITE = "satellite"


class TaskDecision(Enum):
    """Task offloading decisions."""
    LOCAL = "local"
    DYNAMIC = "dynamic"
    SATELLITE = "satellite"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEADLINE_MISSED = "deadline_missed"


@dataclass
class Position:
    """3D position with coordinates."""
    x: float
    y: float
    z: float = 0.0  # altitude for UAVs and satellites
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass
class Velocity:
    """3D velocity vector."""
    vx: float
    vy: float
    vz: float = 0.0
    
    def magnitude(self) -> float:
        """Calculate velocity magnitude."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


@dataclass
class Task:
    """Computational task with all required attributes."""
    id: int
    source_vehicle_id: int
    region_id: int
    
    # Task characteristics
    data_size_in: float  # input data size (MB)
    data_size_out: float  # output data size (MB)
    cpu_cycles: float  # required CPU cycles
    deadline: float  # task deadline (seconds)
    
    # Timing information
    creation_time: float
    arrival_time: float = 0.0
    start_time: float = 0.0
    completion_time: float = 0.0
    
    # Assignment and routing
    assigned_node_id: Optional[int] = None
    route: List[int] = None  # multi-hop route as list of node IDs
    decision: Optional[TaskDecision] = None
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    def __post_init__(self):
        if self.route is None:
            self.route = []
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_deadline_violated(self) -> bool:
        """Check if task deadline is violated."""
        return self.completion_time > self.deadline if self.completion_time > 0 else False


@dataclass
class CommunicationLink:
    """Communication link between two nodes."""
    source_id: int
    destination_id: int
    bandwidth: float  # MHz
    channel_gain: float  # dB
    is_available: bool = True
    
    def data_rate(self, transmit_power: float, noise_power: float) -> float:
        """Calculate Shannon capacity data rate."""
        if not self.is_available:
            return 0.0
        
        # Convert channel gain from dB to linear scale
        gain_linear = 10 ** (self.channel_gain / 10)
        snr = transmit_power * gain_linear / noise_power
        
        # Shannon capacity formula
        return self.bandwidth * np.log2(1 + snr)


@dataclass
class Region:
    """Geographic region in the SAGIN network."""
    id: int
    name: str
    center: Position
    radius: float
    
    # Task generation parameters
    base_intensity: float = 1.0  # baseline task arrival rate
    current_intensity: float = 1.0  # current task intensity
    
    # Network elements in this region
    static_uav_id: Optional[int] = None
    dynamic_uav_ids: List[int] = None
    vehicle_ids: List[int] = None
    
    def __post_init__(self):
        if self.dynamic_uav_ids is None:
            self.dynamic_uav_ids = []
        if self.vehicle_ids is None:
            self.vehicle_ids = []
    
    def contains_position(self, position: Position) -> bool:
        """Check if a position is within this region."""
        return self.center.distance_to(position) <= self.radius


@dataclass
class BurstEvent:
    """Sudden burst event affecting task generation."""
    region_id: int
    start_time: float
    end_time: float
    amplitude: float  # multiplicative factor for task intensity
    
    def is_active(self, current_time: float) -> bool:
        """Check if burst event is currently active."""
        return self.start_time <= current_time <= self.end_time


# System-wide constants and parameters
@dataclass
class SystemParameters:
    """Global system parameters."""
    # Time parameters
    epoch_duration: float = 1.0  # seconds
    total_epochs: int = 1000
    
    # Communication parameters
    min_rate_threshold: float = 1.0  # Mbps
    propagation_speed: float = 3e8  # m/s (speed of light)
    noise_power: float = 1e-13  # W
    
    # Energy parameters
    uav_hover_power: float = 100.0  # W
    uav_flight_energy_factor: float = 0.1  # J/m
    energy_per_cpu_cycle: float = 1e-9  # J/cycle
    comm_energy_factor: float = 1e-6  # J/bit
    min_energy_threshold: float = 1000.0  # J
    
    # UAV parameters
    uav_max_speed: float = 20.0  # m/s
    uav_altitude: float = 100.0  # m
    
    # Load balancing
    max_load_imbalance: float = 0.3
    
    # Reward function weights
    alpha1: float = 0.1  # load imbalance penalty
    alpha2: float = 1.0  # energy violation penalty
    
    # RL parameters
    discount_factor: float = 0.95
    learning_rate: float = 1e-4
