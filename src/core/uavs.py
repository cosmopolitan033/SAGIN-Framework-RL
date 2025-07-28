"""
UAV models for static and dynamic UAVs in the SAGIN system.
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .types import (
    Position, Velocity, NodeType, SystemParameters, 
    Task, TaskStatus, TaskDecision
)


class UAVStatus(Enum):
    """UAV operational status."""
    ACTIVE = "active"
    FLYING = "flying"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


@dataclass
class UAV:
    """Base UAV class with common attributes."""
    
    id: int
    node_type: NodeType
    position: Position = field(default_factory=lambda: Position(0.0, 0.0, 100.0))
    velocity: Velocity = field(default_factory=lambda: Velocity(0.0, 0.0, 0.0))
    
    # Physical capabilities
    max_speed: float = 250.0  # m/s
    altitude: float = 100.0  # m
    
    # Energy system
    battery_capacity: float = 100000.0  # J (100 kJ)
    current_energy: float = 100000.0  # J (start fully charged)
    min_energy_threshold: float = 1000.0  # J
    
    # Communication
    transmit_power: float = 10.0  # W
    antenna_gain: float = 10.0  # dB
    communication_range: float = 1000.0  # m
    
    # Computing
    cpu_capacity: float = 1e6  # cycles per second
    current_workload: float = 0.0  # current CPU cycles
    
    # Task management
    task_queue: List[Task] = field(default_factory=list)
    processing_tasks: List[Task] = field(default_factory=list)
    completed_tasks: List[int] = field(default_factory=list)
    
    # Status
    status: UAVStatus = UAVStatus.ACTIVE
    assigned_region_id: Optional[int] = None
    
    def __post_init__(self):
        """Initialize position altitude."""
        if self.position.z == 0.0:
            self.position.z = self.altitude
    
    @property
    def is_available(self) -> bool:
        """Check if UAV is available for task processing."""
        return self.status == UAVStatus.ACTIVE and self.current_energy > self.min_energy_threshold
    
    @property
    def queue_length(self) -> int:
        """Get current queue length."""
        return len(self.task_queue)
    
    @property
    def total_workload(self) -> float:
        """Get total workload (queued + processing)."""
        queue_workload = sum(task.cpu_cycles for task in self.task_queue)
        processing_workload = sum(task.cpu_cycles for task in self.processing_tasks)
        return queue_workload + processing_workload
    
    def add_task(self, task: Task) -> bool:
        """Add a task to the UAV's queue."""
        if not self.is_available:
            return False
        
        # Check if task can potentially be completed given current queue
        estimated_queue_time = self.total_workload / self.cpu_capacity
        estimated_processing_time = task.cpu_cycles / self.cpu_capacity
        estimated_completion_time = estimated_queue_time + estimated_processing_time
        
        # If the task cannot possibly meet its deadline, reject it
        if estimated_completion_time > task.deadline:
            task.status = TaskStatus.DEADLINE_MISSED
            return False
        
        self.task_queue.append(task)
        task.assigned_node_id = self.id
        task.status = TaskStatus.PENDING
        return True
    
    def process_tasks(self, current_time: float, dt: float) -> List[Task]:
        """Process tasks and return completed ones."""
        completed = []
        
        # Start new tasks if CPU capacity available
        remaining_capacity = self.cpu_capacity
        for task in self.processing_tasks:
            remaining_capacity -= task.cpu_cycles / (current_time - task.start_time + 1e-6)
        
        # Move tasks from queue to processing
        while self.task_queue and remaining_capacity > 0:
            task = self.task_queue.pop(0)
            
            # Check if task can still meet deadline with current processing time
            processing_time = task.cpu_cycles / self.cpu_capacity
            expected_completion_time = current_time + processing_time
            
            if expected_completion_time <= task.deadline:  # More realistic deadline check
                task.status = TaskStatus.IN_PROGRESS
                task.start_time = current_time
                self.processing_tasks.append(task)
                remaining_capacity -= task.cpu_cycles / self.cpu_capacity  # Use actual processing rate
            else:
                task.status = TaskStatus.DEADLINE_MISSED
                completed.append(task)
        
        # Process current tasks
        still_processing = []
        for task in self.processing_tasks:
            processing_time = current_time - task.start_time
            required_time = task.cpu_cycles / self.cpu_capacity
            
            print(f"DEBUG: Task {task.id} processing - current_time={current_time}, start_time={task.start_time}, processing_time={processing_time}, required_time={required_time}")
            
            if processing_time >= required_time:
                # Task completed
                task.status = TaskStatus.COMPLETED
                task.completion_time = current_time
                print(f"DEBUG: UAV {self.id} completed task {task.id} at current_time={current_time}")
                completed.append(task)
                self.completed_tasks.append(task.id)
            else:
                still_processing.append(task)
        
        self.processing_tasks = still_processing
        return completed
    
    def consume_energy(self, dt: float, flight_distance: float = 0.0, 
                      comm_data: float = 0.0) -> None:
        """Consume energy for flight, communication, and computation."""
        # Flight energy
        hover_energy = 100.0 * dt  # W * s
        flight_energy = 0.1 * flight_distance  # J/m * m
        
        # Communication energy
        comm_energy = 1e-6 * comm_data  # J/bit * bits
        
        # Computation energy
        comp_energy = 1e-9 * self.cpu_capacity * dt  # J/cycle * cycles/s * s
        
        total_energy = hover_energy + flight_energy + comm_energy + comp_energy
        self.current_energy = max(0.0, self.current_energy - total_energy)
        
        # Update status based on energy
        if self.current_energy < self.min_energy_threshold:
            self.status = UAVStatus.INACTIVE
    
    def can_reach_position(self, target_position: Position, max_time: float) -> bool:
        """Check if UAV can reach target position within max_time."""
        distance = self.position.distance_to(target_position)
        required_time = distance / self.max_speed
        return required_time <= max_time
    
    def estimate_flight_time(self, target_position: Position) -> float:
        """Estimate flight time to target position."""
        distance = self.position.distance_to(target_position)
        return distance / self.max_speed


@dataclass
class StaticUAV(UAV):
    """Static UAV providing continuous coverage for a region."""
    
    def __init__(self, uav_id: int, region_id: int, position: Position, 
                 cpu_capacity: float = 1e8):  # Increased to 1e8 (100M cycles/s) for more realistic performance
        super().__init__(
            id=uav_id,
            node_type=NodeType.STATIC_UAV,
            position=position,
            cpu_capacity=cpu_capacity
        )
        self.assigned_region_id = region_id
        self.status = UAVStatus.ACTIVE
    
    def make_offloading_decision(self, task: Task, available_dynamic_uavs: int,
                               satellite_available: bool, current_time: float) -> TaskDecision:
        """Make hierarchical offloading decision for a task.
        
        Hierarchy: Vehicle → Static UAV → Dynamic UAV → Satellite
        
        Static UAV tries to handle locally first, then offloads to dynamic UAV if overloaded,
        and finally to satellite if both static and dynamic UAVs are unavailable/overloaded.
        """
        
        # Step 1: Try to handle locally (Static UAV processing)
        local_completion_time = current_time + self.estimate_local_completion_time(task)
        
        # Check if static UAV can handle the task efficiently
        is_overloaded = self.total_workload >= self.cpu_capacity * 1.5  # Lowered threshold for better hierarchy
        can_meet_deadline = local_completion_time <= task.deadline
        
        if not is_overloaded and can_meet_deadline:
            return TaskDecision.LOCAL
        
        # Step 2: Offload to Dynamic UAV if static UAV is overloaded or can't meet deadline
        if available_dynamic_uavs > 0:
            return TaskDecision.DYNAMIC
        
        # Step 3: Offload to Satellite if no dynamic UAVs available or all are overloaded
        if satellite_available:
            return TaskDecision.SATELLITE
        
        # Last resort: Process locally even if overloaded (better than dropping the task)
        return TaskDecision.LOCAL
    
    def estimate_local_completion_time(self, task: Task) -> float:
        """Estimate time to complete task locally."""
        queue_time = self.total_workload / self.cpu_capacity
        processing_time = task.cpu_cycles / self.cpu_capacity
        return queue_time + processing_time
    
    def get_region_state(self) -> Dict:
        """Get current state of the region for RL agent."""
        return {
            'queue_length': self.queue_length,
            'total_workload': self.total_workload,
            'energy_level': self.current_energy / self.battery_capacity,
            'cpu_utilization': min(1.0, self.total_workload / self.cpu_capacity),
            'available': self.is_available
        }


@dataclass
class DynamicUAV(UAV):
    """Dynamic UAV that can be reassigned between regions."""
    
    def __init__(self, uav_id: int, initial_position: Position, 
                 cpu_capacity: float = 1e8):  # Increased to 1e8 (100M cycles/s) for more realistic performance
        super().__init__(
            id=uav_id,
            node_type=NodeType.DYNAMIC_UAV,
            position=initial_position,
            cpu_capacity=cpu_capacity
        )
        
        # Dynamic UAV specific attributes
        self.target_region_id: Optional[int] = None
        self.flight_start_time: Optional[float] = None
        self.flight_destination: Optional[Position] = None
        self.availability_indicator: int = 1  # 1 if available, 0 if flying
    
    def assign_to_region(self, region_id: int, region_center: Position, 
                        current_time: float) -> float:
        """Assign UAV to a new region and start flight."""
        if self.assigned_region_id == region_id:
            return 0.0  # Already in target region
        
        # Start flight to new region
        self.target_region_id = region_id
        self.flight_destination = Position(region_center.x, region_center.y, self.altitude)
        self.flight_start_time = current_time
        self.status = UAVStatus.FLYING
        self.availability_indicator = 0
        
        # Estimate flight time
        flight_time = self.estimate_flight_time(self.flight_destination)
        return flight_time
    
    def update_flight(self, current_time: float, dt: float) -> bool:
        """Update flight progress. Returns True if flight completed."""
        if self.status != UAVStatus.FLYING or self.flight_destination is None:
            return False
        
        # Calculate flight progress
        flight_duration = current_time - self.flight_start_time
        total_distance = self.position.distance_to(self.flight_destination)
        
        if total_distance < 1.0:  # Reached destination
            self.position = self.flight_destination
            self.assigned_region_id = self.target_region_id
            self.status = UAVStatus.ACTIVE
            self.availability_indicator = 1
            self.flight_destination = None
            self.flight_start_time = None
            self.target_region_id = None
            return True
        
        # Update position during flight
        direction_vector = self.flight_destination - self.position
        distance_to_move = min(self.max_speed * dt, total_distance)
        
        if total_distance > 0:
            move_ratio = distance_to_move / total_distance
            self.position = self.position + Position(
                direction_vector.x * move_ratio,
                direction_vector.y * move_ratio,
                direction_vector.z * move_ratio
            )
        
        # Consume flight energy
        self.consume_energy(dt, flight_distance=distance_to_move)
        
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if dynamic UAV is available (not flying)."""
        return (super().is_available and 
                self.status == UAVStatus.ACTIVE and 
                self.availability_indicator == 1)


class UAVManager:
    """Manages all UAVs in the SAGIN system."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.static_uavs: Dict[int, StaticUAV] = {}
        self.dynamic_uavs: Dict[int, DynamicUAV] = {}
        self.next_uav_id = 1
        # Track UAV repositioning history for visualization
        self.repositioning_history: Dict[int, Dict[int, Dict[str, any]]] = {}  # {uav_id: {epoch: {data}}}
    
    def record_repositioning(self, uav_id: int, epoch: int, old_region: int, new_region: int, status: str):
        """Record UAV repositioning for tracking."""
        if uav_id not in self.repositioning_history:
            self.repositioning_history[uav_id] = {}
        
        self.repositioning_history[uav_id][epoch] = {
            'old_region_id': old_region,
            'region_id': new_region,
            'status': status,
            'timestamp': epoch
        }
    
    def create_static_uav(self, region_id: int, position: Position,
                         cpu_capacity: float = 1e8) -> int:  # Increased to 1e8 (100M cycles/s)
        """Create a static UAV for a region."""
        uav_id = self.next_uav_id
        self.next_uav_id += 1
        
        static_uav = StaticUAV(uav_id, region_id, position, cpu_capacity)
        self.static_uavs[uav_id] = static_uav
        
        return uav_id
    
    def create_dynamic_uav(self, initial_position: Position,
                          cpu_capacity: float = 1e8) -> int:  # Increased to 1e8 (100M cycles/s)
        """Create a dynamic UAV."""
        uav_id = self.next_uav_id
        self.next_uav_id += 1
        
        dynamic_uav = DynamicUAV(uav_id, initial_position, cpu_capacity)
        self.dynamic_uavs[uav_id] = dynamic_uav
        
        return uav_id
    
    def get_static_uav_by_region(self, region_id: int) -> Optional[StaticUAV]:
        """Get static UAV assigned to a region."""
        for uav in self.static_uavs.values():
            if uav.assigned_region_id == region_id:
                return uav
        return None
    
    def get_available_dynamic_uavs_in_region(self, region_id: int) -> List[DynamicUAV]:
        """Get available dynamic UAVs in a specific region."""
        return [
            uav for uav in self.dynamic_uavs.values()
            if uav.assigned_region_id == region_id and uav.is_available
        ]
    
    def get_dynamic_uav_allocation(self) -> Dict[int, int]:
        """Get current allocation of dynamic UAVs to regions."""
        allocation = {}
        for uav in self.dynamic_uavs.values():
            if uav.assigned_region_id is not None and uav.is_available:
                region_id = uav.assigned_region_id
                allocation[region_id] = allocation.get(region_id, 0) + 1
        return allocation
    
    def assign_dynamic_uav(self, uav_id: int, region_id: int, 
                          region_center: Position, current_time: float) -> bool:
        """Assign a dynamic UAV to a region."""
        if uav_id not in self.dynamic_uavs:
            return False
        
        uav = self.dynamic_uavs[uav_id]
        flight_time = uav.assign_to_region(region_id, region_center, current_time)
        
        return True
    
    def update_all_uavs(self, current_time: float, dt: float) -> Dict[str, List[Task]]:
        """Update all UAVs and return completed tasks."""
        results = {
            'static_completed': [],
            'dynamic_completed': []
        }
        
        # Update static UAVs
        for uav in self.static_uavs.values():
            completed = uav.process_tasks(current_time, dt)
            results['static_completed'].extend(completed)
            uav.consume_energy(dt)
        
        # Update dynamic UAVs
        for uav in self.dynamic_uavs.values():
            # Update flight if in progress
            uav.update_flight(current_time, dt)
            
            # Process tasks if available
            if uav.is_available:
                completed = uav.process_tasks(current_time, dt)
                results['dynamic_completed'].extend(completed)
            
            # Consume energy (flight energy handled in update_flight)
            if uav.status == UAVStatus.ACTIVE:
                uav.consume_energy(dt)
        
        return results
    
    def get_system_state(self) -> Dict:
        """Get complete system state for RL agents."""
        static_states = {}
        dynamic_states = {}
        
        for region_id, uav in self.static_uavs.items():
            static_states[region_id] = uav.get_region_state()
        
        for uav_id, uav in self.dynamic_uavs.items():
            dynamic_states[uav_id] = {
                'assigned_region': uav.assigned_region_id,
                'is_available': uav.is_available,
                'position': (uav.position.x, uav.position.y, uav.position.z),
                'energy_level': uav.current_energy / uav.battery_capacity,
                'queue_length': uav.queue_length,
                'status': uav.status.value
            }
        
        return {
            'static_uavs': static_states,
            'dynamic_uavs': dynamic_states,
            'total_dynamic_available': sum(1 for uav in self.dynamic_uavs.values() if uav.is_available)
        }
