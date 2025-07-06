"""
Vehicle models and mobility patterns for the SAGIN system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .types import Position, Velocity, NodeType, SystemParameters, Task, Region


class MobilityModel(ABC):
    """Abstract base class for vehicle mobility models."""
    
    @abstractmethod
    def update_position(self, current_pos: Position, current_time: float, dt: float) -> Position:
        """Update vehicle position based on mobility model."""
        pass
    
    @abstractmethod
    def get_velocity(self, current_time: float) -> Velocity:
        """Get current velocity vector."""
        pass


class RandomWaypointMobility(MobilityModel):
    """Random waypoint mobility model for vehicles."""
    
    def __init__(self, area_bounds: Tuple[float, float, float, float], 
                 min_speed: float = 5.0, max_speed: float = 15.0,
                 pause_time: float = 0.0):
        """
        Initialize random waypoint mobility.
        
        Args:
            area_bounds: (min_x, max_x, min_y, max_y) boundaries
            min_speed: Minimum speed in m/s
            max_speed: Maximum speed in m/s
            pause_time: Pause time at waypoints in seconds
        """
        self.area_bounds = area_bounds
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.pause_time = pause_time
        
        # Current mobility state
        self.current_waypoint = self._generate_waypoint()
        self.current_speed = np.random.uniform(min_speed, max_speed)
        self.pause_remaining = 0.0
        self.current_velocity = Velocity(0.0, 0.0, 0.0)
    
    def _generate_waypoint(self) -> Position:
        """Generate a random waypoint within area bounds."""
        x = np.random.uniform(self.area_bounds[0], self.area_bounds[1])
        y = np.random.uniform(self.area_bounds[2], self.area_bounds[3])
        return Position(x, y, 0.0)
    
    def update_position(self, current_pos: Position, current_time: float, dt: float) -> Position:
        """Update position using random waypoint model."""
        # If pausing, don't move
        if self.pause_remaining > 0:
            self.pause_remaining -= dt
            self.current_velocity = Velocity(0.0, 0.0, 0.0)
            return current_pos
        
        # Calculate direction to waypoint
        dx = self.current_waypoint.x - current_pos.x
        dy = self.current_waypoint.y - current_pos.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # If reached waypoint, select new one
        if distance < 1.0:  # 1 meter threshold
            self.current_waypoint = self._generate_waypoint()
            self.current_speed = np.random.uniform(self.min_speed, self.max_speed)
            self.pause_remaining = self.pause_time
            return current_pos
        
        # Move towards waypoint
        direction_x = dx / distance
        direction_y = dy / distance
        
        # Update velocity
        self.current_velocity = Velocity(
            direction_x * self.current_speed,
            direction_y * self.current_speed,
            0.0
        )
        
        # Calculate new position
        move_distance = min(self.current_speed * dt, distance)
        new_x = current_pos.x + direction_x * move_distance
        new_y = current_pos.y + direction_y * move_distance
        
        return Position(new_x, new_y, 0.0)
    
    def get_velocity(self, current_time: float) -> Velocity:
        """Get current velocity vector."""
        return self.current_velocity


class FixedRouteMobility(MobilityModel):
    """Fixed route mobility for predictable vehicles like buses."""
    
    def __init__(self, route_points: List[Position], speed: float = 10.0, 
                 loop: bool = True, stop_time: float = 5.0):
        """
        Initialize fixed route mobility.
        
        Args:
            route_points: List of waypoints defining the route
            speed: Constant speed in m/s
            loop: Whether to loop back to start after reaching end
            stop_time: Time to stop at each waypoint
        """
        self.route_points = route_points
        self.speed = speed
        self.loop = loop
        self.stop_time = stop_time
        
        # Current state
        self.current_waypoint_idx = 0
        self.stop_remaining = 0.0
        self.current_velocity = Velocity(0.0, 0.0, 0.0)
    
    def update_position(self, current_pos: Position, current_time: float, dt: float) -> Position:
        """Update position along fixed route."""
        if len(self.route_points) == 0:
            return current_pos
        
        # If stopping at waypoint
        if self.stop_remaining > 0:
            self.stop_remaining -= dt
            self.current_velocity = Velocity(0.0, 0.0, 0.0)
            return current_pos
        
        target_waypoint = self.route_points[self.current_waypoint_idx]
        
        # Calculate direction to target waypoint
        dx = target_waypoint.x - current_pos.x
        dy = target_waypoint.y - current_pos.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # If reached waypoint
        if distance < 1.0:  # 1 meter threshold
            self.stop_remaining = self.stop_time
            
            # Move to next waypoint
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.route_points):
                if self.loop:
                    self.current_waypoint_idx = 0
                else:
                    self.current_waypoint_idx = len(self.route_points) - 1
            
            return current_pos
        
        # Move towards target waypoint
        direction_x = dx / distance
        direction_y = dy / distance
        
        # Update velocity
        self.current_velocity = Velocity(
            direction_x * self.speed,
            direction_y * self.speed,
            0.0
        )
        
        # Calculate new position
        move_distance = min(self.speed * dt, distance)
        new_x = current_pos.x + direction_x * move_distance
        new_y = current_pos.y + direction_y * move_distance
        
        return Position(new_x, new_y, 0.0)
    
    def get_velocity(self, current_time: float) -> Velocity:
        """Get current velocity vector."""
        return self.current_velocity


@dataclass
class Vehicle:
    """Vehicle with communication capabilities but no computation."""
    
    id: int
    node_type: NodeType = NodeType.VEHICLE
    position: Position = field(default_factory=lambda: Position(0.0, 0.0, 0.0))
    velocity: Velocity = field(default_factory=lambda: Velocity(0.0, 0.0, 0.0))
    
    # Communication parameters
    transmit_power: float = 1.0  # W
    antenna_gain: float = 1.0  # dB
    
    # Mobility
    mobility_model: Optional[MobilityModel] = None
    
    # Task generation
    task_generation_rate: float = 1.0  # tasks per second
    last_task_time: float = 0.0
    
    # Current region assignment
    current_region_id: Optional[int] = None
    
    # Generated tasks (for tracking)
    generated_tasks: List[int] = field(default_factory=list)
    
    def update_position(self, current_time: float, dt: float) -> None:
        """Update vehicle position using mobility model."""
        if self.mobility_model:
            self.position = self.mobility_model.update_position(
                self.position, current_time, dt
            )
            self.velocity = self.mobility_model.get_velocity(current_time)
    
    def should_generate_task(self, current_time: float) -> bool:
        """Check if vehicle should generate a new task."""
        if self.task_generation_rate <= 0:
            return False
        
        # Poisson process for task generation
        time_since_last = current_time - self.last_task_time
        if time_since_last <= 0:
            return False
        
        # Probability of generating task in this interval
        prob = 1 - np.exp(-self.task_generation_rate * time_since_last)
        return np.random.random() < prob
    
    def generate_task(self, current_time: float, task_id: int, 
                     region_id: int, params: SystemParameters) -> Task:
        """Generate a new computational task."""
        
        # Task characteristics (random generation with realistic parameters)
        data_size_in = np.random.exponential(1.0)  # MB
        data_size_out = np.random.exponential(0.5)  # MB
        cpu_cycles = np.random.exponential(1e9)  # cycles
        deadline = current_time + np.random.exponential(5.0)  # seconds
        
        task = Task(
            id=task_id,
            source_vehicle_id=self.id,
            region_id=region_id,
            data_size_in=data_size_in,
            data_size_out=data_size_out,
            cpu_cycles=cpu_cycles,
            deadline=deadline,
            creation_time=current_time,
            arrival_time=current_time
        )
        
        self.generated_tasks.append(task_id)
        self.last_task_time = current_time
        
        return task
    
    def get_communication_range(self) -> float:
        """Get effective communication range based on transmit power."""
        # Simplified range calculation
        return 100.0 * np.sqrt(self.transmit_power)  # meters


class VehicleManager:
    """Manages all vehicles in the SAGIN system."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.vehicles: Dict[int, Vehicle] = {}
        self.next_vehicle_id = 1
    
    def add_vehicle(self, position: Position, mobility_model: MobilityModel,
                   task_generation_rate: float = 1.0) -> int:
        """Add a new vehicle to the system."""
        vehicle_id = self.next_vehicle_id
        self.next_vehicle_id += 1
        
        vehicle = Vehicle(
            id=vehicle_id,
            position=position,
            mobility_model=mobility_model,
            task_generation_rate=task_generation_rate
        )
        
        self.vehicles[vehicle_id] = vehicle
        return vehicle_id
    
    def update_all_vehicles(self, current_time: float, dt: float) -> None:
        """Update positions of all vehicles."""
        for vehicle in self.vehicles.values():
            vehicle.update_position(current_time, dt)
    
    def assign_vehicles_to_regions(self, regions: Dict[int, Region]) -> None:
        """Assign vehicles to regions based on their current positions."""
        for vehicle in self.vehicles.values():
            # Clear previous assignment
            if vehicle.current_region_id is not None:
                old_region = regions.get(vehicle.current_region_id)
                if old_region and vehicle.id in old_region.vehicle_ids:
                    old_region.vehicle_ids.remove(vehicle.id)
            
            # Find new region
            vehicle.current_region_id = None
            for region in regions.values():
                if region.contains_position(vehicle.position):
                    vehicle.current_region_id = region.id
                    if vehicle.id not in region.vehicle_ids:
                        region.vehicle_ids.append(vehicle.id)
                    break
    
    def get_vehicles_in_region(self, region_id: int) -> List[Vehicle]:
        """Get all vehicles currently in a specific region."""
        return [v for v in self.vehicles.values() if v.current_region_id == region_id]
    
    def get_vehicle_count_by_region(self) -> Dict[int, int]:
        """Get count of vehicles in each region."""
        counts = {}
        for vehicle in self.vehicles.values():
            if vehicle.current_region_id is not None:
                counts[vehicle.current_region_id] = counts.get(vehicle.current_region_id, 0) + 1
        return counts
    
    def create_random_waypoint_vehicles(self, count: int, area_bounds: Tuple[float, float, float, float],
                                      min_speed: float = 5.0, max_speed: float = 15.0) -> List[int]:
        """Create multiple vehicles with random waypoint mobility."""
        vehicle_ids = []
        
        for _ in range(count):
            # Random initial position
            x = np.random.uniform(area_bounds[0], area_bounds[1])
            y = np.random.uniform(area_bounds[2], area_bounds[3])
            position = Position(x, y, 0.0)
            
            # Random waypoint mobility
            mobility = RandomWaypointMobility(area_bounds, min_speed, max_speed)
            
            # Random task generation rate
            task_rate = np.random.exponential(1.0)
            
            vehicle_id = self.add_vehicle(position, mobility, task_rate)
            vehicle_ids.append(vehicle_id)
        
        return vehicle_ids
    
    def create_bus_route_vehicles(self, count: int, route_points: List[Position],
                                 speed: float = 10.0) -> List[int]:
        """Create vehicles following fixed bus routes."""
        vehicle_ids = []
        
        for i in range(count):
            # Distribute vehicles along the route
            start_idx = (i * len(route_points)) // count
            position = route_points[start_idx]
            
            # Fixed route mobility
            mobility = FixedRouteMobility(route_points, speed, loop=True, stop_time=5.0)
            
            # Lower task generation rate for buses
            task_rate = np.random.exponential(0.5)
            
            vehicle_id = self.add_vehicle(position, mobility, task_rate)
            vehicle_ids.append(vehicle_id)
        
        return vehicle_ids
