"""
Satellite models for the SAGIN system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from .types import (
    Position, Velocity, NodeType, SystemParameters,
    Task, TaskStatus
)


class OrbitType(Enum):
    """Types of satellite orbits."""
    LEO = "leo"  # Low Earth Orbit
    MEO = "meo"  # Medium Earth Orbit
    GEO = "geo"  # Geostationary Earth Orbit


@dataclass
class OrbitalParameters:
    """Orbital parameters for satellite motion."""
    semi_major_axis: float  # km
    eccentricity: float = 0.0
    inclination: float = 0.0  # degrees
    longitude_of_ascending_node: float = 0.0  # degrees
    argument_of_periapsis: float = 0.0  # degrees
    mean_anomaly: float = 0.0  # degrees
    
    @property
    def orbital_period(self) -> float:
        """Calculate orbital period in seconds."""
        # Using Kepler's third law
        # T = 2π * sqrt(a³ / GM)
        # where GM for Earth ≈ 3.986004418 × 10^14 m³/s²
        GM = 3.986004418e14  # m³/s²
        a = self.semi_major_axis * 1000  # convert km to m
        return 2 * math.pi * math.sqrt(a**3 / GM)


@dataclass
class Satellite:
    """Satellite node in the SAGIN system."""
    
    id: int
    node_type: NodeType = NodeType.SATELLITE
    orbit_type: OrbitType = OrbitType.LEO
    orbital_params: OrbitalParameters = field(default_factory=lambda: OrbitalParameters(550.0))
    
    # Current state
    position: Position = field(default_factory=lambda: Position(0.0, 0.0, 550000.0))
    velocity: Velocity = field(default_factory=lambda: Velocity(0.0, 0.0, 0.0))
    
    # Computing capabilities
    cpu_capacity: float = 5e9  # cycles per second (higher than UAVs)
    current_workload: float = 0.0
    
    # Communication
    transmit_power: float = 100.0  # W (higher than UAVs)
    antenna_gain: float = 20.0  # dB (higher gain)
    downlink_bandwidth: float = 100.0  # MHz
    uplink_bandwidth: float = 50.0  # MHz
    
    # Coverage
    coverage_radius: float = 2000000.0  # m (2000 km radius - more realistic for LEO satellites)
    min_elevation_angle: float = 5.0  # degrees (reduced for better coverage)
    
    # Task management
    task_queue: List[Task] = field(default_factory=list)
    processing_tasks: List[Task] = field(default_factory=list)
    completed_tasks: List[int] = field(default_factory=list)
    
    # Operational state
    is_active: bool = True
    last_update_time: float = 0.0
    
    def __post_init__(self):
        """Initialize satellite position based on orbital parameters."""
        if self.orbit_type == OrbitType.LEO:
            # LEO satellites typically at 550-2000 km altitude
            self.position.z = self.orbital_params.semi_major_axis * 1000
        elif self.orbit_type == OrbitType.MEO:
            # MEO satellites at 2000-35,786 km altitude
            self.position.z = self.orbital_params.semi_major_axis * 1000
        elif self.orbit_type == OrbitType.GEO:
            # GEO satellites at ~35,786 km altitude
            self.position.z = 35786000.0
    
    @property
    def altitude(self) -> float:
        """Get satellite altitude in meters."""
        return self.position.z
    
    @property
    def queue_length(self) -> int:
        """Get current task queue length."""
        return len(self.task_queue)
    
    @property
    def total_workload(self) -> float:
        """Get total workload (queued + processing)."""
        queue_workload = sum(task.cpu_cycles for task in self.task_queue)
        processing_workload = sum(task.cpu_cycles for task in self.processing_tasks)
        return queue_workload + processing_workload
    
    def update_orbital_position(self, current_time: float, dt: float) -> None:
        """Update satellite position based on orbital mechanics."""
        if self.orbit_type == OrbitType.GEO:
            # GEO satellites are stationary relative to Earth
            return
        
        # Earth radius in meters
        EARTH_RADIUS = 6371000.0  # 6,371 km
        
        # Simplified circular orbit model
        period = self.orbital_params.orbital_period
        angular_velocity = 2 * math.pi / period
        
        # Calculate current orbital angle
        time_since_epoch = current_time - self.last_update_time
        angle_change = angular_velocity * time_since_epoch
        
        # Update position (simplified circular orbit)
        # Orbital radius from Earth's center
        orbital_radius = EARTH_RADIUS + (self.orbital_params.semi_major_axis * 1000)
        current_angle = (current_time * angular_velocity) % (2 * math.pi)
        
        # Position in orbital plane (Earth-centered coordinates)
        orbit_x = orbital_radius * math.cos(current_angle)
        orbit_y = orbital_radius * math.sin(current_angle)
        orbit_z = 0.0  # Simplified - assumes orbit above equator
        
        # Convert to ground-relative coordinates for SAGIN simulation
        # We'll use the altitude above ground as z-coordinate
        altitude = self.orbital_params.semi_major_axis * 1000  # Height above Earth surface
        
        # Project orbit position to simulation area (simplified)
        # Map orbital position to simulation coordinates
        sim_area_scale = 50000.0  # Scale factor to map orbit to simulation area
        x = (orbit_x / orbital_radius) * sim_area_scale
        y = (orbit_y / orbital_radius) * sim_area_scale
        z = altitude  # Height above ground level
        
        self.position = Position(x, y, z)
        
        # Update velocity
        v_magnitude = 2 * math.pi * orbital_radius / period
        self.velocity = Velocity(
            -v_magnitude * math.sin(current_angle),
            v_magnitude * math.cos(current_angle),
            0.0
        )
        
        self.last_update_time = current_time
    
    def is_visible_from_position(self, ground_position: Position) -> bool:
        """Check if satellite is visible from a ground position."""
        # Calculate vector from ground to satellite
        ground_to_sat = self.position - ground_position
        
        # Calculate 3D distance
        distance_3d = math.sqrt(ground_to_sat.x**2 + ground_to_sat.y**2 + ground_to_sat.z**2)
        
        # Check if within coverage radius
        if distance_3d > self.coverage_radius:
            return False
        
        # Calculate elevation angle
        horizontal_distance = math.sqrt(ground_to_sat.x**2 + ground_to_sat.y**2)
        
        # Avoid division by zero
        if horizontal_distance == 0:
            elevation_angle = 90.0  # Satellite directly overhead
        else:
            elevation_angle = math.degrees(math.atan2(ground_to_sat.z, horizontal_distance))
        
        # Check minimum elevation angle and ensure satellite is above ground
        return elevation_angle >= self.min_elevation_angle and ground_to_sat.z > 0
    
    def calculate_link_distance(self, ground_position: Position) -> float:
        """Calculate link distance to ground position."""
        return self.position.distance_to(ground_position)
    
    def calculate_propagation_delay(self, ground_position: Position) -> float:
        """Calculate propagation delay to ground position."""
        distance = self.calculate_link_distance(ground_position)
        return distance / 3e8  # speed of light
    
    def add_task(self, task: Task) -> bool:
        """Add a task to the satellite's processing queue."""
        if not self.is_active:
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
            processing_time = current_time - task.start_time
            if processing_time > 0:
                remaining_capacity -= task.cpu_cycles / processing_time
        
        # Move tasks from queue to processing
        while self.task_queue and remaining_capacity > 0:
            task = self.task_queue.pop(0)
            if task.deadline > current_time:  # Check deadline
                task.status = TaskStatus.IN_PROGRESS
                task.start_time = current_time
                self.processing_tasks.append(task)
                # Estimate processing time and reserve capacity
                estimated_time = task.cpu_cycles / self.cpu_capacity
                remaining_capacity -= task.cpu_cycles / estimated_time
            else:
                task.status = TaskStatus.DEADLINE_MISSED
                completed.append(task)
        
        # Process current tasks
        still_processing = []
        for task in self.processing_tasks:
            processing_time = current_time - task.start_time
            required_time = task.cpu_cycles / self.cpu_capacity
            
            if processing_time >= required_time:
                # Task completed
                task.status = TaskStatus.COMPLETED
                task.completion_time = current_time
                completed.append(task)
                self.completed_tasks.append(task.id)
            else:
                still_processing.append(task)
        
        self.processing_tasks = still_processing
        return completed
    
    def get_state(self) -> Dict:
        """Get satellite state for monitoring and RL."""
        return {
            'id': self.id,
            'position': (self.position.x, self.position.y, self.position.z),
            'velocity': (self.velocity.vx, self.velocity.vy, self.velocity.vz),
            'altitude': self.altitude,
            'queue_length': self.queue_length,
            'total_workload': self.total_workload,
            'cpu_utilization': min(1.0, self.total_workload / self.cpu_capacity),
            'is_active': self.is_active,
            'orbit_type': self.orbit_type.value
        }


class SatelliteConstellation:
    """Manages a constellation of satellites."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.satellites: Dict[int, Satellite] = {}
        self.next_satellite_id = 1
        
        # Coverage tracking
        self.coverage_map: Dict[Tuple[int, int], List[int]] = {}  # (region_id, time_slot) -> [satellite_ids]
    
    def create_leo_satellite(self, altitude: float = 550000.0,
                           inclination: float = 53.0) -> int:
        """Create a LEO satellite."""
        sat_id = self.next_satellite_id
        self.next_satellite_id += 1
        
        orbital_params = OrbitalParameters(
            semi_major_axis=altitude / 1000,  # convert m to km
            inclination=inclination,
            eccentricity=0.0001  # nearly circular
        )
        
        satellite = Satellite(
            id=sat_id,
            orbit_type=OrbitType.LEO,
            orbital_params=orbital_params,
            cpu_capacity=5e9,  # 5 GHz processing
            transmit_power=100.0,
            antenna_gain=20.0
        )
        
        self.satellites[sat_id] = satellite
        return sat_id
    
    def create_constellation(self, num_satellites: int, num_planes: int,
                           altitude: float = 550000.0) -> List[int]:
        """Create a constellation of LEO satellites."""
        satellite_ids = []
        
        # Distribute satellites across orbital planes
        sats_per_plane = num_satellites // num_planes
        
        for plane in range(num_planes):
            plane_raan = (360.0 / num_planes) * plane  # Right Ascension of Ascending Node
            
            for sat_in_plane in range(sats_per_plane):
                sat_id = self.create_leo_satellite(altitude)
                satellite = self.satellites[sat_id]
                
                # Set orbital parameters for this satellite
                satellite.orbital_params.longitude_of_ascending_node = plane_raan
                satellite.orbital_params.mean_anomaly = (360.0 / sats_per_plane) * sat_in_plane
                
                satellite_ids.append(sat_id)
        
        return satellite_ids
    
    def update_all_satellites(self, current_time: float, dt: float) -> Dict[str, List[Task]]:
        """Update all satellites and return completed tasks."""
        completed_tasks = []
        
        for satellite in self.satellites.values():
            # Update orbital position
            satellite.update_orbital_position(current_time, dt)
            
            # Process tasks
            completed = satellite.process_tasks(current_time, dt)
            completed_tasks.extend(completed)
        
        return {'satellite_completed': completed_tasks}
    
    def find_visible_satellites(self, ground_position: Position) -> List[Satellite]:
        """Find satellites visible from a ground position."""
        visible = []
        
        for satellite in self.satellites.values():
            if satellite.is_active and satellite.is_visible_from_position(ground_position):
                visible.append(satellite)
        
        return visible
    
    def get_best_satellite(self, ground_position: Position) -> Optional[Satellite]:
        """Get the best satellite for communication from a ground position."""
        visible_satellites = self.find_visible_satellites(ground_position)
        
        if not visible_satellites:
            return None
        
        # Select satellite with lowest workload and good link quality
        best_satellite = None
        best_score = float('inf')
        
        for satellite in visible_satellites:
            # Calculate score based on workload and distance
            workload_factor = satellite.total_workload / satellite.cpu_capacity
            distance_factor = satellite.calculate_link_distance(ground_position) / 1000000  # normalize
            
            score = workload_factor + 0.1 * distance_factor
            
            if score < best_score:
                best_score = score
                best_satellite = satellite
        
        return best_satellite
    
    def assign_task_to_satellite(self, task: Task, ground_position: Position) -> bool:
        """Assign a task to the best available satellite."""
        satellite = self.get_best_satellite(ground_position)
        
        if satellite is None:
            return False
        
        return satellite.add_task(task)
    
    def get_constellation_state(self) -> Dict:
        """Get state of the entire constellation."""
        states = {}
        
        for sat_id, satellite in self.satellites.items():
            states[sat_id] = satellite.get_state()
        
        return {
            'satellites': states,
            'total_satellites': len(self.satellites),
            'active_satellites': sum(1 for sat in self.satellites.values() if sat.is_active),
            'total_workload': sum(sat.total_workload for sat in self.satellites.values()),
            'average_queue_length': np.mean([sat.queue_length for sat in self.satellites.values()]) if self.satellites else 0
        }
    
    def calculate_coverage_percentage(self, regions: List[Position]) -> float:
        """Calculate percentage of regions covered by at least one satellite."""
        if not regions:
            return 0.0
        
        covered_regions = 0
        
        for region_pos in regions:
            visible_sats = self.find_visible_satellites(region_pos)
            if visible_sats:
                covered_regions += 1
        
        return covered_regions / len(regions) * 100.0
    
    def get_satellite_by_id(self, satellite_id: int) -> Optional[Satellite]:
        """Get satellite by ID."""
        return self.satellites.get(satellite_id)
    
    def get_system_metrics(self) -> Dict:
        """Get system-wide metrics for monitoring."""
        if not self.satellites:
            return {}
        
        total_tasks = sum(len(sat.completed_tasks) for sat in self.satellites.values())
        total_queue = sum(sat.queue_length for sat in self.satellites.values())
        total_processing = sum(len(sat.processing_tasks) for sat in self.satellites.values())
        
        return {
            'total_completed_tasks': total_tasks,
            'total_queued_tasks': total_queue,
            'total_processing_tasks': total_processing,
            'average_cpu_utilization': np.mean([
                min(1.0, sat.total_workload / sat.cpu_capacity) 
                for sat in self.satellites.values()
            ]),
            'constellation_health': sum(1 for sat in self.satellites.values() if sat.is_active) / len(self.satellites)
        }
    
    def get_communication_delay(self, ground_position: Position) -> float:
        """Calculate communication delay to best available satellite."""
        best_satellite = self.get_best_satellite(ground_position)
        
        if best_satellite is None:
            return float('inf')  # No satellite available
        
        # Calculate round-trip time
        distance = best_satellite.calculate_link_distance(ground_position)
        
        # Speed of light in vacuum (approximately)
        speed_of_light = 3e8  # m/s
        
        # Propagation delay (one way)
        propagation_delay = distance / speed_of_light
        
        # Add processing delays (simplified)
        transmission_delay = 0.001  # 1ms for transmission processing
        processing_delay = 0.002    # 2ms for satellite processing
        
        # Total delay (round trip)
        total_delay = 2 * propagation_delay + transmission_delay + processing_delay
        
        return total_delay
    
    def get_link_quality_to_region(self, ground_position: Position) -> Dict[str, float]:
        """Get link quality metrics to best satellite from ground position."""
        best_satellite = self.get_best_satellite(ground_position)
        
        if best_satellite is None:
            return {
                'distance': float('inf'),
                'elevation_angle': 0.0,
                'communication_delay': float('inf'),
                'link_available': False
            }
        
        # Calculate metrics
        distance = best_satellite.calculate_link_distance(ground_position)
        
        # Calculate elevation angle
        ground_to_sat = best_satellite.position - ground_position
        horizontal_distance = math.sqrt(ground_to_sat.x**2 + ground_to_sat.y**2)
        elevation_angle = math.degrees(math.atan2(ground_to_sat.z, horizontal_distance)) if horizontal_distance > 0 else 90.0
        
        return {
            'distance': distance,
            'elevation_angle': elevation_angle,
            'communication_delay': self.get_communication_delay(ground_position),
            'satellite_id': best_satellite.id,
            'workload_factor': best_satellite.total_workload / best_satellite.cpu_capacity,
            'link_available': True
        }
