"""
Main SAGIN network class that integrates all components.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict

from .types import Position, Region, SystemParameters, TaskDecision
from .vehicles import VehicleManager
from .uavs import UAVManager
from .satellites import SatelliteConstellation
from .tasks import TaskManager


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    current_time: float = 0.0
    total_tasks_generated: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    success_rate: float = 0.0
    average_latency: float = 0.0
    energy_consumption: float = 0.0
    load_imbalance: float = 0.0
    uav_utilization: float = 0.0
    satellite_utilization: float = 0.0
    coverage_percentage: float = 0.0


class SAGINNetwork:
    """Main SAGIN network orchestrator."""
    
    def __init__(self, system_params: Optional[SystemParameters] = None):
        """Initialize the SAGIN network."""
        self.system_params = system_params or SystemParameters()
        
        # Core components
        self.vehicle_manager = VehicleManager(self.system_params)
        self.uav_manager = UAVManager(self.system_params)
        self.satellite_constellation = SatelliteConstellation(self.system_params)
        self.task_manager = TaskManager(self.system_params)
        
        # Network topology
        self.regions: Dict[int, Region] = {}
        self.next_region_id = 1
        
        # Simulation state
        self.current_time = 0.0
        self.epoch_count = 0
        self.is_running = False
        
        # Performance tracking
        self.metrics = NetworkMetrics()
        self.metrics_history: List[NetworkMetrics] = []
        
        # Event callbacks
        self.event_callbacks = {
            'task_generated': [],
            'task_completed': [],
            'task_failed': [],
            'uav_repositioned': [],
            'energy_low': []
        }
    
    def create_region(self, name: str, center: Position, radius: float,
                     base_intensity: float = 1.0) -> int:
        """Create a new region in the network."""
        region_id = self.next_region_id
        self.next_region_id += 1
        
        region = Region(
            id=region_id,
            name=name,
            center=center,
            radius=radius,
            base_intensity=base_intensity
        )
        
        self.regions[region_id] = region
        
        # Create static UAV for this region
        static_uav_id = self.uav_manager.create_static_uav(
            region_id, center, cpu_capacity=1e9
        )
        region.static_uav_id = static_uav_id
        
        return region_id
    
    def setup_network_topology(self, area_bounds: Tuple[float, float, float, float],
                             num_regions: int = 5) -> None:
        """Setup network topology with regions."""
        min_x, max_x, min_y, max_y = area_bounds
        
        # Create regions in a grid pattern
        cols = int(np.ceil(np.sqrt(num_regions)))
        rows = int(np.ceil(num_regions / cols))
        
        region_width = (max_x - min_x) / cols
        region_height = (max_y - min_y) / rows
        
        for i in range(num_regions):
            row = i // cols
            col = i % cols
            
            center_x = min_x + (col + 0.5) * region_width
            center_y = min_y + (row + 0.5) * region_height
            radius = min(region_width, region_height) / 2
            
            center = Position(center_x, center_y, 0.0)
            region_id = self.create_region(f"Region_{i+1}", center, radius)
            
            # Set random base intensity
            self.regions[region_id].base_intensity = np.random.uniform(0.5, 2.0)
    
    def add_vehicles(self, count: int, area_bounds: Tuple[float, float, float, float],
                    vehicle_type: str = "random") -> List[int]:
        """Add vehicles to the network."""
        if vehicle_type == "random":
            return self.vehicle_manager.create_random_waypoint_vehicles(
                count, area_bounds
            )
        elif vehicle_type == "bus":
            # Create a simple bus route
            min_x, max_x, min_y, max_y = area_bounds
            route_points = [
                Position(min_x + 0.1 * (max_x - min_x), min_y + 0.1 * (max_y - min_y)),
                Position(max_x - 0.1 * (max_x - min_x), min_y + 0.1 * (max_y - min_y)),
                Position(max_x - 0.1 * (max_x - min_x), max_y - 0.1 * (max_y - min_y)),
                Position(min_x + 0.1 * (max_x - min_x), max_y - 0.1 * (max_y - min_y))
            ]
            return self.vehicle_manager.create_bus_route_vehicles(count, route_points)
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    
    def add_dynamic_uavs(self, count: int, area_bounds: Tuple[float, float, float, float]) -> List[int]:
        """Add dynamic UAVs to the network."""
        uav_ids = []
        min_x, max_x, min_y, max_y = area_bounds
        
        for _ in range(count):
            # Random initial position
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            position = Position(x, y, self.system_params.uav_altitude)
            
            uav_id = self.uav_manager.create_dynamic_uav(position)
            uav_ids.append(uav_id)
        
        return uav_ids
    
    def add_satellite_constellation(self, num_satellites: int = 12, 
                                  num_planes: int = 3) -> List[int]:
        """Add satellite constellation to the network."""
        return self.satellite_constellation.create_constellation(
            num_satellites, num_planes
        )
    
    def initialize_simulation(self) -> None:
        """Initialize the simulation."""
        # Initialize task queues for all regions
        region_ids = list(self.regions.keys())
        self.task_manager.initialize_region_queues(region_ids)
        
        # Reset simulation state
        self.current_time = 0.0
        self.epoch_count = 0
        self.is_running = True
        
        # Initial vehicle-to-region assignment
        self.vehicle_manager.assign_vehicles_to_regions(self.regions)
        
        print(f"SAGIN Network initialized with {len(self.regions)} regions")
        print(f"Vehicles: {len(self.vehicle_manager.vehicles)}")
        print(f"Static UAVs: {len(self.uav_manager.static_uavs)}")
        print(f"Dynamic UAVs: {len(self.uav_manager.dynamic_uavs)}")
        print(f"Satellites: {len(self.satellite_constellation.satellites)}")
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Execute one simulation step."""
        if not self.is_running:
            return {}
        
        if dt is None:
            dt = self.system_params.epoch_duration
        
        step_results = {}
        
        # 1. Update vehicle positions and assignments
        self.vehicle_manager.update_all_vehicles(self.current_time, dt)
        self.vehicle_manager.assign_vehicles_to_regions(self.regions)
        
        # 2. Update UAV positions and processing
        uav_results = self.uav_manager.update_all_uavs(self.current_time, dt)
        step_results['uav_completed'] = uav_results
        
        # 3. Update satellite positions and processing
        sat_results = self.satellite_constellation.update_all_satellites(self.current_time, dt)
        step_results['satellite_completed'] = sat_results
        
        # 4. Generate new tasks
        vehicles_by_region = {}
        for region_id in self.regions.keys():
            vehicles_by_region[region_id] = [
                v.id for v in self.vehicle_manager.get_vehicles_in_region(region_id)
            ]
        
        new_tasks = self.task_manager.generate_tasks(
            self.regions, vehicles_by_region, self.current_time, dt
        )
        step_results['new_tasks'] = new_tasks
        
        # 5. Process task offloading decisions
        self._process_task_offloading()
        
        # 6. Clean up expired tasks
        expired_count = self.task_manager.cleanup_expired_tasks(self.current_time)
        step_results['expired_tasks'] = expired_count
        
        # 7. Update metrics
        self._update_metrics()
        
        # 8. Advance time
        self.current_time += dt
        self.epoch_count += 1
        
        return step_results
    
    def _process_task_offloading(self) -> None:
        """Process task offloading decisions for all regions."""
        for region_id, region in self.regions.items():
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if not static_uav:
                continue
            
            # Get pending tasks for this region
            pending_tasks = self.task_manager.get_tasks_for_region(region_id, max_tasks=10)
            
            for task in pending_tasks:
                # Get available resources
                available_dynamic_uavs = len(
                    self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
                )
                
                # Check satellite availability
                satellite_available = len(
                    self.satellite_constellation.find_visible_satellites(region.center)
                ) > 0
                
                # Make offloading decision
                decision = static_uav.make_offloading_decision(
                    task, available_dynamic_uavs, satellite_available, self.current_time
                )
                
                # Execute decision
                self._execute_task_decision(task, decision, region_id)
    
    def _execute_task_decision(self, task, decision: TaskDecision, region_id: int) -> None:
        """Execute a task offloading decision."""
        task.decision = decision
        
        if decision == TaskDecision.LOCAL:
            # Assign to static UAV
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if static_uav:
                static_uav.add_task(task)
        
        elif decision == TaskDecision.DYNAMIC:
            # Assign to available dynamic UAV
            dynamic_uavs = self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
            if dynamic_uavs:
                # Select UAV with lowest workload
                selected_uav = min(dynamic_uavs, key=lambda u: u.total_workload)
                selected_uav.add_task(task)
            else:
                # Fallback to static UAV
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                if static_uav:
                    static_uav.add_task(task)
        
        elif decision == TaskDecision.SATELLITE:
            # Assign to satellite
            region = self.regions[region_id]
            success = self.satellite_constellation.assign_task_to_satellite(
                task, region.center
            )
            if not success:
                # Fallback to static UAV
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                if static_uav:
                    static_uav.add_task(task)
    
    def _update_metrics(self) -> None:
        """Update network performance metrics."""
        # Get task statistics
        task_metrics = self.task_manager.get_system_metrics()
        
        # Calculate load imbalance
        load_imbalance = self._calculate_load_imbalance()
        
        # Calculate utilization
        uav_utilization = self._calculate_uav_utilization()
        satellite_utilization = self._calculate_satellite_utilization()
        
        # Calculate coverage
        region_positions = [region.center for region in self.regions.values()]
        coverage_percentage = self.satellite_constellation.calculate_coverage_percentage(
            region_positions
        )
        
        # Update metrics
        self.metrics.current_time = self.current_time
        self.metrics.total_tasks_generated = task_metrics['total_generated']
        self.metrics.total_tasks_completed = task_metrics['total_completed']
        self.metrics.total_tasks_failed = task_metrics['total_failed']
        self.metrics.success_rate = task_metrics['success_rate']
        self.metrics.average_latency = task_metrics['average_completion_time']
        self.metrics.load_imbalance = load_imbalance
        self.metrics.uav_utilization = uav_utilization
        self.metrics.satellite_utilization = satellite_utilization
        self.metrics.coverage_percentage = coverage_percentage
        
        # Store in history
        self.metrics_history.append(self.metrics)
    
    def _calculate_load_imbalance(self) -> float:
        """Calculate load imbalance across all processing nodes."""
        workloads = []
        
        # Static UAVs
        for uav in self.uav_manager.static_uavs.values():
            workloads.append(uav.total_workload / uav.cpu_capacity)
        
        # Dynamic UAVs
        for uav in self.uav_manager.dynamic_uavs.values():
            if uav.is_available:
                workloads.append(uav.total_workload / uav.cpu_capacity)
        
        # Satellites
        for satellite in self.satellite_constellation.satellites.values():
            workloads.append(satellite.total_workload / satellite.cpu_capacity)
        
        if not workloads:
            return 0.0
        
        mean_workload = np.mean(workloads)
        return np.std(workloads) / (mean_workload + 1e-6)
    
    def _calculate_uav_utilization(self) -> float:
        """Calculate average UAV utilization."""
        utilizations = []
        
        for uav in self.uav_manager.static_uavs.values():
            utilizations.append(min(1.0, uav.total_workload / uav.cpu_capacity))
        
        for uav in self.uav_manager.dynamic_uavs.values():
            if uav.is_available:
                utilizations.append(min(1.0, uav.total_workload / uav.cpu_capacity))
        
        return np.mean(utilizations) if utilizations else 0.0
    
    def _calculate_satellite_utilization(self) -> float:
        """Calculate average satellite utilization."""
        utilizations = []
        
        for satellite in self.satellite_constellation.satellites.values():
            utilizations.append(min(1.0, satellite.total_workload / satellite.cpu_capacity))
        
        return np.mean(utilizations) if utilizations else 0.0
    
    def run_simulation(self, num_epochs: int, progress_callback: Optional[callable] = None) -> None:
        """Run the simulation for a specified number of epochs."""
        print(f"Starting simulation for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            step_results = self.step()
            
            if progress_callback:
                progress_callback(epoch, num_epochs, step_results)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {len(step_results.get('new_tasks', []))} new tasks, "
                      f"Success rate: {self.metrics.success_rate:.3f}")
        
        print(f"Simulation completed after {num_epochs} epochs")
        print(f"Final metrics: Success rate: {self.metrics.success_rate:.3f}, "
              f"Average latency: {self.metrics.average_latency:.3f}s")
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get complete network state for RL agents."""
        return {
            'current_time': self.current_time,
            'epoch_count': self.epoch_count,
            'regions': {rid: {
                'center': (r.center.x, r.center.y),
                'radius': r.radius,
                'base_intensity': r.base_intensity,
                'current_intensity': r.current_intensity,
                'vehicle_count': len(r.vehicle_ids),
                'static_uav_id': r.static_uav_id,
                'dynamic_uav_count': len(r.dynamic_uav_ids)
            } for rid, r in self.regions.items()},
            'uav_states': self.uav_manager.get_system_state(),
            'satellite_states': self.satellite_constellation.get_constellation_state(),
            'task_states': self.task_manager.get_regional_task_stats(),
            'metrics': self.metrics
        }
    
    def add_event_callback(self, event_type: str, callback: callable) -> None:
        """Add event callback for monitoring."""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_epochs': self.epoch_count,
            'simulation_time': self.current_time,
            'final_metrics': self.metrics,
            'network_elements': {
                'regions': len(self.regions),
                'vehicles': len(self.vehicle_manager.vehicles),
                'static_uavs': len(self.uav_manager.static_uavs),
                'dynamic_uavs': len(self.uav_manager.dynamic_uavs),
                'satellites': len(self.satellite_constellation.satellites)
            }
        }
    
    def reset_simulation(self) -> None:
        """Reset the simulation to initial state."""
        self.current_time = 0.0
        self.epoch_count = 0
        self.is_running = False
        self.metrics = NetworkMetrics()
        self.metrics_history = []
        
        # Reset all components
        self.task_manager = TaskManager(self.system_params)
        
        # Re-initialize
        self.initialize_simulation()
    
    def export_results(self, filename: str) -> None:
        """Export simulation results to file."""
        results = {
            'system_params': self.system_params.__dict__,
            'performance_summary': self.get_performance_summary(),
            'metrics_history': [m.__dict__ for m in self.metrics_history],
            'network_state': self.get_network_state()
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
