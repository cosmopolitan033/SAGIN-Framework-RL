"""
Main SAGIN network class that integrates all components.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import os
import sys
import json
from collections import defaultdict

from .types import Position, Region, SystemParameters, TaskDecision
from .vehicles import VehicleManager
from .uavs import UAVManager, UAVStatus
from .satellites import SatelliteConstellation
from .tasks import TaskManager
# Import basic models only to avoid circular dependencies
# Communication models will be imported when needed


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
        self.grid = None  # Will be set when topology is configured
        
        # Communication models - loaded dynamically to avoid import issues
        self.communication_model = None
        self.latency_model = None
        self.load_balancing_metrics = None
        self.shannon_capacity_model = None
        
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
        
        # Decision history for analysis
        self.decision_history: List[Dict[str, Any]] = []
        self.resource_utilization_history: List[Dict[str, Any]] = []
        
        # Logging configuration
        self.log_decisions = True
        self.log_resource_usage = True
        
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
        static_uav_position = Position(center.x, center.y, self.system_params.static_uav_altitude)
        static_uav_id = self.uav_manager.create_static_uav(
            region_id, static_uav_position, cpu_capacity=1e9
        )
        region.static_uav_id = static_uav_id
        
        return region_id
    
    def setup_network_topology(self, grid_config=None, area_bounds=None, num_regions=None, task_config=None) -> None:
        """Setup network topology with regions using grid configuration."""
        # Handle backward compatibility
        if grid_config is None:
            # Import here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from config.grid_config import GridConfig
            
            if area_bounds is None:
                area_bounds = (0.0, 10000.0, 0.0, 10000.0)
            if num_regions is None:
                num_regions = 5
            
            # Create square grid for backward compatibility
            cols = int(np.ceil(np.sqrt(num_regions)))
            rows = int(np.ceil(num_regions / cols))
            grid_config = GridConfig(
                grid_rows=rows,
                grid_cols=cols,
                area_bounds=area_bounds
            )
        
        # Store grid configuration for visualization and other uses
        self.grid = grid_config
        
        # Create regions in grid pattern
        for row in range(grid_config.grid_rows):
            for col in range(grid_config.grid_cols):
                region_id = grid_config.get_region_id(row, col)
                center_x, center_y = grid_config.get_region_center(row, col)
                
                # Calculate radius based on region size
                radius = min(grid_config.region_width, grid_config.region_height) / 2
                
                # Create region
                center = Position(center_x, center_y, 0.0)
                created_region_id = self.create_region(f"Region_{row+1}_{col+1}", center, radius)
                
                # Set base intensity with some variance
                # Use task_config.base_task_rate if available, otherwise fall back to grid_config.region_base_intensity
                if task_config is not None:
                    base_task_rate = task_config.base_task_rate
                else:
                    base_task_rate = grid_config.region_base_intensity
                
                base_intensity = base_task_rate + \
                               np.random.uniform(-grid_config.region_intensity_variance, 
                                               grid_config.region_intensity_variance)
                self.regions[created_region_id].base_intensity = max(0.1, base_intensity)
        
        print(f"Created {grid_config.grid_rows}x{grid_config.grid_cols} grid topology:")
        print(f"  - Total regions: {grid_config.total_regions}")
        print(f"  - Area: {grid_config.area_bounds[1]/1000:.1f}km x {grid_config.area_bounds[3]/1000:.1f}km")
        print(f"  - Region size: {grid_config.region_width/1000:.1f}km x {grid_config.region_height/1000:.1f}km")
    
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
            position = Position(x, y, self.system_params.dynamic_uav_altitude)
            
            uav_id = self.uav_manager.create_dynamic_uav(position)
            uav_ids.append(uav_id)
        
        return uav_ids
    
    def add_satellite_constellation(self, num_satellites: int = 12) -> List[int]:
        """Add satellite constellation to the network."""
        return self.satellite_constellation.create_constellation(
            num_satellites, num_planes=1  # Single orbital plane for simplified model
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
    
    def step(self, dt: Optional[float] = None, verbose: bool = True) -> Dict[str, Any]:
        """Execute one simulation step."""
        if not self.is_running:
            return {}
        
        if dt is None:
            dt = self.system_params.epoch_duration
        
        if verbose:
            print(f"\n⏱️  Epoch {self.epoch_count} (t={self.current_time:.1f}s)")
            print("="*50)
        
        step_results = {}
        
        # 1. Update vehicle positions and assignments
        if verbose:
            print(f"  🚗 Updating vehicle positions...")
        self.vehicle_manager.update_all_vehicles(self.current_time, dt)
        self.vehicle_manager.assign_vehicles_to_regions(self.regions)
        
        # Log vehicle distribution with detailed stats
        if verbose:
            vehicle_counts = self.vehicle_manager.get_vehicle_count_by_region()
            total_vehicles = sum(vehicle_counts.values())
            print(f"    Total vehicles: {total_vehicles}")
            for region_id, count in vehicle_counts.items():
                if count > 0:
                    percentage = (count / total_vehicles) * 100 if total_vehicles > 0 else 0
                    print(f"    Region {region_id}: {count} vehicles ({percentage:.1f}%)")
                    
                    # Show vehicle details for all vehicles
                    vehicles_in_region = self.vehicle_manager.get_vehicles_in_region(region_id)
                    for vehicle in vehicles_in_region:
                        speed = np.sqrt(vehicle.velocity.vx**2 + vehicle.velocity.vy**2 + vehicle.velocity.vz**2)
                        print(f"      Vehicle {vehicle.id}: pos=({vehicle.position.x:.1f}, {vehicle.position.y:.1f}), "
                              f"speed={speed:.1f}m/s")
        
        # 2. Update UAV positions and processing
        if verbose:
            print(f"  🚁 Updating UAVs...")
        uav_results = self.uav_manager.update_all_uavs(self.current_time, dt)
        step_results['uav_completed'] = uav_results
        
        # Process completed tasks from UAVs - CRITICAL FIX
        static_completed = uav_results.get('static_completed', [])
        dynamic_completed = uav_results.get('dynamic_completed', [])
        
        # Mark all UAV completed tasks in the task manager
        for task in static_completed + dynamic_completed:
            self.task_manager.mark_task_completed(task)
            if verbose:
                print(f"    ✅ Task {task.id} completed by UAV and registered")
        
        # Log detailed UAV status
        if verbose:
            static_completed_count = len(static_completed)
            dynamic_completed_count = len(dynamic_completed)
            print(f"    Tasks completed: Static={static_completed_count}, Dynamic={dynamic_completed_count}")
            
            # Log static UAV details
            for uav_id, uav in self.uav_manager.static_uavs.items():
                workload = uav.total_workload
                queue_len = uav.queue_length
                energy_pct = (uav.current_energy / uav.battery_capacity) * 100
                print(f"    Static UAV {uav_id} (Region {uav.assigned_region_id}): "
                      f"workload={workload:.0f}, queue={queue_len}, energy={energy_pct:.1f}%")
            
            # Log dynamic UAV repositioning and status
            for uav_id, uav in self.uav_manager.dynamic_uavs.items():
                workload = uav.total_workload
                energy_pct = (uav.current_energy / uav.battery_capacity) * 100
                current_region = getattr(uav, 'assigned_region_id', 'N/A')
                target_region = getattr(uav, 'target_region_id', 'N/A')
                
                if uav.status.value == "flying":
                    print(f"    Dynamic UAV {uav_id}: flying to region {target_region}, "
                          f"workload={workload:.0f}, energy={energy_pct:.1f}%")
                else:
                    print(f"    Dynamic UAV {uav_id}: {uav.status.value} in region {current_region}, "
                          f"workload={workload:.0f}, energy={energy_pct:.1f}%")

        # 2.1. Dynamic UAV repositioning (every 10 epochs)
        if self.epoch_count % 10 == 0:
            if verbose:
                print(f"  🔄 Checking dynamic UAV repositioning...")
            self._reposition_dynamic_uavs(verbose)
        
        # 3. Update satellite positions and processing
        if verbose:
            print(f"  🛰️  Updating satellites...")
        sat_results = self.satellite_constellation.update_all_satellites(self.current_time, dt)
        step_results['satellite_completed'] = sat_results
        
        # Process completed tasks from satellites - CRITICAL FIX
        satellite_completed = sat_results.get('satellite_completed', [])
        
        # Mark all satellite completed tasks in the task manager
        for task in satellite_completed:
            self.task_manager.mark_task_completed(task)
            if verbose:
                print(f"    ✅ Task {task.id} completed by satellite and registered")
        
        # Log detailed satellite status
        if verbose:
            sat_completed_count = len(satellite_completed)
            print(f"    Tasks completed by satellites: {sat_completed_count}")
            
            # Log satellite details
            total_sat_workload = 0
            visible_sats_per_region = {}
            for region_id, region in self.regions.items():
                # Check satellite visibility from static UAV position instead of ground
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                if static_uav:
                    visible_sats = self.satellite_constellation.find_visible_satellites(static_uav.position)
                else:
                    visible_sats = self.satellite_constellation.find_visible_satellites(region.center)
                visible_sats_per_region[region_id] = len(visible_sats)
            
            for sat_id, satellite in self.satellite_constellation.satellites.items():
                workload = satellite.total_workload
                queue_len = satellite.queue_length
                total_sat_workload += workload
                print(f"    Satellite {sat_id}: workload={workload:.0f}, queue={queue_len}")
            
            if visible_sats_per_region:
                print(f"    Satellite visibility: " + 
                      ", ".join([f"R{r}:{v}" for r, v in visible_sats_per_region.items()]))
        
        # 4. Generate new tasks
        if verbose:
            print(f"  📝 Generating new tasks...")
        vehicles_by_region = {}
        for region_id in self.regions.keys():
            vehicles_by_region[region_id] = [
                v.id for v in self.vehicle_manager.get_vehicles_in_region(region_id)
            ]
        
        new_tasks = self.task_manager.generate_tasks(
            self.regions, vehicles_by_region, self.current_time, dt
        )
        step_results['new_tasks'] = new_tasks
        
        # Log detailed task generation
        if verbose:
            if new_tasks:
                tasks_by_region = {}
                tasks_by_size = {'small': 0, 'medium': 0, 'large': 0}
                for task in new_tasks:
                    tasks_by_region[task.region_id] = tasks_by_region.get(task.region_id, 0) + 1
                    # Categorize by data size
                    if task.data_size_in < 10:
                        tasks_by_size['small'] += 1
                    elif task.data_size_in < 50:
                        tasks_by_size['medium'] += 1
                    else:
                        tasks_by_size['large'] += 1
                
                print(f"    Total new tasks: {len(new_tasks)}")
                for region_id, count in tasks_by_region.items():
                    print(f"      Region {region_id}: {count} tasks")
                
                for size_type, count in tasks_by_size.items():
                    print(f"      Size {size_type}: {count} tasks")
                    
                # Show details for first few tasks
                for i, task in enumerate(new_tasks[:3]):
                    print(f"      Task {task.id}: size={task.data_size_in}MB, "
                          f"cycles={task.cpu_cycles:.0f}, "
                          f"deadline={task.deadline - self.current_time:.1f}s")
            else:
                print(f"    No new tasks generated")
        
        # 5. Process task offloading decisions
        if verbose:
            print(f"  📋 Processing task offloading decisions...")
        decisions_made = self._process_task_offloading(verbose)
        step_results['decisions_made'] = decisions_made
        
        if verbose:
            total_decisions = sum(decisions_made.values())
            if total_decisions > 0:
                print(f"    Total decisions: {total_decisions}")
                print(f"      Local: {decisions_made['local']} ({decisions_made['local']/total_decisions*100:.1f}%)")
                print(f"      Dynamic: {decisions_made['dynamic']} ({decisions_made['dynamic']/total_decisions*100:.1f}%)")
                print(f"      Satellite: {decisions_made['satellite']} ({decisions_made['satellite']/total_decisions*100:.1f}%)")
                print(f"      Failed: {decisions_made['failed']} ({decisions_made['failed']/total_decisions*100:.1f}%)")
            else:
                print(f"    No offloading decisions made")
        
        # 6. Clean up expired tasks
        expired_count = self.task_manager.cleanup_expired_tasks(self.current_time)
        step_results['expired_tasks'] = expired_count
        
        if verbose and expired_count > 0:
            print(f"  🗑️  Cleaned up {expired_count} expired tasks")
        
        # 7. Update metrics
        self._update_metrics()
        
        # 8. Advance time
        self.current_time += dt
        self.epoch_count += 1
        
        # Log extended summary metrics
        if verbose:
            # Calculate additional metrics
            total_tasks = self.metrics.total_tasks_generated
            active_tasks = sum(len(self.task_manager.get_tasks_for_region(r, 100)) for r in self.regions.keys())
            
            print(f"  📊 Epoch Summary:")
            print(f"    Success rate: {self.metrics.success_rate:.3f} ({self.metrics.total_tasks_completed}/{total_tasks})")
            print(f"    Average latency: {self.metrics.average_latency:.3f}s")
            print(f"    Active tasks: {active_tasks}")
            print(f"    UAV utilization: {self.metrics.uav_utilization:.3f}")
            print(f"    Satellite utilization: {self.metrics.satellite_utilization:.3f}")
            print(f"    Energy consumption: {self.metrics.energy_consumption:.1f}J")
            print(f"    Load imbalance: {self.metrics.load_imbalance:.3f}")
            print(f"    Coverage: {self.metrics.coverage_percentage:.1f}%")
        
        return step_results
    
    def _process_task_offloading(self, verbose: bool = True) -> Dict[str, int]:
        """Process task offloading decisions for all regions."""
        decisions_made = {'local': 0, 'dynamic': 0, 'satellite': 0, 'failed': 0}
        
        for region_id, region in self.regions.items():
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if not static_uav:
                continue
            
            # Get pending tasks for this region
            pending_tasks = self.task_manager.get_tasks_for_region(region_id, max_tasks=10)
            
            if pending_tasks and verbose:
                print(f"    Region {region_id}: Processing {len(pending_tasks)} pending tasks")
            
            for task in pending_tasks:
                # Get available resources
                available_dynamic_uavs = self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
                dynamic_uav_count = len(available_dynamic_uavs)
                
                # Check satellite availability from static UAV position (not ground level)
                visible_satellites = self.satellite_constellation.find_visible_satellites(static_uav.position)
                satellite_available = len(visible_satellites) > 0
                
                # Calculate resource metrics for decision
                static_uav_load = static_uav.total_workload
                min_dynamic_load = min([uav.total_workload for uav in available_dynamic_uavs]) if available_dynamic_uavs else float('inf')
                
                # Make offloading decision
                decision = static_uav.make_offloading_decision(
                    task, dynamic_uav_count, satellite_available, self.current_time
                )
                
                # Log detailed decision rationale
                if verbose:
                    urgency = "URGENT" if (task.deadline - self.current_time) < 5.0 else "normal"
                    print(f"      Task {task.id} (size={task.data_size_in}MB, {urgency}): Decision={decision.value}")
                    print(f"        Resources: Static load={static_uav_load:.0f}, Dynamic UAVs={dynamic_uav_count}, "
                          f"Satellites={len(visible_satellites)}")
                    print(f"        Constraints: Deadline={task.deadline - self.current_time:.1f}s, "
                          f"Cycles={task.cpu_cycles:.0f}, Size={task.data_size_in}MB")
                
                # Track decision for analysis
                resources_available = {
                    'dynamic_uavs': dynamic_uav_count,
                    'satellites': len(visible_satellites),
                    'static_uav_load': static_uav_load,
                    'min_dynamic_load': min_dynamic_load
                }
                
                # Execute decision
                success = self._execute_task_decision(task, decision, region_id, verbose)
                
                # Track the decision
                self.track_decision(task.id, region_id, decision.value, resources_available, success)
                
                if success:
                    decisions_made[decision.value] += 1
                else:
                    decisions_made['failed'] += 1
                    if verbose:
                        print(f"        ❌ Failed to execute decision {decision.value} for task {task.id}")
        
        return decisions_made
    
    def _execute_task_decision(self, task, decision: TaskDecision, region_id: int, verbose: bool = True) -> bool:
        """Execute a task offloading decision."""
        task.decision = decision
        
        if decision == TaskDecision.LOCAL:
            # Assign to static UAV
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if static_uav:
                success = static_uav.add_task(task)
                if success and verbose:
                    print(f"        ✅ Task {task.id} assigned to Static UAV {static_uav.id}")
                    print(f"           Queue: {static_uav.queue_length}, Total workload: {static_uav.total_workload:.0f}")
                return success
        
        elif decision == TaskDecision.DYNAMIC:
            # Assign to available dynamic UAV with hierarchical check
            dynamic_uavs = self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
            if dynamic_uavs:
                # Select UAV with lowest workload
                selected_uav = min(dynamic_uavs, key=lambda u: u.total_workload)
                
                # Check if selected dynamic UAV is also overloaded
                # If so, escalate to satellite
                is_dynamic_overloaded = selected_uav.total_workload >= selected_uav.cpu_capacity * 1.5
                
                if is_dynamic_overloaded:
                    # Try satellite first as per hierarchy
                    region = self.regions[region_id]
                    static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                    communication_position = static_uav.position if static_uav else region.center
                    
                    satellite_success = self.satellite_constellation.assign_task_to_satellite(
                        task, communication_position
                    )
                    if satellite_success and verbose:
                        print(f"        🛰️  Task {task.id} escalated to satellite (dynamic UAV {selected_uav.id} overloaded)")
                        print(f"           Dynamic UAV workload: {selected_uav.total_workload:.0f}/{selected_uav.cpu_capacity:.0f}")
                        return True
                    # If satellite fails, still try the dynamic UAV
                
                # Assign to dynamic UAV (either not overloaded or satellite failed)
                success = selected_uav.add_task(task)
                if success and verbose:
                    status = "overloaded but processing" if is_dynamic_overloaded else "available"
                    print(f"        ✅ Task {task.id} assigned to Dynamic UAV {selected_uav.id} ({status})")
                    print(f"           Selected from {len(dynamic_uavs)} available UAVs")
                    print(f"           Workload: {selected_uav.total_workload:.0f}, Energy: {selected_uav.current_energy:.0f}J")
                return success
            else:
                # Fallback to satellite, then static UAV
                region = self.regions[region_id]
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                communication_position = static_uav.position if static_uav else region.center
                
                satellite_success = self.satellite_constellation.assign_task_to_satellite(
                    task, communication_position
                )
                if satellite_success and verbose:
                    print(f"        🛰️  Task {task.id} fallback to satellite (no dynamic UAVs available)")
                    return True
                
                # Final fallback to static UAV
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                if static_uav:
                    success = static_uav.add_task(task)
                    if success and verbose:
                        print(f"        🔄 Task {task.id} fallback to Static UAV {static_uav.id} (no dynamic UAVs or satellites available)")
                    return success
        
        elif decision == TaskDecision.SATELLITE:
            # Assign to satellite from static UAV position
            region = self.regions[region_id]
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            communication_position = static_uav.position if static_uav else region.center
            
            success = self.satellite_constellation.assign_task_to_satellite(
                task, communication_position
            )
            if success and verbose:
                print(f"        🛰️  Task {task.id} assigned to satellite")
                print(f"           Communication delay: {self.satellite_constellation.get_communication_delay(communication_position):.3f}s")
                return success
            else:
                # Fallback to static UAV
                static_uav = self.uav_manager.get_static_uav_by_region(region_id)
                if static_uav:
                    success = static_uav.add_task(task)
                    if success and verbose:
                        print(f"        🔄 Task {task.id} fallback to Static UAV {static_uav.id} (satellite unavailable)")
                    return success
        
        return False
    
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
        
        # Initialize communication models if needed
        self._initialize_communication_models()
        
        # Use load balancing metrics if available
        if self.load_balancing_metrics:
            comprehensive_metrics = self.load_balancing_metrics.calculate_comprehensive_load_metrics(workloads)
            
            # Store detailed metrics for analysis
            self.load_balancing_metrics.update_history(workloads, workloads)
            
            # Return the coefficient of variation as the main imbalance metric
            return comprehensive_metrics['coefficient_of_variation']
        else:
            # Fallback to basic coefficient of variation
            mean_workload = np.mean(workloads)
            if mean_workload == 0:
                return 0.0
            return np.std(workloads) / mean_workload
    
    def get_detailed_load_balancing_metrics(self) -> Dict[str, float]:
        """Get detailed load balancing metrics for analysis."""
        workloads = []
        
        # Collect all workloads
        for uav in self.uav_manager.static_uavs.values():
            workloads.append(uav.total_workload / uav.cpu_capacity)
        
        for uav in self.uav_manager.dynamic_uavs.values():
            if uav.is_available:
                workloads.append(uav.total_workload / uav.cpu_capacity)
        
        for satellite in self.satellite_constellation.satellites.values():
            workloads.append(satellite.total_workload / satellite.cpu_capacity)
        
        # Initialize communication models if needed
        self._initialize_communication_models()
        
        if self.load_balancing_metrics:
            return self.load_balancing_metrics.calculate_comprehensive_load_metrics(workloads)
        else:
            # Fallback to basic metrics
            if not workloads:
                return {
                    'load_imbalance_coefficient': 0.0,
                    'fairness_index': 1.0,
                    'peak_to_average_ratio': 1.0,
                    'coefficient_of_variation': 0.0
                }
            
            mean_workload = np.mean(workloads)
            if mean_workload == 0:
                cv = 0.0
            else:
                cv = np.std(workloads) / mean_workload
            
            return {
                'load_imbalance_coefficient': cv,
                'fairness_index': 1.0 / (1.0 + cv),
                'peak_to_average_ratio': np.max(workloads) / mean_workload if mean_workload > 0 else 1.0,
                'coefficient_of_variation': cv
            }
    
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
    
    def get_system_state_report(self, detailed: bool = True) -> Dict[str, Any]:
        """Get a comprehensive system state report."""
        report = {
            'timestamp': self.current_time,
            'epoch': self.epoch_count,
            'regions': {},
            'vehicles': {},
            'uavs': {},
            'satellites': {},
            'tasks': {},
            'metrics': self.metrics.__dict__.copy()
        }
        
        # Region information
        for region_id, region in self.regions.items():
            vehicles_in_region = self.vehicle_manager.get_vehicles_in_region(region_id)
            pending_tasks = self.task_manager.get_tasks_for_region(region_id, max_tasks=100)
            
            report['regions'][region_id] = {
                'center': (region.center.x, region.center.y),
                'radius': region.radius,
                'vehicle_count': len(vehicles_in_region),
                'pending_tasks': len(pending_tasks),
                'vehicle_ids': [v.id for v in vehicles_in_region] if detailed else None,
                'task_ids': [t.id for t in pending_tasks] if detailed else None
            }
        
        # Vehicle information
        vehicle_counts = self.vehicle_manager.get_vehicle_count_by_region()
        report['vehicles'] = {
            'total_count': len(self.vehicle_manager.vehicles),
            'by_region': vehicle_counts,
            'details': {}
        }
        
        if detailed:
            for vehicle_id, vehicle in self.vehicle_manager.vehicles.items():
                speed = np.sqrt(vehicle.velocity.vx**2 + vehicle.velocity.vy**2 + vehicle.velocity.vz**2)
                report['vehicles']['details'][vehicle_id] = {
                    'position': (vehicle.position.x, vehicle.position.y),
                    'speed': speed,
                    'region_id': vehicle.current_region_id
                }
        
        # UAV information
        static_uavs_info = {}
        dynamic_uavs_info = {}
        
        for uav_id, uav in self.uav_manager.static_uavs.items():
            static_uavs_info[uav_id] = {
                'region_id': uav.assigned_region_id,
                'workload': uav.total_workload,
                'queue_length': uav.queue_length,
                'energy_level': uav.current_energy,
                'energy_percentage': (uav.current_energy / uav.battery_capacity) * 100
            }
        
        for uav_id, uav in self.uav_manager.dynamic_uavs.items():
            dynamic_uavs_info[uav_id] = {
                'current_region_id': getattr(uav, 'current_region_id', None),
                'target_region_id': getattr(uav, 'target_region_id', None),
                'status': uav.status.value,
                'workload': uav.total_workload,
                'energy_level': uav.current_energy,
                'energy_percentage': (uav.current_energy / uav.battery_capacity) * 100
            }
        
        report['uavs'] = {
            'static_count': len(static_uavs_info),
            'dynamic_count': len(dynamic_uavs_info),
            'static_uavs': static_uavs_info,
            'dynamic_uavs': dynamic_uavs_info
        }
        
        # Satellite information
        satellites_info = {}
        for sat_id, satellite in self.satellite_constellation.satellites.items():
            satellites_info[sat_id] = {
                'workload': satellite.total_workload,
                'queue_length': satellite.queue_length,
                'orbital_params': getattr(satellite, 'orbital_params', None)
            }
        
        # Calculate satellite visibility per region (from UAV positions)
        visibility_info = {}
        for region_id, region in self.regions.items():
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            check_position = static_uav.position if static_uav else region.center
            visible_sats = self.satellite_constellation.find_visible_satellites(check_position)
            visibility_info[region_id] = len(visible_sats)
        
        report['satellites'] = {
            'total_count': len(satellites_info),
            'satellites': satellites_info,
            'visibility_per_region': visibility_info
        }
        
        # Task information
        total_pending = 0
        tasks_by_region = {}
        tasks_by_size = {'small': 0, 'medium': 0, 'large': 0}
        
        for region_id in self.regions.keys():
            pending_tasks = self.task_manager.get_tasks_for_region(region_id, max_tasks=100)
            tasks_by_region[region_id] = len(pending_tasks)
            total_pending += len(pending_tasks)
            
            for task in pending_tasks:
                if task.data_size_in < 10:
                    tasks_by_size['small'] += 1
                elif task.data_size_in < 50:
                    tasks_by_size['medium'] += 1
                else:
                    tasks_by_size['large'] += 1
        
        report['tasks'] = {
            'total_pending': total_pending,
            'by_region': tasks_by_region,
            'by_size': tasks_by_size,
            'total_generated': self.metrics.total_tasks_generated,
            'total_completed': self.metrics.total_tasks_completed,
            'total_failed': self.metrics.total_tasks_failed
        }
        
        return report
    
    def print_system_state_report(self, detailed: bool = True) -> None:
        """Print a comprehensive system state report."""
        report = self.get_system_state_report(detailed)
        
        print(f"\n🔍 SYSTEM STATE REPORT - Epoch {report['epoch']} (t={report['timestamp']:.1f}s)")
        print("=" * 60)
        
        # Regions
        print(f"📍 REGIONS ({len(report['regions'])} total):")
        for region_id, region_info in report['regions'].items():
            print(f"  Region {region_id}: {region_info['vehicle_count']} vehicles, "
                  f"{region_info['pending_tasks']} pending tasks")
            if detailed and region_info['vehicle_ids']:
                print(f"    Vehicles: {region_info['vehicle_ids']}")
        
        # Vehicles
        print(f"\n🚗 VEHICLES ({report['vehicles']['total_count']} total):")
        for region_id, count in report['vehicles']['by_region'].items():
            if count > 0:
                print(f"  Region {region_id}: {count} vehicles")
        
        # UAVs
        print(f"\n🚁 UAVs (Static: {report['uavs']['static_count']}, Dynamic: {report['uavs']['dynamic_count']}):")
        for uav_id, uav_info in report['uavs']['static_uavs'].items():
            print(f"  Static UAV {uav_id} (R{uav_info['region_id']}): "
                  f"workload={uav_info['workload']:.0f}, queue={uav_info['queue_length']}, "
                  f"energy={uav_info['energy_percentage']:.1f}%")
        
        for uav_id, uav_info in report['uavs']['dynamic_uavs'].items():
            print(f"  Dynamic UAV {uav_id}: {uav_info['status']} "
                  f"(R{uav_info['current_region_id']}→R{uav_info['target_region_id']}), "
                  f"workload={uav_info['workload']:.0f}, energy={uav_info['energy_percentage']:.1f}%")
        
        # Satellites
        print(f"\n🛰️  SATELLITES ({report['satellites']['total_count']} total):")
        total_sat_workload = sum(sat['workload'] for sat in report['satellites']['satellites'].values())
        print(f"  Total satellite workload: {total_sat_workload:.0f}")
        print(f"  Visibility per region: {report['satellites']['visibility_per_region']}")
        
        # Tasks
        print(f"\n📋 TASKS:")
        print(f"  Generated: {report['tasks']['total_generated']}")
        print(f"  Completed: {report['tasks']['total_completed']}")
        print(f"  Failed: {report['tasks']['total_failed']}")
        print(f"  Pending: {report['tasks']['total_pending']}")
        if report['tasks']['by_size']:
            print(f"  By size: {report['tasks']['by_size']}")
        
        # Metrics
        print(f"\n📊 METRICS:")
        print(f"  Success rate: {report['metrics']['success_rate']:.3f}")
        print(f"  Average latency: {report['metrics']['average_latency']:.3f}s")
        print(f"  UAV utilization: {report['metrics']['uav_utilization']:.3f}")
        print(f"  Satellite utilization: {report['metrics']['satellite_utilization']:.3f}")
        print(f"  Energy consumption: {report['metrics']['energy_consumption']:.1f}J")
        print(f"  Load imbalance: {report['metrics']['load_imbalance']:.3f}")
        print(f"  Coverage: {report['metrics']['coverage_percentage']:.1f}%")
        
        print("=" * 60)
    
    def track_decision(self, task_id: str, region_id: int, decision: str, 
                      resources_available: Dict[str, Any], success: bool) -> None:
        """Track a task offloading decision for analysis."""
        if not self.log_decisions:
            return
            
        decision_record = {
            'timestamp': self.current_time,
            'epoch': self.epoch_count,
            'task_id': task_id,
            'region_id': region_id,
            'decision': decision,
            'resources_available': resources_available.copy(),
            'success': success
        }
        self.decision_history.append(decision_record)
        
        # Keep only last 1000 decisions to prevent memory issues
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def track_resource_utilization(self) -> None:
        """Track current resource utilization for analysis."""
        if not self.log_resource_usage:
            return
            
        utilization_record = {
            'timestamp': self.current_time,
            'epoch': self.epoch_count,
            'static_uav_utilization': {},
            'dynamic_uav_utilization': {},
            'satellite_utilization': {},
            'regional_load': {}
        }
        
        # Track static UAV utilization
        for uav_id, uav in self.uav_manager.static_uavs.items():
            utilization_record['static_uav_utilization'][uav_id] = {
                'region_id': uav.assigned_region_id,
                'workload': uav.total_workload,
                'queue_length': uav.queue_length,
                'energy_percentage': (uav.current_energy / uav.battery_capacity) * 100,
                'utilization': min(1.0, uav.total_workload / uav.cpu_capacity)
            }
        
        # Track dynamic UAV utilization
        for uav_id, uav in self.uav_manager.dynamic_uavs.items():
            utilization_record['dynamic_uav_utilization'][uav_id] = {
                'current_region': getattr(uav, 'current_region_id', None),
                'target_region': getattr(uav, 'target_region_id', None),
                'status': uav.status.value,
                'workload': uav.total_workload,
                'energy_percentage': (uav.current_energy / uav.battery_capacity) * 100,
                'utilization': min(1.0, uav.total_workload / uav.cpu_capacity) if uav.is_available else 0.0
            }
        
        # Track satellite utilization
        for sat_id, satellite in self.satellite_constellation.satellites.items():
            utilization_record['satellite_utilization'][sat_id] = {
                'workload': satellite.total_workload,
                'queue_length': satellite.queue_length,
                'utilization': min(1.0, satellite.total_workload / satellite.cpu_capacity)
            }
        
        # Track regional load
        for region_id in self.regions.keys():
            pending_tasks = len(self.task_manager.get_tasks_for_region(region_id, max_tasks=100))
            vehicle_count = len(self.vehicle_manager.get_vehicles_in_region(region_id))
            visible_sats = len(self.satellite_constellation.find_visible_satellites(self.regions[region_id].center))
            
            utilization_record['regional_load'][region_id] = {
                'pending_tasks': pending_tasks,
                'vehicle_count': vehicle_count,
                'visible_satellites': visible_sats,
                'task_density': pending_tasks / max(1, vehicle_count)
            }
        
        self.resource_utilization_history.append(utilization_record)
        
        # Keep only last 500 records
        if len(self.resource_utilization_history) > 500:
            self.resource_utilization_history = self.resource_utilization_history[-500:]
    
    def get_decision_statistics(self, last_n_epochs: int = 50) -> Dict[str, Any]:
        """Get decision statistics for the last N epochs."""
        if not self.decision_history:
            return {'error': 'No decision history available'}
        
        # Filter decisions from last N epochs
        min_epoch = max(0, self.epoch_count - last_n_epochs)
        recent_decisions = [d for d in self.decision_history if d['epoch'] >= min_epoch]
        
        if not recent_decisions:
            return {'error': 'No recent decisions available'}
        
        # Calculate statistics
        total_decisions = len(recent_decisions)
        decision_counts = {}
        success_counts = {}
        regional_counts = {}
        resource_analysis = {'avg_dynamic_uavs': 0, 'avg_satellites': 0}
        
        for decision in recent_decisions:
            dec_type = decision['decision']
            region_id = decision['region_id']
            
            decision_counts[dec_type] = decision_counts.get(dec_type, 0) + 1
            regional_counts[region_id] = regional_counts.get(region_id, 0) + 1
            
            if decision['success']:
                success_counts[dec_type] = success_counts.get(dec_type, 0) + 1
            
            # Analyze resource availability
            resources = decision['resources_available']
            resource_analysis['avg_dynamic_uavs'] += resources.get('dynamic_uavs', 0)
            resource_analysis['avg_satellites'] += resources.get('satellites', 0)
        
        # Calculate averages
        resource_analysis['avg_dynamic_uavs'] /= total_decisions
        resource_analysis['avg_satellites'] /= total_decisions
        
        # Calculate success rates
        success_rates = {}
        for dec_type, count in decision_counts.items():
            success_rates[dec_type] = success_counts.get(dec_type, 0) / count
        
        return {
            'total_decisions': total_decisions,
            'decision_counts': decision_counts,
            'decision_percentages': {k: (v/total_decisions)*100 for k, v in decision_counts.items()},
            'success_rates': success_rates,
            'regional_distribution': regional_counts,
            'resource_analysis': resource_analysis,
            'epoch_range': (min_epoch, self.epoch_count)
        }
    
    def print_decision_analysis(self, last_n_epochs: int = 50) -> None:
        """Print detailed decision analysis."""
        stats = self.get_decision_statistics(last_n_epochs)
        
        if 'error' in stats:
            print(f"    Decision Analysis: {stats['error']}")
            return
        
        print(f"\n    📊 DECISION ANALYSIS (Last {last_n_epochs} epochs)")
        print("    " + "="*45)
        print(f"    Total Decisions: {stats['total_decisions']}")
        print(f"    Epoch Range: {stats['epoch_range'][0]} - {stats['epoch_range'][1]}")
        print()
        
        print("    Decision Distribution:")
        for decision_type, count in stats['decision_counts'].items():
            percentage = stats['decision_percentages'][decision_type]
            success_rate = stats['success_rates'][decision_type]
            print(f"      {decision_type.upper()}: {count} ({percentage:.1f}%) - Success: {success_rate:.3f}")
        
        print(f"\n    Regional Distribution:")
        for region_id, count in stats['regional_distribution'].items():
            percentage = (count / stats['total_decisions']) * 100
            print(f"      Region {region_id}: {count} decisions ({percentage:.1f}%)")
        
        print(f"\n    Resource Availability:")
        print(f"      Avg Dynamic UAVs: {stats['resource_analysis']['avg_dynamic_uavs']:.1f}")
        print(f"      Avg Satellites: {stats['resource_analysis']['avg_satellites']:.1f}")
        
        print("    " + "="*45)
    
    def get_resource_utilization_trends(self, last_n_epochs: int = 50) -> Dict[str, Any]:
        """Get resource utilization trends over time."""
        if not self.resource_utilization_history:
            return {'error': 'No resource utilization history available'}
        
        # Filter recent records
        min_epoch = max(0, self.epoch_count - last_n_epochs)
        recent_records = [r for r in self.resource_utilization_history if r['epoch'] >= min_epoch]
        
        if not recent_records:
            return {'error': 'No recent utilization records available'}
        
        # Calculate trends
        trends = {
            'static_uav_trends': {},
            'dynamic_uav_trends': {},
            'satellite_trends': {},
            'regional_trends': {},
            'system_trends': {
                'avg_static_utilization': [],
                'avg_dynamic_utilization': [],
                'avg_satellite_utilization': [],
                'total_pending_tasks': []
            }
        }
        
        for record in recent_records:
            epoch = record['epoch']
            
            # System-wide trends
            static_utils = [info['utilization'] for info in record['static_uav_utilization'].values()]
            dynamic_utils = [info['utilization'] for info in record['dynamic_uav_utilization'].values()]
            sat_utils = [info['utilization'] for info in record['satellite_utilization'].values()]
            total_pending = sum(info['pending_tasks'] for info in record['regional_load'].values())
            
            trends['system_trends']['avg_static_utilization'].append(np.mean(static_utils) if static_utils else 0)
            trends['system_trends']['avg_dynamic_utilization'].append(np.mean(dynamic_utils) if dynamic_utils else 0)
            trends['system_trends']['avg_satellite_utilization'].append(np.mean(sat_utils) if sat_utils else 0)
            trends['system_trends']['total_pending_tasks'].append(total_pending)
        
        return trends
    
    def print_resource_utilization_summary(self, last_n_epochs: int = 25) -> None:
        """Print resource utilization summary."""
        trends = self.get_resource_utilization_trends(last_n_epochs)
        
        if 'error' in trends:
            print(f"    Resource Analysis: {trends['error']}")
            return
        
        print(f"\n    📈 RESOURCE UTILIZATION TRENDS (Last {last_n_epochs} epochs)")
        print("    " + "="*50)
        
        # Current utilization
        if self.resource_utilization_history:
            latest = self.resource_utilization_history[-1]
            
            print("    Current Utilization:")
            static_utils = [info['utilization'] for info in latest['static_uav_utilization'].values()]
            dynamic_utils = [info['utilization'] for info in latest['dynamic_uav_utilization'].values()]
            sat_utils = [info['utilization'] for info in latest['satellite_utilization'].values()]
            
            print(f"      Static UAVs: {np.mean(static_utils):.3f} (max: {max(static_utils):.3f})")
            print(f"      Dynamic UAVs: {np.mean(dynamic_utils):.3f} (max: {max(dynamic_utils):.3f})")
            print(f"      Satellites: {np.mean(sat_utils):.3f} (max: {max(sat_utils):.3f})")
            
            print(f"\n    Regional Load:")
            for region_id, load_info in latest['regional_load'].items():
                print(f"      Region {region_id}: {load_info['pending_tasks']} tasks, "
                      f"density: {load_info['task_density']:.2f}, "
                      f"satellites: {load_info['visible_satellites']}")
        
        # Trends
        system_trends = trends['system_trends']
        if system_trends['avg_static_utilization']:
            print(f"\n    Trend Analysis:")
            print(f"      Static UAV utilization: {system_trends['avg_static_utilization'][-1]:.3f}")
            print(f"      Dynamic UAV utilization: {system_trends['avg_dynamic_utilization'][-1]:.3f}")
            print(f"      Satellite utilization: {system_trends['avg_satellite_utilization'][-1]:.3f}")
            print(f"      Total pending tasks: {system_trends['total_pending_tasks'][-1]}")
        
        print("    " + "="*50)
    
    def step_with_detailed_logging(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Execute one simulation step with maximum detailed logging."""
        if not self.is_running:
            return {}
        
        if dt is None:
            dt = self.system_params.epoch_duration
        
        print(f"\n🔍 DETAILED EPOCH {self.epoch_count} (t={self.current_time:.1f}s)")
        print("="*70)
        
        # Track resource utilization before step
        self.track_resource_utilization()
        
        # Execute normal step with verbose logging
        step_results = self.step(dt, verbose=True)
        
        # Print detailed decision analysis every 10 epochs
        if self.epoch_count % 10 == 0:
            self.print_decision_analysis(10)
        
        # Print resource utilization summary every 15 epochs
        if self.epoch_count % 15 == 0:
            self.print_resource_utilization_summary(15)
        
        # Print system state report every 25 epochs
        if self.epoch_count % 25 == 0:
            self.print_system_state_report(detailed=False)
        
        return step_results
    
    def run_simulation_with_detailed_logging(self, num_epochs: int, 
                                           detailed_interval: int = 1,
                                           progress_callback: Optional[callable] = None) -> None:
        """Run simulation with detailed logging at specified intervals."""
        print(f"Starting detailed simulation for {num_epochs} epochs...")
        print(f"Detailed logging every {detailed_interval} epoch(s)")
        
        for epoch in range(num_epochs):
            if epoch % detailed_interval == 0:
                step_results = self.step_with_detailed_logging()
            else:
                step_results = self.step(verbose=False)
            
            if progress_callback:
                progress_callback(epoch, num_epochs, step_results)
            
            # Print progress every 50 epochs
            if epoch % 50 == 0:
                print(f"\n📊 Progress: Epoch {epoch}/{num_epochs} "
                      f"(Success rate: {self.metrics.success_rate:.3f})")
        
        print(f"\n✅ Detailed simulation completed after {num_epochs} epochs")
        print(f"Final metrics: Success rate: {self.metrics.success_rate:.3f}, "
              f"Average latency: {self.metrics.average_latency:.3f}s")
        
        # Print final comprehensive analysis
        print("\n🎯 FINAL SIMULATION ANALYSIS")
        print("="*60)
        self.print_decision_analysis(100)
        self.print_resource_utilization_summary(50)
        self.print_system_state_report(detailed=False)
    
    def analyze_communication_performance(self) -> Dict[str, Any]:
        """Analyze communication performance using Shannon capacity model."""
        # Initialize communication models if needed
        self._initialize_communication_models()
        
        analysis_results = {
            'link_performances': [],
            'average_capacity': 0.0,
            'capacity_variance': 0.0,
            'bottleneck_links': []
        }
        
        capacities = []
        
        # Check if communication models are available
        if not self.shannon_capacity_model:
            # Fallback to basic analysis
            return {
                'link_performances': [],
                'average_capacity': 50.0,  # Default value
                'capacity_variance': 10.0,
                'bottleneck_links': []
            }
        
        # Analyze all active communication links
        for region_id, region in self.regions.items():
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if not static_uav:
                continue
            
            # Analyze vehicle-to-UAV links
            vehicles_in_region = self.vehicle_manager.get_vehicles_in_region(region_id)
            for vehicle in vehicles_in_region:
                # Calculate communication capacity for this link
                capacity_result = self.shannon_capacity_model.calculate_capacity(
                    source_position=vehicle.position,
                    destination_position=static_uav.position,
                    source_power=10.0,  # 10W transmit power
                    dest_antenna_gain=5.0,  # 5dB antenna gain
                    optimize_params=True
                )
                
                link_info = {
                    'source_type': 'vehicle',
                    'destination_type': 'static_uav',
                    'source_id': vehicle.id,
                    'destination_id': static_uav.id,
                    'capacity_mbps': capacity_result['reliable_capacity_bps'] / 1e6,
                    'modulation_scheme': capacity_result['modulation_scheme'],
                    'coding_rate': capacity_result['coding_rate'],
                    'snr_db': capacity_result['snr_db']
                }
                
                analysis_results['link_performances'].append(link_info)
                capacities.append(link_info['capacity_mbps'])
        
        # Calculate statistics
        if capacities:
            analysis_results['average_capacity'] = np.mean(capacities)
            analysis_results['capacity_variance'] = np.var(capacities)
            
            # Identify bottleneck links (bottom 10% of capacity)
            threshold = np.percentile(capacities, 10)
            analysis_results['bottleneck_links'] = [
                link for link in analysis_results['link_performances']
                if link['capacity_mbps'] <= threshold
            ]
        
        return analysis_results
    
    def analyze_latency_performance(self, task_sample_size: int = 10) -> Dict[str, Any]:
        """Analyze latency performance using advanced latency model."""
        # For now, return a simplified analysis
        return {
            'task_latencies': [],
            'average_latency': 2.0,
            'latency_variance': 0.5,
            'bottleneck_components': {},
            'optimization_recommendations': []
        }
    
    def get_region_by_task(self, task):
        """Get the region where a task was generated."""
        if hasattr(task, 'region_id'):
            return self.regions.get(task.region_id)
        return None
    
    def _initialize_communication_models(self):
        """Initialize communication models dynamically when needed."""
        if self.communication_model is None:
            try:
                from ..models.communication import CommunicationModel, LoadBalancingMetrics, ShannonCapacityModel
                self.communication_model = CommunicationModel(self.system_params)
                self.load_balancing_metrics = LoadBalancingMetrics()
                self.shannon_capacity_model = ShannonCapacityModel(self.system_params)
            except ImportError as e:
                print(f"Warning: Could not import communication models: {e}")
                # Create fallback models
                self.communication_model = None
                self.load_balancing_metrics = None
                self.shannon_capacity_model = None
        
        if self.latency_model is None:
            try:
                from ..models.latency import LatencyModel
                self.latency_model = LatencyModel(self.system_params)
            except ImportError as e:
                print(f"Warning: Could not import latency model: {e}")
                self.latency_model = None
    
    def setup_network_topology_with_grid(self, grid_config, task_config=None) -> None:
        """Setup network topology using a grid configuration object."""
        self.setup_network_topology(grid_config=grid_config, task_config=task_config)
    
    def _reposition_dynamic_uavs(self, verbose: bool = False):
        """Reposition dynamic UAVs to balance load across regions."""
        if not self.uav_manager.dynamic_uavs:
            if verbose:
                print(f"    🔄 No dynamic UAVs available for repositioning")
            return
            
        # Calculate region loads (static UAV workload / capacity)
        region_loads = {}
        for region_id in self.regions.keys():
            static_uav = self.uav_manager.get_static_uav_by_region(region_id)
            if static_uav:
                load = static_uav.total_workload / static_uav.cpu_capacity
                region_loads[region_id] = min(1.0, load)
            else:
                region_loads[region_id] = 0.0
        
        if verbose:
            print(f"    📊 Region loads: {region_loads}")
        
        # Find overloaded regions (load > 0.2) and underloaded regions (load < 0.05)
        # Lowered thresholds to make repositioning more likely
        overloaded_regions = [r for r, load in region_loads.items() if load > 0.2]
        underloaded_regions = [r for r, load in region_loads.items() if load < 0.05]
        
        if verbose:
            print(f"    🔴 Overloaded regions (>0.3): {overloaded_regions}")
            print(f"    🟢 Underloaded regions (<0.1): {underloaded_regions}")
        
        if not overloaded_regions:
            if verbose:
                print(f"    ✅ No overloaded regions found, no repositioning needed")
            return
            
        # Find available dynamic UAVs (not currently flying)
        available_dynamic_uavs = [
            uav for uav in self.uav_manager.dynamic_uavs.values() 
            if uav.status == UAVStatus.ACTIVE and uav.current_energy > uav.min_energy_threshold
        ]
        
        if not available_dynamic_uavs:
            if verbose:
                print(f"    ❌ No available dynamic UAVs for repositioning")
            return
            
        if verbose:
            print(f"    🚁 Available dynamic UAVs: {[uav.id for uav in available_dynamic_uavs]}")
        
        # Assign available dynamic UAVs to overloaded regions
        import random
        repositioned = 0
        for uav in available_dynamic_uavs[:len(overloaded_regions)]:  # Limit to available slots
            target_region = random.choice(overloaded_regions)
            region_center = self.regions[target_region].center
            
            # Only move if not already in the target region
            if uav.assigned_region_id != target_region:
                flight_time = uav.assign_to_region(target_region, region_center, self.current_time)
                repositioned += 1
                
                if verbose:
                    print(f"      ✈️  UAV {uav.id}: moving from region {uav.assigned_region_id} to region {target_region} "
                          f"(flight time: {flight_time:.1f}s)")
                
                # Remove this region from overloaded list to avoid multiple UAVs going to same region
                if target_region in overloaded_regions:
                    overloaded_regions.remove(target_region)
            else:
                if verbose:
                    print(f"      ✅ UAV {uav.id}: already in target region {target_region}")
        
        if verbose and repositioned == 0:
            print(f"    ℹ️  No UAVs needed repositioning")
