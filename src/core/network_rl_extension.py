"""
Extension of the SAGINNetwork class to support RL-based orchestration.
This file contains methods to switch between heuristic and RL-based orchestration.
"""

from typing import Dict, Any, Optional
import numpy as np
import time

# Import the original methods to extend
from .network import SAGINNetwork
from .types import TaskDecision, Position
from .uavs import UAVStatus
from ..rl.agents import CentralAgent, LocalAgent


# Add RL orchestration methods to SAGINNetwork class
def set_heuristic_orchestration(self) -> None:
    """Set network to use heuristic-based orchestration (original mode)."""
    self.orchestration_mode = "heuristic"
    self._original_process_task_offloading = self._process_task_offloading
    self._original_step = self.step
    
    # Reset any RL-related attributes
    self.central_agent = None
    self.local_agents = {}
    print("✅ Heuristic orchestration mode activated")


def set_rl_orchestration(self, central_agent: CentralAgent, local_agents: Dict[int, LocalAgent]) -> None:
    """Set network to use RL-based orchestration.
    
    Args:
        central_agent: Trained central RL agent
        local_agents: Dictionary of trained local RL agents by region ID
    """
    self.orchestration_mode = "rl"
    self.central_agent = central_agent
    self.local_agents = local_agents
    
    # Keep reference to original methods
    self._original_process_task_offloading = self._process_task_offloading
    self._original_step = self.step
    
    # Override with RL methods
    self._process_task_offloading = self._process_task_offloading_rl
    
    print(f"✅ RL orchestration mode activated with {len(local_agents)} local agents")


def get_task_offloading_decision_rl(self, task, static_uav_id: int, region_id: int) -> TaskDecision:
    """Make task offloading decision using RL agent.
    
    Args:
        task: Task object to make decision for
        static_uav_id: ID of static UAV making the decision
        region_id: Region ID where task originated
        
    Returns:
        TaskDecision (LOCAL, DYNAMIC, SATELLITE)
    """
    # Check if we have a trained local agent for this region
    if region_id not in self.local_agents:
        # Fall back to heuristic if no agent for this region
        static_uav = self.uav_manager.static_uavs.get(static_uav_id)
        if static_uav:
            available_dynamic_uavs = self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
            dynamic_uav_count = len(available_dynamic_uavs)
            visible_satellites = self.satellite_constellation.find_visible_satellites(static_uav.position)
            satellite_available = len(visible_satellites) > 0
            
            # Use heuristic decision
            return static_uav.make_offloading_decision(
                task, dynamic_uav_count, satellite_available, self.current_time
            )
        return TaskDecision.FAILED
    
    # Get state from the static UAV's perspective
    static_uav = self.uav_manager.static_uavs.get(static_uav_id)
    region = self.regions[region_id]
    
    state = {
        # Queue length at static UAV
        'queue_length': static_uav.queue_length,
        
        # Residual energy
        'residual_energy': static_uav.current_energy / static_uav.battery_capacity,
        
        # Number of dynamic UAVs in region
        'dynamic_uavs': len(self.uav_manager.get_available_dynamic_uavs_in_region(region_id)),
        
        # Task intensity in region
        'task_intensity': region.current_intensity,
        
        # Task properties
        'task_size': task.data_size_in / 10.0,  # Normalized
        'task_cycles': task.cpu_cycles / 1e9,   # Normalized
        'task_urgency': max(0.0, 1.0 - (task.deadline - self.current_time) / 10.0)  # Normalized
    }
    
    # Get action from agent
    local_agent = self.local_agents[region_id]
    
    # Create task info dictionary for the agent
    task_info = {
        'data_size': task.data_size_in,
        'cpu_cycles': task.cpu_cycles,
        'deadline': task.deadline,
        'priority': getattr(task, 'priority', 1.0)
    }
    
    # Use state dictionary instead of array
    action = local_agent.select_action(state, task_info, explore=False)
    
    # Convert action string to task decision
    if action == 'local':
        return TaskDecision.LOCAL
    elif action == 'dynamic':
        return TaskDecision.DYNAMIC
    elif action == 'satellite':
        return TaskDecision.SATELLITE
    else:
        # Default fallback
        return TaskDecision.LOCAL


def _process_task_offloading_rl(self, verbose: bool = True) -> Dict[str, int]:
    """Process task offloading decisions using RL agents."""
    decisions_made = {'local': 0, 'dynamic': 0, 'satellite': 0, 'failed': 0}
    
    # First, if central agent exists and it's time for central allocation,
    # allocate dynamic UAVs to regions
    if self.central_agent and self.epoch_count % 5 == 0:  # Every 5 epochs
        self._allocate_dynamic_uavs_with_rl()
    
    # Then process offloading decisions for all regions
    for region_id, region in self.regions.items():
        static_uav = self.uav_manager.get_static_uav_by_region(region_id)
        if not static_uav:
            continue
        
        # Get pending tasks for this region
        pending_tasks = self.task_manager.get_tasks_for_region(region_id, max_tasks=10)
        
        if pending_tasks and verbose:
            print(f"    Region {region_id}: Processing {len(pending_tasks)} pending tasks")
        
        for task in pending_tasks:
            # Get decision from RL or fallback to heuristic
            decision = self.get_task_offloading_decision_rl(task, static_uav.id, region_id)
            
            # Log detailed decision rationale
            if verbose:
                urgency = "URGENT" if (task.deadline - self.current_time) < 5.0 else "normal"
                print(f"      Task {task.id} (size={task.data_size_in}MB, {urgency}): Decision={decision.value}")
                
                # Get available resources for logging
                available_dynamic_uavs = self.uav_manager.get_available_dynamic_uavs_in_region(region_id)
                dynamic_uav_count = len(available_dynamic_uavs)
                visible_satellites = self.satellite_constellation.find_visible_satellites(static_uav.position)
                static_uav_load = static_uav.total_workload
                min_dynamic_load = min([uav.total_workload for uav in available_dynamic_uavs]) if available_dynamic_uavs else float('inf')
                
                print(f"        Resources: Static load={static_uav_load:.0f}, Dynamic UAVs={dynamic_uav_count}, "
                      f"Satellites={len(visible_satellites)}")
                print(f"        Constraints: Deadline={task.deadline - self.current_time:.1f}s, "
                      f"Cycles={task.cpu_cycles:.0f}, Size={task.data_size_in}MB")
            
            # Track decision resources for analysis
            resources_available = {
                'dynamic_uavs': len(self.uav_manager.get_available_dynamic_uavs_in_region(region_id)),
                'satellites': len(self.satellite_constellation.find_visible_satellites(static_uav.position)),
                'static_uav_load': static_uav.total_workload,
                'min_dynamic_load': min([uav.total_workload for uav in available_dynamic_uavs]) if available_dynamic_uavs else float('inf')
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


def _allocate_dynamic_uavs_with_rl(self) -> None:
    """Allocate dynamic UAVs to regions using central RL agent."""
    if not self.central_agent:
        return
    
    # Get available dynamic UAVs
    available_uavs = [uav_id for uav_id, uav in self.uav_manager.dynamic_uavs.items() 
                     if uav.is_available]
    
    if not available_uavs:
        return  # No UAVs to allocate
    
    # Get region IDs
    region_ids = list(self.regions.keys())
    
    # Build state dictionary in the format expected by CentralAgent
    # The model was trained with fixed sizes: 16 regions, 5 dynamic UAVs
    EXPECTED_REGIONS = 16
    EXPECTED_DYNAMIC_UAVS = 5
    
    # Pad or truncate regions to match expected size
    all_region_ids = list(range(1, EXPECTED_REGIONS + 1))
    
    state_dict = {
        'regions': {rid: {'id': rid} for rid in all_region_ids},
        'task_arrival_rates': {},
        'static_uav_queues': {},
        'static_uav_energy': {},
        'dynamic_uav_positions': {},
        'dynamic_uav_availability': {},
        'current_epoch': self.epoch_count
    }
    
    # Fill actual region data, pad missing regions with zeros
    for rid in all_region_ids:
        if rid in self.regions:
            state_dict['task_arrival_rates'][rid] = self._get_region_task_rate(rid)
            state_dict['static_uav_queues'][rid] = self._get_static_uav_queue(rid)
            state_dict['static_uav_energy'][rid] = self._get_static_uav_energy(rid)
        else:
            state_dict['task_arrival_rates'][rid] = 0.0
            state_dict['static_uav_queues'][rid] = 0.0
            state_dict['static_uav_energy'][rid] = 0.0
    
    # Pad or truncate UAVs to match expected size
    padded_uav_ids = list(range(1, EXPECTED_DYNAMIC_UAVS + 1))
    actual_uavs = available_uavs[:EXPECTED_DYNAMIC_UAVS]  # Take first N available
    
    for i, uav_id in enumerate(padded_uav_ids):
        if i < len(actual_uavs):
            real_uav_id = actual_uavs[i]
            state_dict['dynamic_uav_positions'][uav_id] = (
                self.uav_manager.dynamic_uavs[real_uav_id].position.x,
                self.uav_manager.dynamic_uavs[real_uav_id].position.y,
                self.uav_manager.dynamic_uavs[real_uav_id].position.z
            )
            state_dict['dynamic_uav_availability'][uav_id] = 1.0
        else:
            # Pad with dummy UAV at origin
            state_dict['dynamic_uav_positions'][uav_id] = (0.0, 0.0, 0.0)
            state_dict['dynamic_uav_availability'][uav_id] = 0.0
    
    # Get action from central agent - use the fixed expected region/UAV lists
    actions = self.central_agent.select_action(state_dict, padded_uav_ids, all_region_ids, explore=False)
    
    # Apply actions: map from padded UAV IDs back to real UAV IDs
    for padded_uav_id, target_region_id in actions.items():
        # Find the real UAV corresponding to this padded ID
        if padded_uav_id <= len(actual_uavs):
            real_uav_id = actual_uavs[padded_uav_id - 1]  # Convert 1-based to 0-based index
            
            if real_uav_id not in self.uav_manager.dynamic_uavs:
                continue
                
            uav = self.uav_manager.dynamic_uavs[real_uav_id]
            
            # Validate region ID (only apply to regions that actually exist)
            if target_region_id not in self.regions:
                continue  # Skip invalid region assignment
            
            # Skip if already in this region
            if uav.assigned_region_id == target_region_id:
                continue
                
            # Assign to new region using the proper method
            region = self.regions[target_region_id]
            uav.assign_to_region(target_region_id, region.center, self.current_time)


def _get_global_state_for_rl(self) -> list:
    """Get global state representation for central RL agent.
    
    Returns:
        List of state features
    """
    state_features = []
    
    # 1. Task arrival rates by region
    for region_id in sorted(self.regions.keys()):
        region = self.regions[region_id]
        state_features.append(region.current_intensity)
    
    # 2. Queue lengths at static UAVs
    for region_id in sorted(self.regions.keys()):
        static_uav = self.uav_manager.get_static_uav_by_region(region_id)
        queue_length = static_uav.queue_length if static_uav else 0
        state_features.append(queue_length)
    
    # 3. Residual energy of static UAVs
    for region_id in sorted(self.regions.keys()):
        static_uav = self.uav_manager.get_static_uav_by_region(region_id)
        energy = (static_uav.current_energy / static_uav.battery_capacity) if static_uav else 0.0
        state_features.append(energy)
    
    # 4. Availability of dynamic UAVs
    dynamic_uavs = list(self.uav_manager.dynamic_uavs.values())
    for uav in dynamic_uavs:
        state_features.append(uav.availability_indicator)
    
    # 5. Positions of dynamic UAVs (normalized by area bounds)
    area_width = self.grid.area_bounds[1] - self.grid.area_bounds[0]
    area_height = self.grid.area_bounds[3] - self.grid.area_bounds[2]
    
    for uav in dynamic_uavs:
        # Normalize positions to [0,1] range
        norm_x = (uav.position.x - self.grid.area_bounds[0]) / area_width
        norm_y = (uav.position.y - self.grid.area_bounds[2]) / area_height
        state_features.append(norm_x)
        state_features.append(norm_y)
    
    return state_features


def _get_region_load(self, region_id: int) -> float:
    """Calculate normalized load for a specific region."""
    if region_id not in self.regions:
        return 0.0
    
    # Get pending tasks for this region
    pending_tasks = len(self.task_manager.get_tasks_for_region(region_id, max_tasks=100))
    
    # Get vehicle count (task generators)
    vehicle_count = len(self.vehicle_manager.get_vehicles_in_region(region_id))
    
    # Get static UAV load
    static_uav = self.uav_manager.get_static_uav_by_region(region_id)
    static_load = 0.0
    if static_uav:
        static_load = static_uav.total_workload / static_uav.cpu_capacity
    
    # Calculate combined load metric
    task_density = pending_tasks / max(1, vehicle_count)  # tasks per vehicle
    combined_load = (task_density * 0.3) + (static_load * 0.7)  # weighted combination
    
    return min(1.0, combined_load)  # normalize to [0,1]


def _get_region_task_rate(self, region_id: int) -> float:
    """Get task arrival rate for a region."""
    if region_id not in self.regions:
        return 0.0
    
    # Get recent task generation rate (tasks per second)
    # This is a simplified metric based on pending tasks
    pending_tasks = len(self.task_manager.get_tasks_for_region(region_id, max_tasks=50))
    vehicle_count = len(self.vehicle_manager.get_vehicles_in_region(region_id))
    
    # Estimate task rate based on current activity
    base_rate = self.regions[region_id].base_intensity
    activity_multiplier = max(0.1, vehicle_count / 10.0)  # More vehicles = more tasks
    
    rate = base_rate * activity_multiplier
    return max(0.0, min(10.0, rate))  # Clamp to reasonable range [0, 10]


def _get_static_uav_queue(self, region_id: int) -> float:
    """Get static UAV queue length for a region."""
    static_uav = self.uav_manager.get_static_uav_by_region(region_id)
    if not static_uav:
        return 0.0
    
    queue_len = float(static_uav.queue_length)
    return max(0.0, min(100.0, queue_len))  # Clamp to reasonable range [0, 100]


def _get_static_uav_energy(self, region_id: int) -> float:
    """Get static UAV energy level for a region."""
    static_uav = self.uav_manager.get_static_uav_by_region(region_id)
    if not static_uav:
        return 1.0  # Default to full energy if no UAV
    
    # Check for valid energy values and return energy percentage (0.0 to 1.0)
    if (hasattr(static_uav, 'battery_capacity') and 
        hasattr(static_uav, 'current_energy') and 
        static_uav.battery_capacity > 0 and 
        not np.isnan(static_uav.current_energy) and 
        not np.isnan(static_uav.battery_capacity)):
        energy_ratio = static_uav.current_energy / static_uav.battery_capacity
        return max(0.0, min(1.0, energy_ratio))  # Clamp to [0,1]
    else:
        return 1.0  # Default to full energy if values are invalid


# Add methods to SAGINNetwork class
SAGINNetwork.set_heuristic_orchestration = set_heuristic_orchestration
SAGINNetwork.set_rl_orchestration = set_rl_orchestration
SAGINNetwork.get_task_offloading_decision_rl = get_task_offloading_decision_rl
SAGINNetwork._process_task_offloading_rl = _process_task_offloading_rl
SAGINNetwork._allocate_dynamic_uavs_with_rl = _allocate_dynamic_uavs_with_rl
SAGINNetwork._get_global_state_for_rl = _get_global_state_for_rl
SAGINNetwork._get_region_load = _get_region_load
SAGINNetwork._get_region_task_rate = _get_region_task_rate
SAGINNetwork._get_static_uav_queue = _get_static_uav_queue
SAGINNetwork._get_static_uav_energy = _get_static_uav_energy
