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
    
    # Convert to numpy array for the agent
    state_array = np.array([
        state['queue_length'],
        state['residual_energy'],
        state['dynamic_uavs'],
        state['task_intensity'],
        state['task_size'],
        state['task_cycles'],
        state['task_urgency']
    ], dtype=np.float32)
    
    # Get action from agent
    local_agent = self.local_agents[region_id]
    action = local_agent.select_action(state_array)
    
    # Convert action index to task decision
    if action == 0:
        return TaskDecision.LOCAL
    elif action == 1:
        return TaskDecision.DYNAMIC
    else:
        return TaskDecision.SATELLITE


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
    
    # Get global state for central agent
    global_state = self._get_global_state_for_rl()
    
    # Convert to numpy array
    state_array = np.array(global_state, dtype=np.float32)
    
    # Get action from central agent
    actions = self.central_agent.select_action(state_array)
    
    # Apply actions: allocate dynamic UAVs to regions
    dynamic_uavs = list(self.uav_manager.dynamic_uavs.values())
    num_regions = len(self.regions)
    
    for i, uav in enumerate(dynamic_uavs):
        # Skip if action index is out of bounds
        if i >= len(actions):
            continue
            
        # Get target region from action
        target_region = int(actions[i]) + 1  # +1 because regions are 1-indexed
        
        # Validate region ID
        if target_region not in self.regions:
            target_region = 1  # Default to first region if invalid
        
        # Skip if already in this region
        if uav.assigned_region_id == target_region:
            continue
            
        # Assign to new region
        uav.assigned_region_id = target_region
        region = self.regions[target_region]
        
        # Move towards region center
        uav.set_target_position(region.center)
        uav.status = UAVStatus.MOVING


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


# Add methods to SAGINNetwork class
SAGINNetwork.set_heuristic_orchestration = set_heuristic_orchestration
SAGINNetwork.set_rl_orchestration = set_rl_orchestration
SAGINNetwork.get_task_offloading_decision_rl = get_task_offloading_decision_rl
SAGINNetwork._process_task_offloading_rl = _process_task_offloading_rl
SAGINNetwork._allocate_dynamic_uavs_with_rl = _allocate_dynamic_uavs_with_rl
SAGINNetwork._get_global_state_for_rl = _get_global_state_for_rl
