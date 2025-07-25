"""
Reinforcement Learning environment for SAGIN system.

This module implements the RL environment that interfaces with the SAGIN network
to provide states, process actions, and calculate rewards as described in the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..core.types import TaskDecision, Position
from ..core.network import SAGINNetwork


class SAGINRLEnvironment:
    """
    RL environment for the SAGIN system that implements the MDP formulation from the paper.
    
    The environment handles:
    - State observation for both global and local agents
    - Action processing for dynamic UAV allocation and task offloading
    - Reward calculation based on task completion, load balancing, and energy constraints
    - Transition dynamics through the underlying SAGIN network
    """
    
    def __init__(self, sagin_network, config=None, alpha_1=None, alpha_2=None, episodes=1000, visualize=False):
        """
        Initialize the hierarchical SAGIN RL environment.
        
        Args:
            sagin_network: The SAGIN network instance
            config: Configuration dictionary (legacy) or None
            alpha_1: Load imbalance penalty weight (default: 0.1)
            alpha_2: Energy penalty weight (default: 0.0 - energy doesn't matter!)
            episodes: Number of training episodes
            visualize: Whether to enable visualization
        """
        self.network = sagin_network
        
        # Handle both old config pattern and new explicit parameters
        if config is not None:
            # Legacy config pattern
            self.alpha_1 = config.get('alpha_1', 0.1)
            self.alpha_2 = config.get('alpha_2', 0.0)  # ZERO energy penalty by default
        else:
            # New explicit parameter pattern
            self.alpha_1 = alpha_1 if alpha_1 is not None else 0.1
            self.alpha_2 = alpha_2 if alpha_2 is not None else 0.0  # ZERO energy penalty by default
        
        # Minimum energy threshold from paper
        self.E_min = getattr(sagin_network, 'min_energy_threshold', 0.1)
        
        # Timing parameters
        self.central_action_interval = 5
        self.current_epoch = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.total_reward = 0.0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def get_global_state(self) -> Dict[str, Any]:
        """
        Get the global state for the central agent as defined in the paper:
        s^global_t = ({λ_r(t)}, {L_{v^stat_r}(t)}, {E_{v^stat_r}(t)}, {A_n(t)}, {x_{v^dyn_n}(t)})
        
        Returns:
            Dictionary containing the global state components
        """
        network_state = self.network.get_network_state()
        
        # Task arrival rates per region
        task_arrival_rates = {}
        for region_id, stats in network_state['task_states'].items():
            task_arrival_rates[region_id] = stats.get('arrival_rate', 0.0)
        
        # Static UAV queue lengths and energy levels
        static_uav_queues = {}
        static_uav_energy = {}
        for uav_id, uav_info in network_state['uav_states'].get('static_uavs', {}).items():
            region_id = uav_info.get('assigned_region_id', -1)
            if region_id >= 0:
                static_uav_queues[region_id] = uav_info.get('queue_length', 0)
                static_uav_energy[region_id] = uav_info.get('energy_level', 0.0)
        
        # Dynamic UAV availability and positions
        dynamic_uav_availability = {}
        dynamic_uav_positions = {}
        for uav_id, uav_info in network_state['uav_states'].get('dynamic_uavs', {}).items():
            dynamic_uav_availability[uav_id] = int(uav_info.get('status') == 'available')
            dynamic_uav_positions[uav_id] = uav_info.get('position', (0, 0, 0))
        
        return {
            'task_arrival_rates': task_arrival_rates,
            'static_uav_queues': static_uav_queues,
            'static_uav_energy': static_uav_energy,
            'dynamic_uav_availability': dynamic_uav_availability, 
            'dynamic_uav_positions': dynamic_uav_positions,
            'current_epoch': self.current_epoch,
            'regions': network_state['regions']
        }
    
    def get_local_state(self, region_id: int) -> Dict[str, Any]:
        """
        Get the local state for a specific region (for SharedStaticUAVAgent).
        
        Args:
            region_id: The ID of the region
            
        Returns:
            Dictionary containing the local state for the region
        """
        network_state = self.network.get_network_state()
        
        # Static UAVs are nested under uav_states
        uav_states = network_state.get('uav_states', {})
        static_uavs = uav_states.get('static_uavs', {})
        
        # Find static UAV assigned to this region
        static_uav = static_uavs.get(region_id)
        
        if not static_uav:
            return None
        
        # Extract required information
        local_state = {
            'queue_length': static_uav.get('queue_length', 0),
            'residual_energy': static_uav.get('energy_percentage', 100.0) / 100.0,
            'available_dynamic_uavs': uav_states.get('total_dynamic_available', 0),
            'task_intensity': len(network_state.get('regions', {}).get(region_id, {}).get('tasks', []))
        }
        
        return local_state
    
    def process_central_action(self, action: Dict[int, int]) -> None:
        """
        Process central agent's dynamic UAV allocation action.
        
        Args:
            action: Dictionary mapping dynamic UAV IDs to target region IDs
        """
        # Apply dynamic UAV allocations through the network
        for uav_id, target_region_id in action.items():
            # Get region center for the target region
            row, col = self.network.grid.get_grid_position(target_region_id)
            center_x, center_y = self.network.grid.get_region_center(row, col)
            region_center = Position(center_x, center_y, self.network.system_params.dynamic_uav_altitude)
            current_time = self.network.current_time
            
            self.network.uav_manager.assign_dynamic_uav(
                uav_id, target_region_id, region_center, current_time
            )
    
    def process_local_action(self, region_id: int, task_id: str, decision: str) -> bool:
        """
        Process local agent's task offloading decision.
        
        Args:
            region_id: ID of the region where the decision is made
            task_id: ID of the task to be offloaded
            decision: Offloading decision ('local', 'dynamic', 'satellite')
            
        Returns:
            Boolean indicating whether the offloading was successful
        """
        # Convert string decision to TaskDecision enum
        decision_map = {
            'local': TaskDecision.LOCAL,
            'dynamic': TaskDecision.DYNAMIC, 
            'satellite': TaskDecision.SATELLITE
        }
        task_decision = decision_map.get(decision, TaskDecision.LOCAL)
        
        # Get the task object
        task = self.network.task_manager.get_task_by_id(task_id)
        if not task:
            return False
        
        # Execute offloading decision
        success = self.network._execute_task_decision(task, task_decision, region_id)
        return success
    
    def calculate_reward(self, task_decisions: Dict[str, bool], 
                        load_imbalance: float = None) -> float:
        """
        Calculate the reward as defined in the paper with COMPLETELY REBALANCED components:
        r_t = sum_j[I(T_total,j ≤ τ_j)] - α_1 * ΔL_t - α_2 * sum_v[I(E_v(t) < E_min)]
        
        Args:
            task_decisions: Dictionary mapping task IDs to success status
            load_imbalance: The load imbalance metric (if not provided, will be calculated)
            
        Returns:
            The calculated reward value (normalized to [-10, +10] range)
        """
        # Task completion reward (main positive signal) - NORMALIZED
        total_tasks = len(task_decisions)
        successful_tasks = sum(1 for success in task_decisions.values() if success)
        
        if total_tasks > 0:
            success_rate = successful_tasks / total_tasks
            # Normalize to [0, 5] range: high success = positive reward
            task_completion_reward = success_rate * 5.0
        else:
            # Neutral reward when no tasks (avoid artificial inflation)
            task_completion_reward = 0.0
        
        # Load imbalance penalty - NORMALIZED to [-2, 0] range
        if load_imbalance is None:
            load_imbalance = self.network.metrics.load_imbalance
        # Normalize load imbalance to reasonable penalty
        normalized_load_penalty = min(load_imbalance / 10.0, 2.0)  # Max 2 point penalty
        load_imbalance_penalty = self.alpha_1 * normalized_load_penalty
        
        # Energy violations penalty - NORMALIZED to [-2, 0] range
        energy_violations = 0
        total_uavs = 0
        
        for uav_id, uav in self.network.uav_manager.static_uavs.items():
            total_uavs += 1
            if uav.current_energy < self.E_min:
                energy_violations += 1
        
        for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
            total_uavs += 1
            if uav.current_energy < self.E_min:
                energy_violations += 1
        
        # Normalize energy penalty to max 2 points
        if total_uavs > 0:
            energy_violation_rate = energy_violations / total_uavs
            energy_penalty = self.alpha_2 * energy_violation_rate * 2.0  # Max 2 point penalty
        else:
            energy_penalty = 0
        
        # Network efficiency bonus - NORMALIZED to [0, 2] range
        efficiency_bonus = 0
        if hasattr(self.network.metrics, 'coverage_percentage'):
            coverage = getattr(self.network.metrics, 'coverage_percentage', 100)
            efficiency_bonus = (coverage / 100.0) * 2.0  # Max 2 bonus points
        
        # Latency bonus - NORMALIZED to [0, 1] range
        latency_bonus = 0
        if hasattr(self.network.metrics, 'average_latency'):
            avg_latency = getattr(self.network.metrics, 'average_latency', 0)
            if avg_latency > 0:
                # Normalize latency: good latency (<5) gets bonus
                latency_bonus = max(0, min(1.0, (5 - avg_latency) / 5.0))
        
        # BALANCED reward calculation
        # Positive components: task_completion (0-5) + efficiency (0-2) + latency (0-1) = [0, 8]
        # Negative components: load_penalty (0-2) + energy_penalty (0-2) = [0, 4]  
        # Final range: approximately [-4, +8] with positive bias for good performance
        reward = (task_completion_reward + efficiency_bonus + latency_bonus 
                 - load_imbalance_penalty - energy_penalty)
        
        # Debug info for training (can be removed later)
        if hasattr(self, '_debug_rewards') and self._debug_rewards:
            print(f"  Reward breakdown: Tasks={task_completion_reward:.1f}, "
                  f"Efficiency={efficiency_bonus:.1f}, Latency={latency_bonus:.1f}, "
                  f"LoadPenalty={load_imbalance_penalty:.1f}, EnergyPenalty={energy_penalty:.1f}, "
                  f"Total={reward:.1f}")
        
        # Track the reward
        self.total_reward += reward
        self.episode_rewards.append(reward)
        
        return reward
    
    def step(self, central_action: Dict[int, int] = None, 
             local_actions: Dict[int, Dict[str, str]] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment (one epoch).
        
        Args:
            central_action: Dictionary mapping dynamic UAV IDs to target regions
            local_actions: Dictionary mapping region IDs to {task_id: decision} dictionaries
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Process central action if it's time (every central_action_interval epochs)
        if central_action is not None and self.current_epoch % self.central_action_interval == 0:
            self.process_central_action(central_action)
        
        # Process local actions (task offloading decisions)
        task_decisions = {}
        if local_actions:
            for region_id, task_decisions_dict in local_actions.items():
                for task_id, decision in task_decisions_dict.items():
                    success = self.process_local_action(region_id, task_id, decision)
                    task_decisions[task_id] = success
        
        # Step the network forward in time (without processing tasks - RL controls that)
        step_results = self.network.step(process_tasks=False)
        self.current_epoch += 1
        
        # Calculate reward
        reward = self.calculate_reward(task_decisions)
        
        # Check completion status
        done = False  # In continuous operation, we'd set done based on specific criteria
        
        # Get next state
        next_global_state = self.get_global_state()
        
        # Prepare info dictionary
        info = {
            'completed_tasks': step_results.get('decisions_made', {}).get('local', 0) + 
                               step_results.get('decisions_made', {}).get('dynamic', 0) + 
                               step_results.get('decisions_made', {}).get('satellite', 0),
            'failed_tasks': step_results.get('decisions_made', {}).get('failed', 0),
            'load_imbalance': self.network.metrics.load_imbalance,
            'energy_violations': sum(1 for uav in self.network.uav_manager.static_uavs.values() 
                                    if uav.current_energy < self.E_min) +
                                sum(1 for uav in self.network.uav_manager.dynamic_uavs.values() 
                                    if uav.current_energy < self.E_min),
            'step_results': step_results
        }
        
        # Update performance tracking
        self.completed_tasks += info['completed_tasks']
        self.failed_tasks += info['failed_tasks']
        
        return next_global_state, reward, done, info
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial global state
        """
        # COMPLETE NETWORK RESET: Fixed to eliminate cyclic pattern
        self.network.reset_simulation()
        
        # Reset RL environment tracking (only need this since network does the rest)
        self.current_epoch = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        return self.get_global_state()
    
    def reset_episode_only(self) -> Dict[str, Any]:
        """
        Reset only episode-level tracking WITHOUT reinitializing the network.
        This is the KEY fix for eliminating cyclic patterns.
        
        Returns:
            Current global state
        """
        # CRITICAL: Do NOT call self.network.reset_simulation() here!
        # Network stays initialized, only reset RL episode tracking
        
        self.current_epoch = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Return current state without network reset
        return self.get_global_state()
    
    def get_pending_tasks_by_region(self) -> Dict[int, List[str]]:
        """
        Get IDs of pending tasks by region for local agents to make decisions.
        
        Returns:
            Dictionary mapping region IDs to lists of task IDs
        """
        pending_tasks = {}
        
        for region_id in self.network.regions.keys():
            tasks = self.network.task_manager.peek_tasks_for_region(region_id)
            pending_tasks[region_id] = [task.id for task in tasks]
        
        return pending_tasks
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the environment.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'total_reward': self.total_reward,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'task_success_rate': self.completed_tasks / (self.completed_tasks + self.failed_tasks) 
                                if (self.completed_tasks + self.failed_tasks) > 0 else 0.0,
            'load_imbalance': self.network.metrics.load_imbalance,
            'energy_consumption': self.network.metrics.energy_consumption,
            'average_latency': self.network.metrics.average_latency
        }
