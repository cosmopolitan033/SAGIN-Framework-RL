"""
Reinforcement Learning environment for SAGIN system.

This module implements the RL environment that interfaces with the SAGIN network
to provide states, process actions, and calculate rewards as described in the paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..core.types import TaskDecision
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
    
    def __init__(self, network: SAGINNetwork, config: Dict[str, Any]):
        """
        Initialize the RL environment.
        
        Args:
            network: The SAGIN network instance
            config: Configuration parameters including reward weights
        """
        self.network = network
        self.config = config
        
        # Reward function weights
        self.alpha_1 = config.get('load_imbalance_weight', 0.5)
        self.alpha_2 = config.get('energy_penalty_weight', 1.0)
        
        # Minimum energy threshold from paper
        self.E_min = config.get('min_energy_threshold', 
                               self.network.system_params.min_energy_threshold)
        
        # Timing parameters
        self.central_action_interval = config.get('central_action_interval', 5)
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
        Get the local state for a static UAV in a given region as defined in the paper:
        s^local_{r,t} = (Q_r(t), E_{v^stat_r}(t), N^dyn_r(t), Λ_r(t))
        
        Args:
            region_id: ID of the region for which to get the local state
            
        Returns:
            Dictionary containing the local state components
        """
        network_state = self.network.get_network_state()
        
        # Find the static UAV for this region
        static_uav_id = None
        static_uav_info = None
        for uav_id, uav_info in network_state['uav_states'].get('static_uavs', {}).items():
            if uav_info.get('assigned_region_id') == region_id:
                static_uav_id = uav_id
                static_uav_info = uav_info
                break
        
        if not static_uav_info:
            return None  # No static UAV in this region
        
        # Current task queue
        task_queue = static_uav_info.get('task_queue', [])
        
        # Residual energy
        residual_energy = static_uav_info.get('energy_level', 0.0)
        
        # Number of available dynamic UAVs in this region
        available_dynamic_uavs = 0
        for uav_id, uav_info in network_state['uav_states'].get('dynamic_uavs', {}).items():
            if (uav_info.get('status') == 'available' and 
                uav_info.get('current_region_id') == region_id):
                available_dynamic_uavs += 1
        
        # Current task intensity in the region
        region_info = network_state['regions'].get(region_id, {})
        task_intensity = region_info.get('current_intensity', 0.0)
        
        return {
            'task_queue': task_queue,
            'queue_length': static_uav_info.get('queue_length', 0),
            'residual_energy': residual_energy,
            'available_dynamic_uavs': available_dynamic_uavs,
            'task_intensity': task_intensity,
            'region_id': region_id,
            'static_uav_id': static_uav_id
        }
    
    def process_central_action(self, action: Dict[int, int]) -> None:
        """
        Process actions from the central agent to allocate dynamic UAVs to regions.
        
        Args:
            action: Dictionary mapping dynamic UAV IDs to target region IDs
        """
        # Apply dynamic UAV allocations through the network
        for uav_id, target_region_id in action.items():
            self.network.uav_manager.assign_dynamic_uav_to_region(uav_id, target_region_id)
    
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
        Calculate the reward as defined in the paper:
        r_t = sum_j[I(T_total,j ≤ τ_j)] - α_1 * ΔL_t - α_2 * sum_v[I(E_v(t) < E_min)]
        
        Args:
            task_decisions: Dictionary mapping task IDs to success status
            load_imbalance: The load imbalance metric (if not provided, will be calculated)
            
        Returns:
            The calculated reward value
        """
        # Task completion reward
        task_completion_reward = sum(1 for success in task_decisions.values() if success)
        
        # Load imbalance penalty
        if load_imbalance is None:
            load_imbalance = self.network.metrics.load_imbalance
        load_imbalance_penalty = self.alpha_1 * load_imbalance
        
        # Energy threshold violations penalty
        energy_violations = 0
        for uav_id, uav in self.network.uav_manager.static_uavs.items():
            if uav.current_energy < self.E_min:
                energy_violations += 1
        
        for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
            if uav.current_energy < self.E_min:
                energy_violations += 1
        
        energy_penalty = self.alpha_2 * energy_violations
        
        # Total reward
        reward = task_completion_reward - load_imbalance_penalty - energy_penalty
        
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
        
        # Step the network forward in time
        step_results = self.network.step()
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
        self.network.reset_simulation()
        self.current_epoch = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        return self.get_global_state()
    
    def get_pending_tasks_by_region(self) -> Dict[int, List[str]]:
        """
        Get IDs of pending tasks by region for local agents to make decisions.
        
        Returns:
            Dictionary mapping region IDs to lists of task IDs
        """
        pending_tasks = {}
        
        for region_id in self.network.regions.keys():
            tasks = self.network.task_manager.get_tasks_for_region(region_id)
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
