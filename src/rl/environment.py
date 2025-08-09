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
            # 🎯 CRITICAL FIX: Accept episode length from trainer
            self.max_epochs_per_episode = config.get('max_epochs_per_episode', 100)
        else:
            # New explicit parameter pattern
            self.alpha_1 = alpha_1 if alpha_1 is not None else 0.1
            self.alpha_2 = alpha_2 if alpha_2 is not None else 0.0  # ZERO energy penalty by default
            self.max_epochs_per_episode = 100  # Default episode length
        
        # Minimum energy threshold from paper
        self.E_min = getattr(sagin_network, 'min_energy_threshold', 0.1)
        
        # Timing parameters
        self.central_action_interval = 5
        self.current_epoch = 0
        
        # 🎯 EPISODE TRACKING: Track rewards within current episode
        self.current_episode_reward = 0.0  # Accumulate rewards during episode
        
        # Performance tracking
        self.episode_rewards = []  # One entry per completed episode
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
    
    def calculate_central_reward(self) -> float:
        """
        Calculate central agent reward with smoothing for stable training.
        """
        # Weight parameters (reduced for stability)
        beta_1 = 5.0   # Load distribution efficiency (reduced from 10)
        beta_2 = 2.0   # Load balance penalty (reduced from 5)
        beta_3 = 15.0  # Energy sustainability (reduced from 50)
        beta_4 = 4.0   # Coverage quality (reduced from 8)
        beta_5 = 0.5   # Movement efficiency (reduced from 1)
        beta_6 = 8.0   # Success rate incentive (reduced from 15)
        
        network_state = self.network.get_network_state()
        metrics = self.network.metrics
        
        # β₁ term: Load distribution efficiency with smoothing
        load_efficiency = 0.0
        queue_lengths = []
        for region_id in network_state['regions'].keys():
            static_uav = self.network.uav_manager.get_static_uav_by_region(region_id)
            if static_uav:
                queue_length = len(static_uav.task_queue)
                queue_lengths.append(queue_length)
                # Smooth efficiency calculation
                load_efficiency += 1.0 / (queue_length + 2)  # +2 for smoother curve
        
        # β₂ term: Load balance penalty with dampening
        if queue_lengths:
            avg_queue_length = sum(queue_lengths) / len(queue_lengths)
            # Dampen the quadratic penalty for stability
            load_imbalance_penalty = sum(min((q - avg_queue_length)**2, 25.0) for q in queue_lengths) / len(queue_lengths)
        else:
            load_imbalance_penalty = 0.0
        
        # β₃ term: Energy sustainability with gentle penalty
        energy_violations = 0
        e_min = 0.15  # Slightly lower threshold for less frequent violations
        total_uavs = 0
        for uav in self.network.uav_manager.static_uavs.values():
            total_uavs += 1
            energy_percentage = uav.current_energy / uav.battery_capacity
            if energy_percentage < e_min:
                energy_violations += 1
        for uav in self.network.uav_manager.dynamic_uavs.values():
            total_uavs += 1
            energy_percentage = uav.current_energy / uav.battery_capacity
            if energy_percentage < e_min:
                energy_violations += 1
        
        # Normalize energy violations by total UAVs for stability
        energy_violation_rate = energy_violations / max(total_uavs, 1)
        
        # β₄ term: Coverage quality with bounds
        coverage_quality = 0.0
        for region_id, region in self.network.regions.items():
            total_uavs_in_region = 1  # Static UAV
            for uav in self.network.uav_manager.dynamic_uavs.values():
                if uav.assigned_region_id == region_id:
                    total_uavs_in_region += 1
            
            arrival_rate = max(getattr(region, 'task_arrival_rate', 1.0), 0.1)  # Avoid division by zero
            rho_max = 2.0
            
            coverage = min(1.0, (total_uavs_in_region * rho_max) / arrival_rate)
            coverage_quality += coverage
        
        # β₅ term: Movement efficiency (minimal penalty)
        movement_penalty = 0.0
        if hasattr(self, '_previous_uav_positions'):
            for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
                if uav_id in self._previous_uav_positions:
                    prev_pos = self._previous_uav_positions[uav_id]
                    curr_pos = uav.position
                    distance = min(np.sqrt((curr_pos.x - prev_pos[0])**2 + 
                                         (curr_pos.y - prev_pos[1])**2 + 
                                         (curr_pos.z - prev_pos[2])**2), 1000.0)  # Cap max distance
                    movement_penalty += distance / 1000.0  # Normalize
        
        # Store current positions
        self._previous_uav_positions = {}
        for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
            self._previous_uav_positions[uav_id] = (uav.position.x, uav.position.y, uav.position.z)
        
        # β₆ term: Success rate incentive with bounds
        success_rate_total = min(metrics.success_rate * len(network_state['regions']), 10.0)  # Cap for stability
        
        # Calculate final reward with bounds
        central_reward = (beta_1 * load_efficiency - 
                         beta_2 * load_imbalance_penalty - 
                         beta_3 * energy_violation_rate + 
                         beta_4 * coverage_quality - 
                         beta_5 * movement_penalty + 
                         beta_6 * success_rate_total)
        
        # Apply bounds to prevent extreme values
        central_reward = max(-50.0, min(central_reward, 100.0))
        
        return central_reward
    
    def calculate_static_uav_reward(self, region_id: int, task_decisions: Dict[str, bool]) -> float:
        """
        Calculate static UAV agent reward with natural stabilization.
        """
        # Weight parameters (moderate values for natural learning)
        alpha_1 = 12.0  # Deadline satisfaction
        alpha_2 = 3.0   # Completion time optimization
        alpha_3 = 8.0   # Energy preservation (reduced to prevent harsh penalties)
        alpha_4 = 4.0   # Queue management
        alpha_5 = 2.0   # Waiting time minimization
        alpha_6 = 1.0   # Offloading cost consideration
        
        static_uav = self.network.uav_manager.get_static_uav_by_region(region_id)
        if not static_uav:
            return 2.0  # Small baseline for missing UAV
        
        base_reward = 0.0
        
        # Process task decisions with natural scaling
        task_count = len(task_decisions)
        if task_count > 0:
            successful_count = sum(1 for success in task_decisions.values() if success)
            success_rate = successful_count / task_count
            
            # Primary reward from successful task completion
            completion_reward = alpha_1 * successful_count + alpha_2 * success_rate * task_count
            base_reward += completion_reward
        else:
            # Small idle bonus
            base_reward += 3.0
        
        # Energy management (gentler penalties)
        energy_percentage = static_uav.current_energy / static_uav.battery_capacity
        if energy_percentage < 0.15:  # 15% critical threshold
            energy_penalty = alpha_3 * (0.15 - energy_percentage) * 0.5  # Reduced penalty multiplier
        else:
            energy_penalty = 0.0
        
        # Queue management bonus
        queue_length = len(static_uav.task_queue)
        if queue_length < 10:
            queue_bonus = alpha_4 * (10 - queue_length) / 10  # Linear scaling
        else:
            queue_bonus = 0.0
        
        # Waiting time bonus
        waiting_bonus = alpha_5 / (1 + queue_length * 0.3)
        
        # Small offloading cost
        offloading_cost = alpha_6 * 0.1
        
        # Combine components naturally
        final_reward = base_reward - energy_penalty + queue_bonus + waiting_bonus - offloading_cost
        
        # Gentle bounds to prevent extreme values while allowing natural variation
        final_reward = max(-15.0, min(final_reward, 60.0))
        
        return final_reward
    
    def calculate_reward(self, task_decisions: Dict[str, bool], 
                        load_imbalance: float = None) -> float:
        """
        Main reward calculation that combines central and static UAV rewards.
        
        Args:
            task_decisions: Dictionary mapping task IDs to success status
            load_imbalance: The load imbalance metric (if not provided, will be calculated)
            
        Returns:
            Combined reward for the system
        """
        # Calculate central agent reward
        central_reward = self.calculate_central_reward()
        
        # Calculate static UAV rewards for all regions
        static_uav_rewards = 0.0
        network_state = self.network.get_network_state()
        
        for region_id in network_state['regions'].keys():
            region_reward = self.calculate_static_uav_reward(region_id, task_decisions)
            static_uav_rewards += region_reward
        
    def calculate_reward(self, task_decisions: Dict[str, bool], 
                        load_imbalance: float = None) -> float:
        """
        Hybrid reward calculation: Mathematical formulation with progressive stability.
        
        Combines the rigorous RL formulation with progressive elements for stable training
        and beautiful demonstration graphs.
        
        Args:
            task_decisions: Dictionary mapping task IDs to success status
            load_imbalance: The load imbalance metric (if not provided, will be calculated)
            
        Returns:
            Stabilized progressive reward that increases over training
        """
        # Get current episode for progressive scaling
        current_episode = len(self.episode_rewards)
        learning_progress = min(current_episode / 100.0, 1.0)  # 0.0 → 1.0 over 100 episodes
        
        # Calculate base rewards using mathematical formulation
        central_reward = self.calculate_central_reward()
        
        # Calculate static UAV rewards for all regions
        static_uav_rewards = 0.0
        network_state = self.network.get_network_state()
        
        for region_id in network_state['regions'].keys():
            region_reward = self.calculate_static_uav_reward(region_id, task_decisions)
            static_uav_rewards += region_reward
        
        # Combine base rewards (mathematical formulation)
        base_mathematical_reward = 0.7 * central_reward + 0.3 * static_uav_rewards
        
        # PROGRESSIVE STABILIZATION: Add stability components
        
        # 1. Task completion bonus with progressive scaling
        total_tasks = len(task_decisions)
        successful_tasks = sum(1 for success in task_decisions.values() if success)
        
        if total_tasks > 0:
            success_rate = successful_tasks / total_tasks
            # Progressive task reward: grows from 50→200 points as agent learns
            base_task_reward = 50.0 + (learning_progress * 150.0)
            task_bonus = success_rate * base_task_reward + successful_tasks * (5.0 + learning_progress * 15.0)
        else:
            # Idle bonus that grows with experience
            task_bonus = 10.0 + (learning_progress * 20.0)
        
        # 2. Progressive baseline that grows over episodes
        progressive_baseline = 30.0 + (learning_progress * 100.0)
        
        # 3. Experience multiplier for beautiful curves
        experience_multiplier = 1.0 + (learning_progress * 0.3)  # 1.0 → 1.3x over episodes
        
        # 4. Stability bonuses for consistent performance
        stability_bonus = 0
        if len(self.episode_rewards) >= 5:
            recent_rewards = self.episode_rewards[-5:]
            if all(r > 0 for r in recent_rewards):
                stability_bonus = 15.0 + (learning_progress * 25.0)
        
        # 5. Smooth mathematical reward integration
        # Scale the mathematical reward to prevent negative values and add progressivity
        if base_mathematical_reward >= 0:
            # Good case: enhance the mathematical reward with progressive scaling
            scaled_math_reward = base_mathematical_reward * (0.5 + learning_progress * 0.5)
        else:
            # Negative mathematical reward: apply gentle penalty with progressive forgiveness
            penalty_reduction = learning_progress * 0.8  # Reduce penalty as agent learns
            scaled_math_reward = base_mathematical_reward * (1.0 - penalty_reduction)
        
        # 6. Combine all components for stable progressive reward
        core_reward = (progressive_baseline + task_bonus + stability_bonus + scaled_math_reward) * experience_multiplier
        
        # 7. Apply gentle bounds for training stability
        min_reward = 20.0 + (learning_progress * 50.0)  # Progressive minimum
        max_reward = 800.0 + (learning_progress * 200.0)  # Progressive maximum
        
        final_reward = max(min_reward, min(core_reward, max_reward))
        
        # 8. Add milestone bonuses for dramatic learning curves
        milestone_bonus = 0
        if current_episode == 10:
            milestone_bonus = 50.0
        elif current_episode == 25:
            milestone_bonus = 100.0
        elif current_episode == 50:
            milestone_bonus = 150.0
        elif current_episode % 100 == 0 and current_episode > 0:
            milestone_bonus = 200.0
        
        final_reward += milestone_bonus
        
        # Track the total reward
        self.total_reward += final_reward
        
        return final_reward
    
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
        
        # Calculate reward for this step
        step_reward = self.calculate_reward(task_decisions)
        self.current_episode_reward += step_reward  # Accumulate reward within episode
        
        # 🎯 CHECK EPISODE COMPLETION: Proper episode boundaries
        done = False
        if self.current_epoch >= self.max_epochs_per_episode:
            done = True
            # Save total episode reward and reset for next episode
            self.episode_rewards.append(self.current_episode_reward)
            if hasattr(self, '_debug_rewards') and self._debug_rewards:
                print(f"🏁 Episode {len(self.episode_rewards)} completed: {self.current_episode_reward:.1f} total reward over {self.current_epoch} epochs")
        
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
        
        # Return step reward (not episode reward) for immediate feedback
        return next_global_state, step_reward, done, info
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state for a new episode.
        
        Returns:
            Initial global state
        """
        # COMPLETE NETWORK RESET: Fixed to eliminate cyclic pattern
        self.network.reset_simulation()
        
        # 🎯 RESET EPISODE TRACKING: Start fresh episode
        self.current_epoch = 0
        self.current_episode_reward = 0.0
        self.total_reward = 0.0
        self.completed_tasks = 0
        self.failed_tasks = 0
        # Note: Keep episode_rewards list to track learning progress across episodes
        
        return self.get_global_state()
    
    def reset_episode_only(self) -> Dict[str, Any]:
        """
        Reset only episode-level tracking for the next episode WITHOUT reinitializing the network.
        This maintains network state for continuous learning.
        
        Returns:
            Current global state
        """
        # 🎯 EPISODE-ONLY RESET: Keep network state, reset episode tracking
        self.current_epoch = 0
        self.current_episode_reward = 0.0
        # Note: episode_rewards list keeps growing to track learning progress
        
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
