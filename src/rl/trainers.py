"""
Trainers for the hierarchical reinforcement learning system.

This module implements trainers that coordinate the central and local agents
in the hierarchical RL approach described in the paper.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

from ..core.network import SAGINNetwork
from .environment import SAGINRLEnvironment
from .agents import CentralAgent, LocalAgent, SharedStaticUAVAgent, SharedDynamicUAVAgent


class HierarchicalRLTrainer:
    """
    Trainer for the hierarchical RL system with central and local agents.
    
    This class coordinates training of the central agent (dynamic UAV allocation)
    and the local agents (task offloading decisions) in a hierarchical manner.
    """
    
    def __init__(self, network: SAGINNetwork, config: Dict[str, Any]):
        """
        Initialize the hierarchical RL trainer.
        
        Args:
            network: The SAGIN network instance
            config: Configuration parameters
        """
        self.config = config
        
        # Create environment
        self.env = SAGINRLEnvironment(network, config.get('env_config', {}))
        
        # Create central agent
        central_state_dim = self._estimate_central_state_dim(network)
        central_action_dim = self._estimate_central_action_dim(network)
        
        self.central_agent = CentralAgent(
            central_state_dim,
            central_action_dim,
            config.get('central_agent_config', {})
        )
        
        # Create shared static UAV agent (replaces per-region local agents)
        # According to the paper: all static UAVs share one RL model
        self.action_space = ['local', 'dynamic', 'satellite']
        local_state_dim = self._estimate_local_state_dim()
        
        self.shared_static_uav_agent = SharedStaticUAVAgent(
            local_state_dim,
            self.action_space,
            config.get('static_uav_agent_config', {})
        )
        
        # Note: Dynamic UAVs do NOT need RL agents (they only compute, no offloading decisions)
        # All static UAVs now share the single SharedStaticUAVAgent
        
        # Training parameters
        self.num_episodes = config.get('num_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 100)
        self.central_update_frequency = config.get('central_update_frequency', 5)
        
        # Performance tracking
        self.rewards_history = []
        self.central_losses = []
        self.static_uav_losses = []  # For shared static UAV agent
        
        # Results directory
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _estimate_central_state_dim(self, network: SAGINNetwork) -> int:
        """
        Estimate the dimension of the central agent's state space.
        
        Args:
            network: The SAGIN network instance
            
        Returns:
            The estimated state dimension
        """
        # Global state includes features for each region and UAV
        num_regions = len(network.regions)
        num_dynamic_uavs = len(network.uav_manager.dynamic_uavs)
        
        # For each region: task_rate, queue_length, energy (3 features)
        # For each UAV: availability, x, y, z (4 features)
        # Plus the current epoch
        return num_regions * 3 + num_dynamic_uavs * 4 + 1
    
    def _estimate_central_action_dim(self, network: SAGINNetwork) -> int:
        """
        Estimate the dimension of the central agent's action space.
        
        Args:
            network: The SAGIN network instance
            
        Returns:
            The estimated action dimension
        """
        # Each dynamic UAV can be assigned to any region
        num_dynamic_uavs = len(network.uav_manager.dynamic_uavs)
        num_regions = len(network.regions)
        
        return num_dynamic_uavs * num_regions
    
    def _estimate_local_state_dim(self) -> int:
        """
        Estimate the dimension of the local agent's state space.
        
        Returns:
            The estimated state dimension
        """
        # State includes:
        # - Queue length
        # - Residual energy
        # - Number of available dynamic UAVs
        # - Task intensity
        # - Task features: data_size, workload, deadline, priority
        return 8
    
    def _estimate_dynamic_uav_state_dim(self) -> int:
        """
        Estimate the dimension of the dynamic UAV agent's state space.
        
        Returns:
            The estimated state dimension
        """
        # State includes:
        # - UAV state: queue_length, residual_energy, cpu_utilization, current_region (4 features)
        # - Task info: urgency, complexity, deadline, type_encoding (4 features)
        return 8
    
    def train(self, verbose: bool = True):
        """
        Train the hierarchical RL system.
        
        Args:
            verbose: Whether to print training progress
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting training for {self.num_episodes} episodes...")
            
        for episode in range(1, self.num_episodes + 1):
            episode_reward = 0
            state = self.env.reset()
            
            for step in range(1, self.max_steps_per_episode + 1):
                # Determine if central action should be taken this step
                take_central_action = (step % self.central_update_frequency == 0)
                
                # Get available dynamic UAVs and regions
                network_state = self.env.network.get_network_state()
                available_uavs = [
                    uav_id for uav_id, info in 
                    network_state['uav_states'].get('dynamic_uavs', {}).items()
                    if info.get('status') == 'available'
                ]
                regions = list(network_state['regions'].keys())
                
                # Select central action if it's time
                central_action = None
                if take_central_action:
                    central_action = self.central_agent.select_action(
                        state, available_uavs, regions
                    )
                
                # Get pending tasks by region
                pending_tasks = self.env.get_pending_tasks_by_region()
                
                # Select local actions for each region with pending tasks
                local_actions = {}
                local_task_info = {}
                
                for region_id, task_ids in pending_tasks.items():
                    if task_ids:  # Use shared static UAV agent for all regions
                        local_state = self.env.get_local_state(region_id)
                        local_actions[region_id] = {}
                        local_task_info[region_id] = {}
                        
                        for task_id in task_ids:
                            # Get task info
                            task = self.env.network.task_manager.get_task_by_id(task_id)
                            if task:
                                task_info = {
                                    'data_size_in': task.data_size_in,
                                    'workload': task.cpu_cycles,
                                    'deadline': task.deadline,
                                    'priority': 1.0  # Default priority since Task doesn't have this attribute
                                }
                                
                                # Use shared static UAV agent for action selection
                                action = self.shared_static_uav_agent.select_action(
                                    local_state, task_info
                                )
                                
                                local_actions[region_id][task_id] = action
                                local_task_info[region_id][task_id] = task_info
                
                # Take step in environment with both central and local actions
                next_state, reward, done, info = self.env.step(central_action, local_actions)
                episode_reward += reward
                
                # Store experience for central agent if action was taken
                if take_central_action and central_action:
                    self.central_agent.store_experience(
                        state, central_action, reward, next_state, done
                    )
                
                # Store experiences for shared static UAV agent
                for region_id, task_dict in local_actions.items():
                    local_state = self.env.get_local_state(region_id)
                    next_local_state = self.env.get_local_state(region_id)
                    
                    for task_id, action in task_dict.items():
                        if task_id in local_task_info[region_id]:
                            task_info = local_task_info[region_id][task_id]
                            # Store experience in shared static UAV agent
                            self.shared_static_uav_agent.store_experience(
                                local_state, task_info, action, reward, 
                                next_local_state, task_info, done
                            )
                
                # Update state
                state = next_state
                
                # Break if episode is done
                if done:
                    break
            
            # Train central agent
            central_loss = self.central_agent.train()
            self.central_losses.append(central_loss)
            
            # Train shared static UAV agent
            static_uav_loss = self.shared_static_uav_agent.train()
            self.static_uav_losses.append(static_uav_loss)
            
            # Record episode reward
            self.rewards_history.append(episode_reward)
            
            # Print progress
            if verbose and (episode % 10 == 0 or episode == 1):
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.rewards_history[-10:]) if episode > 10 else episode_reward
                
                print(f"Episode {episode}/{self.num_episodes} [{elapsed:.1f}s] - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Central Loss: {central_loss:.4f}")
            
            # Save checkpoints periodically
            if episode % 100 == 0:
                self.save_checkpoints(episode)
                
        # Final checkpoint
        self.save_checkpoints(self.num_episodes, final=True)
        
        # Plot and save results
        self.plot_results()
        
        if verbose:
            print(f"Training completed in {time.time() - start_time:.1f} seconds")
            print(f"Final average reward: {np.mean(self.rewards_history[-100:]):.2f}")
        
        # Get final network performance metrics
        final_metrics = self.env.network.metrics
        
        # Return the trained agents and training statistics
        training_stats = {
            'rewards_history': self.rewards_history,
            'central_losses': self.central_losses,
            'static_uav_losses': self.static_uav_losses,
            'final_average_reward': np.mean(self.rewards_history[-100:]) if self.rewards_history else 0.0,
            'total_episodes': self.num_episodes,
            'training_time': time.time() - start_time,
            # Add final network performance metrics
            'success_rate': final_metrics.success_rate,
            'avg_latency': final_metrics.average_latency
        }
        
        return self.central_agent, self.shared_static_uav_agent, training_stats
    
    def save_checkpoints(self, episode: int, final: bool = False):
        """
        Save agent checkpoints.
        
        Args:
            episode: Current episode number
            final: Whether this is the final checkpoint
        """
        checkpoint_dir = os.path.join(self.results_dir, f"checkpoint_ep{episode}")
        if final:
            checkpoint_dir = os.path.join(self.results_dir, "final_model")
            
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save central agent
        self.central_agent.save(os.path.join(checkpoint_dir, "central_agent.pt"))
        
        # Save shared static UAV agent
        self.shared_static_uav_agent.save(os.path.join(checkpoint_dir, "shared_static_uav_agent.pt"))
        
        # Save training history
        with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
            json.dump({
                'rewards': self.rewards_history,
                'central_losses': self.central_losses,
                'static_uav_losses': self.static_uav_losses
            }, f)
    
    def plot_results(self):
        """Plot and save training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot rewards
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards_history)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"rewards_{timestamp}.png"))
        
        # Plot central losses
        plt.figure(figsize=(12, 6))
        plt.plot(self.central_losses)
        plt.title('Central Agent Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"central_losses_{timestamp}.png"))
        
        # Plot static UAV agent losses
        plt.figure(figsize=(12, 6))
        plt.plot(self.static_uav_losses)
        plt.title('Shared Static UAV Agent Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"static_uav_losses_{timestamp}.png"))
    
    def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict[str, float]:
        """
        Evaluate the trained agents.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render/print episode details
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        completed_tasks = []
        failed_tasks = []
        avg_latencies = []
        load_imbalances = []
        energy_consumptions = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                # Get available dynamic UAVs and regions
                network_state = self.env.network.get_network_state()
                available_uavs = [
                    uav_id for uav_id, info in 
                    network_state['uav_states'].get('dynamic_uavs', {}).items()
                    if info.get('status') == 'available'
                ]
                regions = list(network_state['regions'].keys())
                
                # Select central action (only every central_update_frequency steps)
                central_action = None
                if step % self.central_update_frequency == 0:
                    central_action = self.central_agent.select_action(
                        state, available_uavs, regions, explore=False
                    )
                
                # Get pending tasks by region
                pending_tasks = self.env.get_pending_tasks_by_region()
                
                # Select local actions for each region with pending tasks
                local_actions = {}
                
                for region_id, task_ids in pending_tasks.items():
                    if task_ids:  # Use shared static UAV agent for all regions
                        local_state = self.env.get_local_state(region_id)
                        local_actions[region_id] = {}
                        
                        for task_id in task_ids:
                            # Get task info
                            task = self.env.network.task_manager.get_task_by_id(task_id)
                            if task:
                                task_info = {
                                    'data_size_in': task.data_size_in,
                                    'workload': task.cpu_cycles,
                                    'deadline': task.deadline,
                                    'priority': 1.0  # Default priority since Task doesn't have this attribute
                                }
                                
                                # Select action without exploration using shared agent
                                action = self.shared_static_uav_agent.select_action(
                                    local_state, task_info, explore=False
                                )
                                
                                local_actions[region_id][task_id] = action
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(central_action, local_actions)
                episode_reward += reward
                
                # Update state
                state = next_state
                
                # Break if episode is done
                if done:
                    break
                
                # Print step info if rendering
                if render and step % 10 == 0:
                    print(f"Episode {episode+1}, Step {step} - "
                          f"Reward: {reward:.2f}, "
                          f"Tasks: {info['completed_tasks']}/{info['completed_tasks'] + info['failed_tasks']}")
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            metrics = self.env.get_performance_metrics()
            completed_tasks.append(metrics['completed_tasks'])
            failed_tasks.append(metrics['failed_tasks'])
            avg_latencies.append(metrics['average_latency'])
            load_imbalances.append(metrics['load_imbalance'])
            energy_consumptions.append(metrics['energy_consumption'])
            
            if render:
                print(f"Episode {episode+1} - "
                      f"Reward: {episode_reward:.2f}, "
                      f"Task Success Rate: {metrics['task_success_rate']:.3f}, "
                      f"Avg Latency: {metrics['average_latency']:.3f}s")
        
        # Calculate overall metrics
        total_completed = sum(completed_tasks)
        total_tasks = total_completed + sum(failed_tasks)
        success_rate = total_completed / total_tasks if total_tasks > 0 else 0
        
        evaluation_metrics = {
            'avg_reward': np.mean(episode_rewards),
            'success_rate': success_rate,
            'avg_latency': np.mean(avg_latencies),
            'avg_load_imbalance': np.mean(load_imbalances),
            'avg_energy_consumption': np.mean(energy_consumptions),
            'completed_tasks': total_completed,
            'failed_tasks': sum(failed_tasks)
        }
        
        if render:
            print("\nEvaluation Results:")
            for key, value in evaluation_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        return evaluation_metrics
