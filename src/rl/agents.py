"""
Reinforcement Learning agents for SAGIN system.

This module implements the central agent for dynamic UAV allocation and
local agents for task offloading decisions as described in the paper.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, namedtuple

from .models import ActorCriticNetwork, PolicyNetwork

Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class CentralAgent:
    """
    Central agent responsible for dynamic UAV allocation decisions.
    
    As described in the paper, this agent observes the global state and decides
    where to allocate dynamic UAVs based on regional demands.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize the central agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor-critic network for the central agent
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim=config.get('hidden_dim', 256))
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), 
                                   lr=config.get('learning_rate', 0.001))
        
        # Experience replay buffer
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 64)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.tau = config.get('tau', 0.005)     # Soft update parameter
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Create target network for stable learning
        self.target_network = ActorCriticNetwork(state_dim, action_dim, hidden_dim=config.get('hidden_dim', 256))
        self.target_network.to(self.device)
        self._update_target_network(tau=1.0)  # Hard update initially
        
        # Performance tracking
        self.training_losses = []
    
    def preprocess_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the global state into a format suitable for the neural network.
        
        Args:
            state: Dictionary containing the global state components
            
        Returns:
            Tensor representation of the state
        """
        # Extract all regions, UAVs, and create fixed-size vectors
        num_regions = len(state.get('regions', {}))
        num_dynamic_uavs = len(state.get('dynamic_uav_positions', {}))
        
        # Initialize state vectors with zeros
        region_features = np.zeros((num_regions, 3))  # [arrival_rate, queue_length, energy]
        uav_features = np.zeros((num_dynamic_uavs, 4))  # [available, x, y, z]
        
        # Fill region features
        for i, region_id in enumerate(sorted(state.get('regions', {}).keys())):
            region_features[i, 0] = state['task_arrival_rates'].get(region_id, 0.0)
            region_features[i, 1] = state['static_uav_queues'].get(region_id, 0.0)
            region_features[i, 2] = state['static_uav_energy'].get(region_id, 0.0)
        
        # Fill UAV features
        for i, uav_id in enumerate(sorted(state.get('dynamic_uav_positions', {}).keys())):
            uav_features[i, 0] = state['dynamic_uav_availability'].get(uav_id, 0)
            position = state['dynamic_uav_positions'].get(uav_id, (0, 0, 0))
            uav_features[i, 1:4] = position
        
        # Flatten and concatenate
        flat_state = np.concatenate([
            region_features.flatten(),
            uav_features.flatten(),
            [state.get('current_epoch', 0)]
        ])
        
        return torch.FloatTensor(flat_state).to(self.device)
    
    def select_action(self, state: Dict[str, Any], available_uavs: List[int], 
                    regions: List[int], explore: bool = True) -> Dict[int, int]:
        """
        Select dynamic UAV allocation actions based on the current state.
        
        Args:
            state: Dictionary containing the global state components
            available_uavs: List of available dynamic UAV IDs
            regions: List of region IDs
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Dictionary mapping UAV IDs to target region IDs
        """
        if not available_uavs:
            return {}  # No UAVs to allocate
            
        if explore and random.random() < self.epsilon:
            # Random exploration
            return {uav_id: random.choice(regions) for uav_id in available_uavs}
        
        # Preprocess state for the network
        state_tensor = self.preprocess_state(state)
        
        # Get policy distribution from the actor network
        with torch.no_grad():
            action_probs = self.network.actor(state_tensor.unsqueeze(0)).cpu().numpy()[0]
        
        # For each UAV, select a region based on the policy
        actions = {}
        reshaped_probs = action_probs.reshape(len(available_uavs), len(regions))
        
        for i, uav_id in enumerate(available_uavs):
            if i < reshaped_probs.shape[0]:
                # Select region index based on probability distribution
                region_idx = np.random.choice(len(regions), p=softmax(reshaped_probs[i]))
                actions[uav_id] = regions[region_idx]
        
        return actions
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.append(Experience(state, action, reward, next_state, done))
    
    def train(self):
        """Train the agent using experiences from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough experiences for training
        
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.cat([self.preprocess_state(e.state).unsqueeze(0) for e in batch])
        
        # Process actions (needs custom logic based on action representation)
        action_indices = []
        for e in batch:
            # Convert actions to indices in the action space
            # This depends on how actions are represented and action space is structured
            pass
        
        action_batch = torch.LongTensor(action_indices).to(self.device)
        reward_batch = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_state_batch = torch.cat([self.preprocess_state(e.next_state).unsqueeze(0) for e in batch])
        done_batch = torch.FloatTensor([float(e.done) for e in batch]).to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get values from target critic network
            next_values = self.target_network.critic(next_state_batch)
            target_values = reward_batch + self.gamma * next_values * (1 - done_batch)
        
        # Get current values from network
        values = self.network.critic(state_batch)
        
        # Compute value loss (MSE)
        value_loss = nn.MSELoss()(values, target_values)
        
        # Get actor loss
        policy_probs = self.network.actor(state_batch)
        log_probs = torch.log(policy_probs + 1e-10)  # Add small constant to avoid log(0)
        
        # Compute advantage (simple version)
        advantage = (target_values - values).detach()
        
        # Compute policy loss (using policy gradient)
        policy_loss = -torch.mean(torch.sum(log_probs * action_batch, dim=1) * advantage)
        
        # Total loss is weighted sum of value and policy losses
        total_loss = value_loss + policy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._update_target_network(self.tau)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track loss
        loss_value = total_loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def _update_target_network(self, tau: float):
        """
        Soft update of target network parameters.
        
        θ_target = τ*θ_local + (1-τ)*θ_target
        
        Args:
            tau: Interpolation parameter (1.0 for hard update)
        """
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.network.parameters()):
            target_param.data.copy_(tau * local_param.data + 
                                   (1.0 - tau) * target_param.data)
    
    def save(self, path: str):
        """Save agent model to disk."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent model from disk."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config']


class LocalAgent:
    """
    Local agent responsible for task offloading decisions at a static UAV.
    
    As described in the paper, each static UAV has its own local agent that
    decides how to offload incoming tasks based on local observations.
    """
    
    def __init__(self, region_id: int, state_dim: int, action_space: List[str], config: Dict[str, Any]):
        """
        Initialize the local agent.
        
        Args:
            region_id: ID of the region this agent is responsible for
            state_dim: Dimension of the local state space
            action_space: List of possible actions ('local', 'dynamic', 'satellite')
            config: Configuration parameters
        """
        self.region_id = region_id
        self.action_space = action_space
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Number of possible actions
        self.n_actions = len(action_space)
        
        # Create policy network for this local agent
        self.network = PolicyNetwork(state_dim, self.n_actions, hidden_dim=config.get('hidden_dim', 128))
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), 
                                   lr=config.get('learning_rate', 0.001))
        
        # Experience buffer (simpler than central agent, just a list)
        self.experiences = []
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        
        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Performance tracking
        self.training_losses = []
    
    def preprocess_state(self, state: Dict[str, Any], task_info: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess the local state and task info into a format suitable for the network.
        
        Args:
            state: Dictionary containing local state components
            task_info: Information about the task to be offloaded
            
        Returns:
            Tensor representation of the state-task combination
        """
        if state is None:
            return None
            
        # Extract state features
        queue_length = state.get('queue_length', 0)
        residual_energy = state.get('residual_energy', 0.0)
        available_dynamic_uavs = state.get('available_dynamic_uavs', 0)
        task_intensity = state.get('task_intensity', 0.0)
        
        # Extract task features
        data_size = task_info.get('data_size_in', 0.0)
        workload = task_info.get('workload', 0.0)
        deadline = task_info.get('deadline', 0.0)
        priority = task_info.get('priority', 1.0)
        
        # Combine into feature vector
        features = np.array([
            queue_length, 
            residual_energy,
            available_dynamic_uavs,
            task_intensity,
            data_size,
            workload,
            deadline,
            priority
        ], dtype=np.float32)
        
        return torch.FloatTensor(features).to(self.device)
    
    def select_action(self, state: Dict[str, Any], task_info: Dict[str, Any], 
                    explore: bool = True) -> str:
        """
        Select task offloading action for a specific task.
        
        Args:
            state: Dictionary containing local state components
            task_info: Information about the task to be offloaded
            explore: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Selected action ('local', 'dynamic', or 'satellite')
        """
        if explore and random.random() < self.epsilon:
            # Random exploration
            return random.choice(self.action_space)
        
        # Preprocess state and task info
        state_tensor = self.preprocess_state(state, task_info)
        if state_tensor is None:
            return self.action_space[0]  # Default to 'local' if state is invalid
        
        # Get action probabilities from network
        with torch.no_grad():
            action_probs = self.network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
        
        # Select action with highest probability
        action_idx = np.argmax(action_probs)
        
        return self.action_space[action_idx]
    
    def store_experience(self, state, task_info, action, reward, next_state, next_task_info, done):
        """
        Store experience for later training.
        
        Args:
            state: State at time of decision
            task_info: Task information at time of decision
            action: Action taken (offloading decision)
            reward: Reward received
            next_state: State after action
            next_task_info: Task information in next state
            done: Whether episode is done
        """
        action_idx = self.action_space.index(action)
        
        self.experiences.append({
            'state': state,
            'task_info': task_info,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state,
            'next_task_info': next_task_info,
            'done': done
        })
    
    def train(self, batch_size: int = 32):
        """
        Train the local agent using collected experiences.
        
        Args:
            batch_size: Number of experiences to use in one training step
            
        Returns:
            Loss value from training
        """
        if len(self.experiences) < batch_size:
            return 0.0  # Not enough experiences
        
        # Sample random batch
        batch = random.sample(self.experiences, batch_size)
        
        # Process batch
        state_tensors = []
        next_state_tensors = []
        action_indices = []
        rewards = []
        dones = []
        
        for exp in batch:
            state_tensor = self.preprocess_state(exp['state'], exp['task_info'])
            if exp['next_state'] is not None:
                next_state_tensor = self.preprocess_state(exp['next_state'], exp['next_task_info'])
            else:
                next_state_tensor = torch.zeros_like(state_tensor)
            
            state_tensors.append(state_tensor)
            next_state_tensors.append(next_state_tensor)
            action_indices.append(exp['action'])
            rewards.append(exp['reward'])
            dones.append(float(exp['done']))
        
        # Convert to tensors
        states = torch.stack(state_tensors)
        next_states = torch.stack(next_state_tensors)
        actions = torch.LongTensor(action_indices).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute loss using REINFORCE algorithm (policy gradient)
        self.optimizer.zero_grad()
        
        # Get log probabilities
        log_probs = torch.log(self.network(states))
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Simple reward-to-go estimation
        loss = -(selected_log_probs * rewards).mean()
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def save(self, path: str):
        """Save agent model to disk."""
        torch.save({
            'region_id': self.region_id,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'action_space': self.action_space
        }, path)
    
    def load(self, path: str):
        """Load agent model from disk."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config']
        self.action_space = checkpoint['action_space']
        assert self.region_id == checkpoint['region_id'], "Loading model for wrong region"


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
