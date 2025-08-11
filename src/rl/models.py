"""
Neural network models for reinforcement learning.

This module implements the neural network architectures used by the RL agents
in the SAGIN system, including actor-critic networks and policy networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any


class ActorNetwork(nn.Module):
    """
    Actor network for the central agent (outputs action probabilities).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic network for the central agent (outputs state values).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class QNetwork(nn.Module):
    """
    Q-Network for static UAV agents (DQN implementation).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class ActorCriticNetwork(nn.Module):
    """
    Legacy Actor-critic network (kept for backward compatibility).
    
    This network has two heads:
    - The actor head outputs a probability distribution over actions
    - The critic head outputs a value estimate for the current state
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim: Dimension of the state input
            action_dim: Dimension of the action output
            hidden_dim: Size of hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output value estimate
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        features = self.feature_extractor(state)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def actor(self, state):
        """
        Get action probabilities from the actor network.
        
        Args:
            state: State tensor
            
        Returns:
            Action probability distribution
        """
        features = self.feature_extractor(state)
        return self.actor_head(features)
    
    def critic(self, state):
        """
        Get state value from the critic network.
        
        Args:
            state: State tensor
            
        Returns:
            State value estimate
        """
        features = self.feature_extractor(state)
        return self.critic_head(features)


class PolicyNetwork(nn.Module):
    """
    Policy network used by the local agents.
    
    This network outputs a probability distribution over the discrete
    action space (local, dynamic, satellite).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state input
            action_dim: Number of discrete actions
            hidden_dim: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Action probability distribution
        """
        return self.model(state)
