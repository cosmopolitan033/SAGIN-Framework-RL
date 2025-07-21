"""
Reinforcement Learning components for SAGIN network optimization.

This module implements a hierarchical reinforcement learning approach
with a central agent controlling dynamic UAV allocation and multiple local agents
handling task offloading decisions, plus a shared agent for dynamic UAVs.
"""

from .agents import CentralAgent, LocalAgent, SharedStaticUAVAgent, SharedDynamicUAVAgent
from .environment import SAGINRLEnvironment
from .trainers import HierarchicalRLTrainer
from .rl_integration import RLModelManager

__all__ = [
    "CentralAgent",
    "LocalAgent", 
    "SharedStaticUAVAgent",
    "SharedDynamicUAVAgent",
    "SAGINRLEnvironment",
    "HierarchicalRLTrainer",
    "RLModelManager"
]
