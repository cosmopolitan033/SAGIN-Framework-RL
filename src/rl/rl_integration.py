"""
Enhanced RL Integration for SAGIN System
========================================

This module provides comprehensive RL integration with:
- Model training, saving, and loading
- Interactive model selection
- Integration with demo system
- Proper state/reward formulation per the paper
"""

import os
import torch
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np

from .agents import CentralAgent, LocalAgent, SharedStaticUAVAgent
from .trainers import HierarchicalRLTrainer
from .environment import SAGINRLEnvironment


class RLModelManager:
    """Manages RL model training, saving, loading, and selection."""
    
    def __init__(self, models_dir: str = "models/rl"):
        """Initialize the RL model manager.
        
        Args:
            models_dir: Directory to store trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Model registry file
        self.registry_file = os.path.join(models_dir, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                    
                # Validate registry structure
                if not isinstance(registry, dict):
                    print(f"⚠️ Warning: Model registry is not a valid dictionary, creating new one")
                    return {}
                    
                # Filter out invalid entries
                valid_registry = {}
                for name, info in registry.items():
                    if isinstance(info, dict):
                        valid_registry[name] = info
                    else:
                        print(f"⚠️ Warning: Skipping invalid entry '{name}' in model registry")
                        
                return valid_registry
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Model registry JSON is corrupted ({e}), creating new one")
                return {}
            except Exception as e:
                print(f"⚠️ Warning: Error loading model registry ({e}), creating new one")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the model registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, model_info: Dict[str, Any]):
        """Register a trained model.
        
        Args:
            model_name: Unique name for the model
            model_info: Model metadata (config, performance, etc.)
        """
        timestamp = datetime.now().isoformat()
        self.registry[model_name] = {
            **model_info,
            'timestamp': timestamp,
            'model_path': os.path.join(self.models_dir, f"{model_name}")
        }
        self._save_registry()
        print(f"✅ Registered model: {model_name}")
    
    def save_model(self, model_name: str, central_agent: CentralAgent, 
                   shared_static_uav_agent: SharedStaticUAVAgent,
                   model_info: Dict[str, Any]):
        """Save trained RL models.
        
        Args:
            model_name: Unique name for this model
            central_agent: Trained central agent
            shared_static_uav_agent: Shared agent for all static UAVs
            model_info: Model metadata and performance metrics
        """
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save central agent
        central_path = os.path.join(model_path, "central_agent.pth")
        torch.save({
            'state_dict': central_agent.network.state_dict(),
            'config': central_agent.config,
            'training_losses': central_agent.training_losses
        }, central_path)
        
        # Save shared static UAV agent
        shared_static_path = os.path.join(model_path, "shared_static_uav_agent.pth")
        torch.save({
            'state_dict': shared_static_uav_agent.network.state_dict(),
            'config': shared_static_uav_agent.config,
            'training_losses': shared_static_uav_agent.training_losses,
            'usage_count': shared_static_uav_agent.usage_count
        }, shared_static_path)
        
        # Save metadata
        metadata = {
            **model_info,
            'architecture': 'central_plus_shared_static',
            'central_state_dim': central_agent.network.feature_extractor[0].in_features,
            'central_action_dim': central_agent.network.actor_head[-2].out_features,
            'static_uav_usage_count': shared_static_uav_agent.usage_count
        }
        
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Register the model
        self.register_model(model_name, metadata)
        print(f"💾 Saved model: {model_name} to {model_path}")
    
    def load_model(self, model_name: str, network) -> Tuple[CentralAgent, SharedStaticUAVAgent]:
        """Load trained RL models.
        
        Args:
            model_name: Name of the model to load
            network: SAGIN network instance (for config inference)
            
        Returns:
            Tuple of (central_agent, shared_static_uav_agent)
        """
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = self.registry[model_name]['model_path']
        
        # Load metadata
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load central agent
        central_path = os.path.join(model_path, "central_agent.pth")
        central_checkpoint = torch.load(central_path, map_location='cpu')
        
        central_agent = CentralAgent(
            state_dim=metadata['central_state_dim'],
            action_dim=metadata['central_action_dim'],
            config=central_checkpoint['config']
        )
        central_agent.network.load_state_dict(central_checkpoint['state_dict'])
        central_agent.training_losses = central_checkpoint['training_losses']
        
        # Load shared static UAV agent
        shared_static_path = os.path.join(model_path, "shared_static_uav_agent.pth")
        shared_static_checkpoint = torch.load(shared_static_path, map_location='cpu')
        
        # Infer state dim from saved model
        shared_state_dim = shared_static_checkpoint['state_dict']['model.0.weight'].shape[1]
        
        shared_static_uav_agent = SharedStaticUAVAgent(
            state_dim=shared_state_dim,
            action_space=['local', 'dynamic', 'satellite'],
            config=shared_static_checkpoint['config']
        )
        shared_static_uav_agent.network.load_state_dict(shared_static_checkpoint['state_dict'])
        shared_static_uav_agent.training_losses = shared_static_checkpoint['training_losses']
        shared_static_uav_agent.usage_count = shared_static_checkpoint.get('usage_count', 0)
        
        print(f"📂 Loaded model: {model_name}")
        return central_agent, shared_static_uav_agent
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available trained models."""
        models = []
        for name, info in self.registry.items():
            try:
                # Ensure info is a dictionary, not a string or other type
                if not isinstance(info, dict):
                    print(f"⚠️ Warning: Skipping malformed entry '{name}' in model registry")
                    continue
                    
                models.append({
                    'name': name,
                    'description': info.get('description', 'No description'),
                    'timestamp': info.get('timestamp', 'Unknown'),
                    'performance': info.get('performance', {}),
                    'config': info.get('config', {})
                })
            except Exception as e:
                print(f"⚠️ Warning: Error processing model '{name}': {e}")
                continue
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def interactive_model_selection(self) -> Optional[str]:
        """Interactive CLI for model selection.
        
        Returns:
            Selected model name or None if cancelled
        """
        models = self.list_models()
        
        if not models:
            print("❌ No trained models available")
            return None
        
        print("\n🤖 Available RL Models:")
        print("=" * 60)
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['name']}")
            print(f"     Description: {model['description']}")
            print(f"     Trained: {model['timestamp'][:19]}")
            if 'success_rate' in model['performance']:
                print(f"     Success Rate: {model['performance']['success_rate']:.2%}")
            if 'avg_latency' in model['performance']:
                print(f"     Avg Latency: {model['performance']['avg_latency']:.3f}s")
            print()
        
        print(f"{len(models)+1:2d}. Use Heuristic (Original) Method")
        print(f"{len(models)+2:2d}. Cancel")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(models)+2}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]['name']
                    print(f"✅ Selected RL model: {selected_model}")
                    return selected_model
                elif choice_num == len(models) + 1:
                    print("✅ Selected: Heuristic method")
                    return "heuristic"
                elif choice_num == len(models) + 2:
                    print("❌ Cancelled")
                    return None
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(models)+2}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")


class EnhancedSAGINRLEnvironment(SAGINRLEnvironment):
    """Enhanced RL environment with proper state/reward formulation per the paper."""
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get global state for central agent as per paper formulation.
        
        Returns:
            Global state s^global_t with:
            - λ_r(t): task arrival rates per region
            - L_v^stat_r(t): queue lengths at static UAVs  
            - E_v^stat_r(t): residual energy of static UAVs
            - A_n(t): availability of dynamic UAVs
            - x_v^dyn_n(t): positions of dynamic UAVs
        """
        state = {}
        
        # Task arrival rates per region
        arrival_rates = {}
        for region_id in self.network.regions.keys():
            arrival_rates[region_id] = self.network.task_manager.get_region_task_rate(region_id)
        state['task_arrival_rates'] = arrival_rates
        
        # Static UAV queue lengths and energy
        static_queues = {}
        static_energy = {}
        for region_id in self.network.regions.keys():
            static_uav = self.network.uav_manager.get_static_uav_by_region(region_id)
            if static_uav:
                static_queues[region_id] = static_uav.queue_length
                static_energy[region_id] = static_uav.current_energy / static_uav.battery_capacity
            else:
                static_queues[region_id] = 0
                static_energy[region_id] = 0
        
        state['static_uav_queues'] = static_queues
        state['static_uav_energy'] = static_energy
        
        # Dynamic UAV availability and positions
        dynamic_availability = {}
        dynamic_positions = {}
        for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
            dynamic_availability[uav_id] = 1 if uav.is_available else 0
            dynamic_positions[uav_id] = (uav.position.x, uav.position.y, uav.position.z)
        
        state['dynamic_uav_availability'] = dynamic_availability
        state['dynamic_uav_positions'] = dynamic_positions
        
        # Current epoch
        state['current_epoch'] = self.network.current_epoch
        
        return state
    
    def get_local_state(self, region_id: int, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get local state for static UAV as per paper formulation.
        
        Args:
            region_id: Region ID
            task_info: Information about current task
            
        Returns:
            Local state s^local_r,t with:
            - Q_r(t): current task queue at static UAV
            - E_v^stat_r(t): residual energy
            - N^dyn_r(t): number of available dynamic UAVs in region
            - Λ_r(t): current spatio-temporal task intensity
        """
        static_uav = self.network.uav_manager.get_static_uav_by_region(region_id)
        
        if not static_uav:
            return {}
        
        # Available dynamic UAVs in region
        dynamic_uavs_in_region = self.network.uav_manager.get_available_dynamic_uavs_in_region(region_id)
        
        state = {
            'queue_length': static_uav.queue_length,
            'energy_level': static_uav.current_energy / static_uav.battery_capacity,
            'num_dynamic_uavs': len(dynamic_uavs_in_region),
            'task_intensity': self.network.task_manager.get_region_task_rate(region_id),
            'task_deadline': task_info.get('deadline', 0),
            'task_cpu_cycles': task_info.get('cpu_cycles', 0),
            'task_data_size': task_info.get('data_size', 0),
            'current_load': static_uav.total_workload / static_uav.cpu_capacity
        }
        
        return state
    
    def calculate_reward(self) -> float:
        """Calculate system-wide reward as per paper formulation.
        
        Returns:
            Reward r_t = Σ_j I(T_total,j ≤ τ_j) - α₁ΔL_t - α₂Σ_v I(E_v(t) < E_min)
        """
        alpha_1 = 0.5  # Load imbalance penalty weight
        alpha_2 = 1.0  # Energy penalty weight
        
        # Term 1: Tasks completed within deadline
        completed_tasks = 0
        total_tasks = 0
        
        for task in self.network.completed_tasks:
            total_tasks += 1
            if task.completion_time <= task.deadline:
                completed_tasks += 1
        
        success_rate = completed_tasks / max(total_tasks, 1)
        
        # Term 2: Load imbalance penalty
        load_imbalance = self.network.calculate_load_imbalance()
        
        # Term 3: Energy penalty (UAVs below minimum threshold)
        energy_violations = 0
        total_uavs = 0
        
        # Check static UAVs
        for uav in self.network.uav_manager.static_uavs.values():
            total_uavs += 1
            if uav.current_energy < uav.min_energy_threshold:
                energy_violations += 1
        
        # Check dynamic UAVs
        for uav in self.network.uav_manager.dynamic_uavs.values():
            total_uavs += 1
            if uav.current_energy < uav.min_energy_threshold:
                energy_violations += 1
        
        # Calculate final reward
        reward = success_rate - alpha_1 * load_imbalance - alpha_2 * (energy_violations / max(total_uavs, 1))
        
        return reward


def create_rl_config() -> Dict[str, Any]:
    """Create default RL configuration."""
    return {
        'num_episodes': 1000,
        'max_steps_per_episode': 100,
        'central_update_frequency': 5,
        'save_frequency': 100,
        'eval_frequency': 50,
        
        'central_agent_config': {
            'hidden_dim': 256,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'buffer_size': 10000,
            'batch_size': 64,
            'tau': 0.005
        },
        
        'local_agent_config': {
            'hidden_dim': 128,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995
        },
        
        'env_config': {
            'load_imbalance_weight': 0.5,
            'energy_penalty_weight': 1.0,
            'min_energy_threshold': 1000.0,
            'central_action_interval': 5
        }
    }
