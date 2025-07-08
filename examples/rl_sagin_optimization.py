"""
Example script demonstrating the RL-based SAGIN optimization.

This script shows how to use the hierarchical RL system to train
agents that optimize dynamic UAV allocation and task offloading in
the SAGIN system according to the paper.
"""

import os
import sys
import time
import numpy as np
import argparse
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import SystemParameters
from src.core.network import SAGINNetwork
from src.rl.trainers import HierarchicalRLTrainer
from src.rl.environment import SAGINRLEnvironment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='RL-based SAGIN optimization')
    
    parser.add_argument('--train', action='store_true',
                      help='Train the RL agents')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate trained RL agents')
    parser.add_argument('--model-path', type=str, default='results/final_model',
                      help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    parser.add_argument('--regions', type=int, default=5,
                      help='Number of regions in the simulation')
    parser.add_argument('--vehicles', type=int, default=20,
                      help='Number of vehicles in the simulation')
    parser.add_argument('--dynamic-uavs', type=int, default=3,
                      help='Number of dynamic UAVs')
    parser.add_argument('--satellites', type=int, default=3,
                      help='Number of satellites')
    
    return parser.parse_args()


def create_network(args) -> SAGINNetwork:
    """Create and initialize the SAGIN network."""
    # Create system parameters
    system_params = SystemParameters(
        epoch_duration=1.0,  # 1 second per epoch
        min_rate_threshold=1.0,  # Minimum data rate (Mbps)
        min_energy_threshold=1000.0,  # Minimum energy threshold (J)
        propagation_speed=3e8  # Speed of light (m/s)
    )
    
    # Create network
    network = SAGINNetwork(system_params=system_params)
    
    # Setup simulation area
    area_bounds = (-1000, -1000, 1000, 1000)  # (x_min, y_min, x_max, y_max)
    network.setup_network_topology(area_bounds, num_regions=args.regions)
    
    # Add vehicles
    network.add_vehicles(args.vehicles, area_bounds)
    
    # Add dynamic UAVs
    network.add_dynamic_uavs(args.dynamic_uavs, area_bounds)
    
    # Add satellite constellation
    network.add_satellite_constellation(args.satellites, num_planes=1)
    
    # Initialize the simulation
    network.initialize_simulation()
    
    return network


def get_rl_config(args) -> Dict[str, Any]:
    """Get configuration for the RL system."""
    return {
        'num_episodes': args.episodes,
        'max_steps_per_episode': 100,
        'central_update_frequency': 5,
        'results_dir': 'results',
        
        'env_config': {
            'load_imbalance_weight': 0.5,
            'energy_penalty_weight': 1.0,
            'min_energy_threshold': 1000.0,
            'central_action_interval': 5
        },
        
        'central_agent_config': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.995,
            'hidden_dim': 256,
            'buffer_size': 10000,
            'batch_size': 64,
            'tau': 0.005
        },
        
        'local_agent_config': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.99,
            'hidden_dim': 128
        }
    }


def train_rl(args):
    """Train the RL system."""
    print("Setting up SAGIN network for RL training...")
    network = create_network(args)
    
    print("Creating RL trainer...")
    config = get_rl_config(args)
    trainer = HierarchicalRLTrainer(network, config)
    
    print(f"Starting training for {args.episodes} episodes...")
    start_time = time.time()
    trainer.train(verbose=True)
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds.")


def evaluate_rl(args):
    """Evaluate the trained RL system."""
    print("Setting up SAGIN network for RL evaluation...")
    network = create_network(args)
    
    print("Creating RL trainer...")
    config = get_rl_config(args)
    trainer = HierarchicalRLTrainer(network, config)
    
    # Load trained models
    print(f"Loading trained models from {args.model_path}...")
    try:
        # Loading would be implemented here
        pass
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print(f"Evaluating for {args.eval_episodes} episodes...")
    metrics = trainer.evaluate(args.eval_episodes, render=True)
    
    print("\nEvaluation results:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.train:
        train_rl(args)
    
    if args.eval:
        evaluate_rl(args)
    
    if not args.train and not args.eval:
        print("Please specify --train or --eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
