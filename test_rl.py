#!/usr/bin/env python3
"""
Simple test script to verify that the RL module can be imported
and interfaces correctly with the SAGIN system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.types import SystemParameters
from src.core.network import SAGINNetwork
from src.rl.environment import SAGINRLEnvironment


def test_rl_environment():
    """Test that the RL environment can be created and interfaces with SAGIN."""
    print("Testing RL environment...")
    
    # Create system parameters
    system_params = SystemParameters(
        epoch_duration=1.0,
        min_rate_threshold=1.0,
        min_energy_threshold=1000.0,
        propagation_speed=3e8
    )
    
    # Create network
    print("Creating SAGIN network...")
    network = SAGINNetwork(system_params=system_params)
    
    # Setup simple topology
    area_bounds = (-500, -500, 500, 500)
    print("Setting up network topology...")
    network.setup_network_topology(area_bounds, num_regions=3)
    
    # Add network elements
    print("Adding network elements...")
    network.add_vehicles(10, area_bounds)
    network.add_dynamic_uavs(2, area_bounds)
    network.add_satellite_constellation(2, num_planes=1)
    
    # Initialize simulation
    print("Initializing simulation...")
    network.initialize_simulation()
    
    # Create RL environment
    print("Creating RL environment...")
    env_config = {
        'load_imbalance_weight': 0.5,
        'energy_penalty_weight': 1.0,
        'min_energy_threshold': 1000.0,
        'central_action_interval': 5
    }
    env = SAGINRLEnvironment(network, env_config)
    
    # Test getting global state
    print("Getting global state...")
    global_state = env.get_global_state()
    print(f"Global state contains {len(global_state)} elements")
    
    # Test getting local state
    print("Getting local state for regions...")
    for region_id in network.regions:
        local_state = env.get_local_state(region_id)
        if local_state:
            print(f"Local state for region {region_id} contains {len(local_state)} elements")
    
    # Test stepping environment
    print("Stepping environment...")
    next_state, reward, done, info = env.step()
    
    print(f"Step completed with reward: {reward}")
    print(f"Info: {info}")
    
    # Reset environment
    print("Resetting environment...")
    initial_state = env.reset()
    
    print("RL environment test completed successfully!")
    return True


if __name__ == "__main__":
    test_rl_environment()
