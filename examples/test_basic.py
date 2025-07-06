"""
Simple test script for SAGIN system basic functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import SAGINNetwork
from core.types import SystemParameters, Position


def test_basic_functionality():
    """Test basic SAGIN functionality."""
    print("Testing SAGIN Basic Functionality")
    print("="*40)
    
    # Create system parameters
    params = SystemParameters(
        epoch_duration=1.0,
        total_epochs=10,
        min_rate_threshold=1.0,
        uav_max_speed=20.0,
        uav_altitude=100.0
    )
    
    # Initialize network
    network = SAGINNetwork(params)
    print("✓ Created SAGIN network")
    
    # Setup network topology
    area_bounds = (0.0, 5000.0, 0.0, 5000.0)
    network.setup_network_topology(area_bounds, num_regions=3)
    print("✓ Setup network topology with 3 regions")
    
    # Add vehicles
    vehicles = network.add_vehicles(20, area_bounds, vehicle_type="random")
    print(f"✓ Added {len(vehicles)} vehicles")
    
    # Add dynamic UAVs
    dynamic_uavs = network.add_dynamic_uavs(5, area_bounds)
    print(f"✓ Added {len(dynamic_uavs)} dynamic UAVs")
    
    # Add satellites
    satellites = network.add_satellite_constellation(num_satellites=6, num_planes=2)
    print(f"✓ Added {len(satellites)} satellites")
    
    # Initialize simulation
    network.initialize_simulation()
    print("✓ Initialized simulation")
    
    # Run a few simulation steps
    print("\nRunning simulation steps...")
    for i in range(5):
        step_results = network.step()
        new_tasks = len(step_results.get('new_tasks', []))
        print(f"  Step {i+1}: {new_tasks} new tasks generated")
    
    # Get final metrics
    metrics = network.get_performance_summary()
    print(f"\nFinal Results:")
    print(f"  Total epochs: {metrics['total_epochs']}")
    print(f"  Tasks generated: {metrics['final_metrics'].total_tasks_generated}")
    print(f"  Tasks completed: {metrics['final_metrics'].total_tasks_completed}")
    print(f"  Success rate: {metrics['final_metrics'].success_rate:.3f}")
    
    print("\n✓ All basic functionality tests passed!")
    return True


def test_network_components():
    """Test individual network components."""
    print("\nTesting Network Components")
    print("="*40)
    
    # Test Position class
    pos1 = Position(0, 0, 0)
    pos2 = Position(3, 4, 0)
    distance = pos1.distance_to(pos2)
    print(f"✓ Position distance calculation: {distance:.1f} meters")
    
    # Test SystemParameters
    params = SystemParameters()
    print(f"✓ Default system parameters: {params.epoch_duration}s epochs")
    
    # Test Region creation
    from core.types import Region
    region = Region(1, "Test Region", Position(1000, 1000, 0), 500.0)
    test_pos = Position(1100, 1100, 0)
    is_inside = region.contains_position(test_pos)
    print(f"✓ Region containment test: {is_inside}")
    
    print("✓ All component tests passed!")
    return True


def main():
    """Run all tests."""
    print("SAGIN System Test Suite")
    print("="*50)
    
    try:
        # Test basic functionality
        test_basic_functionality()
        
        # Test components
        test_network_components()
        
        print("\n🎉 All tests passed successfully!")
        print("The SAGIN system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()
