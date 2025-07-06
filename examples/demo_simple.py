"""
Simple demonstration of SAGIN system without complex dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import SAGINNetwork
from core.types import SystemParameters, Position


def demo_sagin_system():
    """Demonstrate the SAGIN system with a simple scenario."""
    print("SAGIN System Demonstration")
    print("="*50)
    
    # Create system parameters
    params = SystemParameters(
        epoch_duration=1.0,  # 1 second per epoch
        total_epochs=50,     # Short demo
        min_rate_threshold=1.0,
        uav_max_speed=20.0,
        uav_altitude=100.0,
        max_load_imbalance=0.3
    )
    
    # Initialize network
    network = SAGINNetwork(params)
    print("✓ Created SAGIN network")
    
    # Setup a 5km x 5km area with 3 regions
    area_bounds = (0.0, 5000.0, 0.0, 5000.0)
    network.setup_network_topology(area_bounds, num_regions=3)
    print("✓ Setup network topology with 3 regions")
    
    # Add vehicles
    random_vehicles = network.add_vehicles(30, area_bounds, vehicle_type="random")
    bus_vehicles = network.add_vehicles(10, area_bounds, vehicle_type="bus")
    print(f"✓ Added {len(random_vehicles)} random vehicles and {len(bus_vehicles)} bus vehicles")
    
    # Add dynamic UAVs
    dynamic_uavs = network.add_dynamic_uavs(5, area_bounds)
    print(f"✓ Added {len(dynamic_uavs)} dynamic UAVs")
    
    # Add satellite constellation
    satellites = network.add_satellite_constellation(num_satellites=6, num_planes=2)
    print(f"✓ Added {len(satellites)} satellites")
    
    # Initialize simulation
    network.initialize_simulation()
    print("✓ Initialized simulation")
    
    # Add some burst events to make it interesting
    network.task_manager.add_burst_event(
        region_id=1, start_time=20.0, duration=15.0, amplitude=2.0
    )
    print("✓ Added burst event to region 1")
    
    print("\nStarting simulation...")
    print("-" * 50)
    
    # Run simulation with progress tracking
    metrics_snapshots = []
    
    for epoch in range(50):
        step_results = network.step()
        
        # Track metrics every 10 epochs
        if epoch % 10 == 0:
            metrics = network.metrics
            snapshot = {
                'epoch': epoch,
                'tasks_generated': metrics.total_tasks_generated,
                'tasks_completed': metrics.total_tasks_completed,
                'success_rate': metrics.success_rate,
                'avg_latency': metrics.average_latency,
                'uav_utilization': metrics.uav_utilization,
                'load_imbalance': metrics.load_imbalance
            }
            metrics_snapshots.append(snapshot)
            
            print(f"Epoch {epoch:2d}: "
                  f"Generated: {metrics.total_tasks_generated:3d}, "
                  f"Completed: {metrics.total_tasks_completed:3d}, "
                  f"Success: {metrics.success_rate:.3f}, "
                  f"Latency: {metrics.average_latency:.3f}s")
    
    print("-" * 50)
    print("Simulation completed!")
    
    # Show final results
    print("\nFinal Results:")
    print("="*30)
    performance = network.get_performance_summary()
    final_metrics = performance['final_metrics']
    
    print(f"Total Simulation Time: {final_metrics.current_time:.1f} seconds")
    print(f"Total Tasks Generated: {final_metrics.total_tasks_generated}")
    print(f"Total Tasks Completed: {final_metrics.total_tasks_completed}")
    print(f"Total Tasks Failed: {final_metrics.total_tasks_failed}")
    print(f"Overall Success Rate: {final_metrics.success_rate:.3f}")
    print(f"Average Task Latency: {final_metrics.average_latency:.3f} seconds")
    print(f"UAV Utilization: {final_metrics.uav_utilization:.3f}")
    print(f"Satellite Utilization: {final_metrics.satellite_utilization:.3f}")
    print(f"Load Imbalance: {final_metrics.load_imbalance:.3f}")
    print(f"Coverage Percentage: {final_metrics.coverage_percentage:.1f}%")
    
    # Show network state
    print(f"\nNetwork Elements:")
    elements = performance['network_elements']
    for element_type, count in elements.items():
        print(f"  {element_type.replace('_', ' ').title()}: {count}")
    
    # Show regional distribution
    print(f"\nRegional Distribution:")
    network_state = network.get_network_state()
    for region_id, region_info in network_state['regions'].items():
        print(f"  Region {region_id}: {region_info['vehicle_count']} vehicles, "
              f"{region_info['dynamic_uav_count']} dynamic UAVs")
    
    # Show progression over time
    print(f"\nPerformance Progression:")
    print("Epoch | Generated | Completed | Success | Latency | UAV_Util | Load_Imb")
    print("-" * 70)
    for snapshot in metrics_snapshots:
        print(f"{snapshot['epoch']:5d} | "
              f"{snapshot['tasks_generated']:9d} | "
              f"{snapshot['tasks_completed']:9d} | "
              f"{snapshot['success_rate']:7.3f} | "
              f"{snapshot['avg_latency']:7.3f} | "
              f"{snapshot['uav_utilization']:8.3f} | "
              f"{snapshot['load_imbalance']:8.3f}")
    
    print("\n🎉 SAGIN system demonstration completed successfully!")
    
    return network


def show_system_architecture():
    """Show the system architecture and components."""
    print("\nSAGIN System Architecture")
    print("="*50)
    
    print("""
    The SAGIN system consists of three main layers:
    
    1. GROUND LAYER (Vehicles)
       - Random mobility vehicles (cars, pedestrians)
       - Predictable mobility vehicles (buses, trains)
       - Generate computational tasks
       - Communicate with UAVs
    
    2. AIR LAYER (UAVs)
       - Static UAVs: One per region, continuous coverage
       - Dynamic UAVs: Repositionable based on demand
       - Process tasks locally or forward to satellites
       - Energy-constrained operations
    
    3. SPACE LAYER (Satellites)
       - LEO constellation for global coverage
       - High computational capacity
       - Handle overflow tasks from UAVs
       - Inter-satellite communication
    
    Key Features:
    - Hierarchical task offloading decisions
    - Dynamic UAV repositioning
    - Energy-aware operations
    - Load balancing across the network
    - Real-time latency optimization
    """)


def main():
    """Main demonstration function."""
    print("Welcome to the SAGIN System!")
    print("Space-Air-Ground Integrated Network Simulation")
    print("="*60)
    
    # Show architecture
    show_system_architecture()
    
    # Run demonstration
    network = demo_sagin_system()
    
    print("\nDemonstration Summary:")
    print("- Created a heterogeneous SAGIN network")
    print("- Simulated vehicle mobility and task generation")
    print("- Demonstrated hierarchical task offloading")
    print("- Showed dynamic UAV allocation")
    print("- Tracked performance metrics over time")
    
    print("\nThe system successfully demonstrates:")
    print("✓ Multi-layer network architecture")
    print("✓ Dynamic task generation and processing")
    print("✓ Energy-aware UAV operations")
    print("✓ Load balancing across network elements")
    print("✓ Real-time performance monitoring")
    
    print("\nFor more advanced features, see the RL-based optimization modules!")


if __name__ == "__main__":
    main()
