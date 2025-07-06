"""
Basic example of SAGIN network simulation.
This script demonstrates the core functionality of the SAGIN system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from core.network import SAGINNetwork
from core.types import SystemParameters, Position


def create_sample_network() -> SAGINNetwork:
    """Create a sample SAGIN network for demonstration."""
    # Create system parameters
    params = SystemParameters(
        epoch_duration=1.0,  # 1 second per epoch
        total_epochs=500,
        min_rate_threshold=1.0,  # 1 Mbps minimum
        uav_max_speed=20.0,  # 20 m/s
        uav_altitude=100.0,  # 100 m altitude
        max_load_imbalance=0.3
    )
    
    # Initialize network
    network = SAGINNetwork(params)
    
    # Define area (10km x 10km)
    area_bounds = (0.0, 10000.0, 0.0, 10000.0)
    
    # Setup network topology with 5 regions
    network.setup_network_topology(area_bounds, num_regions=5)
    
    # Add vehicles
    random_vehicles = network.add_vehicles(80, area_bounds, vehicle_type="random")
    bus_vehicles = network.add_vehicles(20, area_bounds, vehicle_type="bus")
    
    # Add dynamic UAVs
    dynamic_uavs = network.add_dynamic_uavs(10, area_bounds)
    
    # Add satellite constellation
    satellites = network.add_satellite_constellation(num_satellites=12, num_planes=3)
    
    print(f"Created SAGIN network with:")
    print(f"  - 5 regions")
    print(f"  - {len(random_vehicles)} random vehicles")
    print(f"  - {len(bus_vehicles)} bus vehicles")
    print(f"  - 5 static UAVs (1 per region)")
    print(f"  - {len(dynamic_uavs)} dynamic UAVs")
    print(f"  - {len(satellites)} satellites")
    
    return network


def add_burst_events(network: SAGINNetwork):
    """Add some burst events to make the simulation more interesting."""
    # Add burst events in different regions at different times
    network.task_manager.add_burst_event(
        region_id=1, start_time=100.0, duration=50.0, amplitude=3.0
    )
    
    network.task_manager.add_burst_event(
        region_id=3, start_time=200.0, duration=30.0, amplitude=2.5
    )
    
    network.task_manager.add_burst_event(
        region_id=2, start_time=350.0, duration=40.0, amplitude=2.0
    )
    
    print("Added burst events to regions 1, 2, and 3")


def run_simulation(network: SAGINNetwork, num_epochs: int = 500):
    """Run the SAGIN simulation."""
    print(f"\nStarting simulation for {num_epochs} epochs...")
    
    # Initialize the simulation
    network.initialize_simulation()
    
    # Add burst events
    add_burst_events(network)
    
    # Track metrics over time
    metrics_history = []
    
    def progress_callback(epoch: int, total_epochs: int, step_results: Dict):
        """Callback function to track progress."""
        if epoch % 50 == 0:
            metrics = network.metrics
            metrics_history.append({
                'epoch': epoch,
                'success_rate': metrics.success_rate,
                'average_latency': metrics.average_latency,
                'load_imbalance': metrics.load_imbalance,
                'uav_utilization': metrics.uav_utilization,
                'satellite_utilization': metrics.satellite_utilization,
                'tasks_generated': len(step_results.get('new_tasks', [])),
                'tasks_completed': len(step_results.get('uav_completed', {}).get('static_completed', [])) + 
                                 len(step_results.get('uav_completed', {}).get('dynamic_completed', [])) +
                                 len(step_results.get('satellite_completed', {}).get('satellite_completed', []))
            })
            
            print(f"Epoch {epoch:3d}: "
                  f"Success rate: {metrics.success_rate:.3f}, "
                  f"Avg latency: {metrics.average_latency:.3f}s, "
                  f"Load imbalance: {metrics.load_imbalance:.3f}")
    
    # Run simulation
    network.run_simulation(num_epochs, progress_callback)
    
    return metrics_history


def plot_results(metrics_history: List[Dict], network: SAGINNetwork):
    """Plot simulation results."""
    if not metrics_history:
        print("No metrics to plot")
        return
    
    epochs = [m['epoch'] for m in metrics_history]
    success_rates = [m['success_rate'] for m in metrics_history]
    avg_latencies = [m['average_latency'] for m in metrics_history]
    load_imbalances = [m['load_imbalance'] for m in metrics_history]
    uav_utilizations = [m['uav_utilization'] for m in metrics_history]
    satellite_utilizations = [m['satellite_utilization'] for m in metrics_history]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SAGIN Network Simulation Results', fontsize=16)
    
    # Success rate
    axes[0, 0].plot(epochs, success_rates, 'b-', linewidth=2)
    axes[0, 0].set_title('Task Success Rate')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Average latency
    axes[0, 1].plot(epochs, avg_latencies, 'r-', linewidth=2)
    axes[0, 1].set_title('Average Task Latency')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Latency (s)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Load imbalance
    axes[0, 2].plot(epochs, load_imbalances, 'g-', linewidth=2)
    axes[0, 2].set_title('Load Imbalance')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Load Imbalance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # UAV utilization
    axes[1, 0].plot(epochs, uav_utilizations, 'm-', linewidth=2)
    axes[1, 0].set_title('UAV Utilization')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Utilization')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Satellite utilization
    axes[1, 1].plot(epochs, satellite_utilizations, 'c-', linewidth=2)
    axes[1, 1].set_title('Satellite Utilization')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Utilization')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # System overview - final metrics
    final_metrics = network.get_performance_summary()
    final_text = f"""Final System Performance:
Total Tasks Generated: {final_metrics['final_metrics'].total_tasks_generated}
Total Tasks Completed: {final_metrics['final_metrics'].total_tasks_completed}
Total Tasks Failed: {final_metrics['final_metrics'].total_tasks_failed}
Final Success Rate: {final_metrics['final_metrics'].success_rate:.3f}
Final Average Latency: {final_metrics['final_metrics'].average_latency:.3f}s
Final Load Imbalance: {final_metrics['final_metrics'].load_imbalance:.3f}
UAV Utilization: {final_metrics['final_metrics'].uav_utilization:.3f}
Satellite Utilization: {final_metrics['final_metrics'].satellite_utilization:.3f}
Coverage: {final_metrics['final_metrics'].coverage_percentage:.1f}%"""
    
    axes[1, 2].text(0.05, 0.95, final_text, transform=axes[1, 2].transAxes,
                   verticalalignment='top', fontsize=10, family='monospace')
    axes[1, 2].set_title('Final Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_network_summary(network: SAGINNetwork):
    """Print a summary of the network state."""
    print("\n" + "="*60)
    print("SAGIN NETWORK SUMMARY")
    print("="*60)
    
    # Get network state
    network_state = network.get_network_state()
    performance = network.get_performance_summary()
    
    print(f"Simulation Time: {network_state['current_time']:.1f} seconds")
    print(f"Total Epochs: {network_state['epoch_count']}")
    print()
    
    print("Network Elements:")
    elements = performance['network_elements']
    for element_type, count in elements.items():
        print(f"  {element_type.replace('_', ' ').title()}: {count}")
    print()
    
    print("Task Statistics:")
    metrics = performance['final_metrics']
    print(f"  Total Generated: {metrics.total_tasks_generated}")
    print(f"  Total Completed: {metrics.total_tasks_completed}")
    print(f"  Total Failed: {metrics.total_tasks_failed}")
    print(f"  Success Rate: {metrics.success_rate:.3f}")
    print(f"  Average Latency: {metrics.average_latency:.3f} seconds")
    print()
    
    print("Resource Utilization:")
    print(f"  UAV Utilization: {metrics.uav_utilization:.3f}")
    print(f"  Satellite Utilization: {metrics.satellite_utilization:.3f}")
    print(f"  Load Imbalance: {metrics.load_imbalance:.3f}")
    print(f"  Coverage: {metrics.coverage_percentage:.1f}%")
    print()
    
    print("Regional Distribution:")
    for region_id, region_info in network_state['regions'].items():
        print(f"  Region {region_id}: {region_info['vehicle_count']} vehicles, "
              f"{region_info['dynamic_uav_count']} dynamic UAVs")


def main():
    """Main function to run the SAGIN simulation example."""
    print("SAGIN Network Simulation Example")
    print("="*40)
    
    # Create network
    network = create_sample_network()
    
    # Run simulation
    metrics_history = run_simulation(network, num_epochs=500)
    
    # Print summary
    print_network_summary(network)
    
    # Plot results
    try:
        plot_results(metrics_history, network)
    except Exception as e:
        print(f"Could not display plots: {e}")
        print("(This is normal if running in a headless environment)")
    
    # Export results
    try:
        network.export_results("sagin_simulation_results.json")
        print("\nResults exported to 'sagin_simulation_results.json'")
    except Exception as e:
        print(f"Could not export results: {e}")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
