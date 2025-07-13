"""
Comprehensive SAGIN Network Simulation Demo (Version 2)
======================================================

This script uses the comprehensive configuration system to run different
SAGIN scenarios without needing code modifications. All parameters can be
configured in the config files.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import time
import json

from src.core.network import SAGINNetwork
from src.core.uavs import UAVStatus
from src.core.types import Position
from src.visualization.real_time_visualizer import RealTimeNetworkVisualizer
from config.grid_config import get_sagin_config, list_available_configs, print_config_summary


class SAGINDemo:
    """SAGIN demonstration using comprehensive configuration system."""
    
    def __init__(self):
        self.network = None
        self.metrics_history = []
        self.current_config = None
    
    def create_network(self, config_name: str = "medium_demo") -> SAGINNetwork:
        """Create a SAGIN network using a predefined configuration."""
        # Get the configuration
        self.current_config = get_sagin_config(config_name)
        config = self.current_config
        
        # Convert to system parameters
        system_params = config.get_system_parameters()
        
        # Initialize network
        network = SAGINNetwork(system_params)
        
        # Setup network topology using grid configuration
        network.setup_network_topology_with_grid(config.grid)
        
        # Add vehicles according to configuration
        random_veh = network.add_vehicles(
            config.vehicles.random_vehicles, 
            config.grid.area_bounds, 
            vehicle_type="random"
        )
        bus_veh = network.add_vehicles(
            config.vehicles.bus_vehicles, 
            config.grid.area_bounds, 
            vehicle_type="bus"
        )
        
        # Add dynamic UAVs according to configuration
        dynamic_uav_list = network.add_dynamic_uavs(
            config.uavs.dynamic_uavs, 
            config.grid.area_bounds
        )
        
        # Assign dynamic UAVs to initial regions based on their positions
        for uav_id in dynamic_uav_list:
            uav = network.uav_manager.dynamic_uavs[uav_id]
            # Find the closest region to this UAV's position
            closest_region_id = None
            min_distance = float('inf')
            
            for region_id, region in network.regions.items():
                distance = uav.position.distance_to(region.center)
                if distance < min_distance:
                    min_distance = distance
                    closest_region_id = region_id
            
            if closest_region_id is not None:
                uav.assigned_region_id = closest_region_id
                uav.status = UAVStatus.ACTIVE
                uav.availability_indicator = 1
                print(f"  Assigned dynamic UAV {uav_id} to region {closest_region_id}")
            else:
                print(f"  Warning: Could not assign dynamic UAV {uav_id} to any region")
        
        # Add satellite constellation according to configuration
        satellite_list = network.add_satellite_constellation(
            num_satellites=config.satellites.num_satellites, 
            num_planes=config.satellites.num_planes
        )
        
        # Configure task generation
        if config.tasks.burst_events:
            for region_id, start_time, duration, amplitude in config.tasks.burst_events:
                network.task_manager.add_burst_event(region_id, start_time, duration, amplitude)
        
        print(f"Created {config_name} SAGIN network:")
        print(f"  - {config.grid.grid_rows}x{config.grid.grid_cols} grid ({config.grid.total_regions} regions)")
        print(f"  - Area: {config.grid.area_bounds[1]/1000:.1f}km x {config.grid.area_bounds[3]/1000:.1f}km")
        print(f"  - {len(random_veh)} random vehicles")
        print(f"  - {len(bus_veh)} bus vehicles")
        print(f"  - {config.grid.total_regions} static UAVs (1 per region)")
        print(f"  - {len(dynamic_uav_list)} dynamic UAVs")
        print(f"  - {len(satellite_list)} satellites")
        if config.tasks.burst_events:
            print(f"  - {len(config.tasks.burst_events)} burst events configured")
        
        self.network = network
        return network
    
    def run_simulation(self, config_name: str = "medium_demo"):
        """Run simulation using the specified configuration."""
        print(f"🚀 SAGIN SIMULATION: {config_name.upper()}")
        print("="*60)
        
        # Create network with configuration
        network = self.create_network(config_name)
        config = self.current_config
        
        # Configure logging based on config
        if config.simulation.logging_level in ["medium", "high"]:
            network.log_decisions = config.simulation.log_decisions
            network.log_resource_usage = config.simulation.log_resource_usage
        
        network.initialize_simulation()
        
        print(f"\nRunning {config.simulation.total_epochs}-epoch simulation with {config.simulation.logging_level} logging...")
        
        # Track metrics
        self.metrics_history = []
        
        def progress_callback(epoch: int, total_epochs: int, step_results: Dict):
            """Track progress and metrics."""
            if epoch % config.simulation.progress_interval == 0:
                metrics = network.metrics
                self.metrics_history.append({
                    'epoch': epoch,
                    'success_rate': metrics.success_rate,
                    'average_latency': metrics.average_latency,
                    'load_imbalance': metrics.load_imbalance,
                    'uav_utilization': metrics.uav_utilization,
                    'satellite_utilization': metrics.satellite_utilization,
                    'tasks_generated': metrics.total_tasks_generated,
                    'tasks_completed': metrics.total_tasks_completed
                })
                
                print(f"Epoch {epoch:3d}: Success: {metrics.success_rate:.3f}, "
                      f"Latency: {metrics.average_latency:.3f}s, "
                      f"Load: {metrics.load_imbalance:.3f}")
        
        # Run simulation
        if config.simulation.detailed_interval > 0 and config.simulation.logging_level == "high":
            print(f"\n🔍 Running with detailed logging every {config.simulation.detailed_interval} epochs...")
            network.run_simulation_with_detailed_logging(
                config.simulation.total_epochs, config.simulation.detailed_interval, progress_callback
            )
        else:
            print(f"\n📊 Running standard simulation...")
            network.run_simulation(config.simulation.total_epochs, progress_callback)
        
        # Show detailed analysis
        if config.simulation.logging_level in ["medium", "high"]:
            print("\n🔍 DETAILED ANALYSIS")
            print("="*50)
            network.print_decision_analysis(min(50, config.simulation.total_epochs))
            network.print_resource_utilization_summary(min(25, config.simulation.total_epochs))
        
        return network
    
    def print_summary(self, network: SAGINNetwork):
        """Print comprehensive network summary."""
        print("\n" + "="*60)
        print("SAGIN NETWORK SUMMARY")
        print("="*60)
        
        network_state = network.get_network_state()
        performance = network.get_performance_summary()
        
        print(f"Configuration: {self.current_config.name}")
        print(f"Description: {self.current_config.description}")
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
    
    def plot_results(self, network: SAGINNetwork):
        """Plot comprehensive simulation results."""
        if not self.metrics_history:
            print("No metrics history to plot")
            return
        
        epochs = [m['epoch'] for m in self.metrics_history]
        success_rates = [m['success_rate'] for m in self.metrics_history]
        avg_latencies = [m['average_latency'] for m in self.metrics_history]
        load_imbalances = [m['load_imbalance'] for m in self.metrics_history]
        uav_utilizations = [m['uav_utilization'] for m in self.metrics_history]
        satellite_utilizations = [m['satellite_utilization'] for m in self.metrics_history]
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SAGIN Network Simulation Results: {self.current_config.name}', fontsize=16)
        
        # Plot metrics
        axes[0, 0].plot(epochs, success_rates, 'b-', linewidth=2)
        axes[0, 0].set_title('Task Success Rate')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].plot(epochs, avg_latencies, 'r-', linewidth=2)
        axes[0, 1].set_title('Average Task Latency')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs, load_imbalances, 'g-', linewidth=2)
        axes[0, 2].set_title('Load Imbalance')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Load Imbalance')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, uav_utilizations, 'm-', linewidth=2)
        axes[1, 0].set_title('UAV Utilization')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Utilization')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].plot(epochs, satellite_utilizations, 'c-', linewidth=2)
        axes[1, 1].set_title('Satellite Utilization')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Utilization')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        # Final performance summary
        final_metrics = network.get_performance_summary()
        final_text = f"""Final Performance:
Tasks Generated: {final_metrics['final_metrics'].total_tasks_generated}
Tasks Completed: {final_metrics['final_metrics'].total_tasks_completed}
Success Rate: {final_metrics['final_metrics'].success_rate:.3f}
Avg Latency: {final_metrics['final_metrics'].average_latency:.3f}s
Load Imbalance: {final_metrics['final_metrics'].load_imbalance:.3f}
UAV Utilization: {final_metrics['final_metrics'].uav_utilization:.3f}
Satellite Utilization: {final_metrics['final_metrics'].satellite_utilization:.3f}
Coverage: {final_metrics['final_metrics'].coverage_percentage:.1f}%"""
        
        axes[1, 2].text(0.05, 0.95, final_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10, family='monospace')
        axes[1, 2].set_title('Final Performance Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, network: SAGINNetwork):
        """Export simulation results."""
        try:
            filename = self.current_config.simulation.results_filename
            network.export_results(filename)
            print(f"\n💾 Results exported to '{filename}'")
        except Exception as e:
            print(f"Could not export results: {e}")


def main():
    """Main function with configuration-based menu."""
    print("🛰️  SAGIN Network Simulation (Configuration-Based)")
    print("Space-Air-Ground Integrated Network")
    print("="*60)
    
    demo = SAGINDemo()
    
    while True:
        print("\nSelect simulation mode:")
        print("1. 📋 List available configurations")
        print("2. ℹ️  Show configuration details")
        print("3. 🚀 Run simulation with configuration")
        print("4. 🧪 Quick test (small_test config)")
        print("5. 📡 Real-time visualization")
        print("0. ❌ Exit")
        print("=" * 60)
        
        try:
            choice = input("\nEnter choice (0-5): ").strip()
            
            if choice == '1':
                print_config_summary()
                
            elif choice == '2':
                config_name = input("Enter configuration name: ").strip()
                print_config_summary(config_name)
                
            elif choice == '3':
                print("\nAvailable configurations:")
                configs = list_available_configs()
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                
                try:
                    selection = input("\nEnter configuration name or number: ").strip()
                    if selection.isdigit():
                        config_idx = int(selection) - 1
                        if 0 <= config_idx < len(configs):
                            config_name = configs[config_idx]
                        else:
                            print("Invalid selection")
                            continue
                    else:
                        config_name = selection
                    
                    network = demo.run_simulation(config_name)
                    demo.print_summary(network)
                    
                except (ValueError, IndexError):
                    print("Invalid selection")
                    continue
                
            elif choice == '4':
                network = demo.run_simulation("small_test")
                demo.print_summary(network)
                
            elif choice == '5':
                print("\n📡 Real-Time Network Visualization")
                print("=" * 40)
                print("Available configurations:")
                configs = list_available_configs()
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                
                try:
                    selection = input("\nEnter configuration name or number for visualization: ").strip()
                    if selection.isdigit():
                        config_idx = int(selection) - 1
                        if 0 <= config_idx < len(configs):
                            config_name = configs[config_idx]
                        else:
                            print("Invalid selection")
                            continue
                    else:
                        config_name = selection
                    
                    # Create network for visualization
                    print(f"\n🔧 Setting up {config_name} network for real-time visualization...")
                    network = demo.create_network(config_name)
                    network.initialize_simulation()
                    
                    print("🎨 Starting real-time visualization...")
                    print("   - Color-coded regions show computational load")
                    print("   - Blue circles: Cars, Orange squares: Buses")
                    print("   - Green triangles: Static UAVs, Red/Orange triangles: Dynamic UAVs")
                    print("   - Coverage zones show UAV communication range")
                    print("   - Green lines: Links to Static UAVs, Red lines: Links to Dynamic UAVs")
                    print("\n⚠️  Close the visualization window to return to menu")
                    
                    try:
                        visualizer = RealTimeNetworkVisualizer(network)
                        visualizer.run()
                    except Exception as e:
                        print(f"Visualization error: {e}")
                        print("Make sure matplotlib is properly installed.")
                    
                except (ValueError, IndexError):
                    print("Invalid selection")
                    continue
                
            elif choice == '0':
                print("Goodbye! 👋")
                break
                
            else:
                print("Invalid choice. Please try again.")
                continue
            
            if choice in ['3', '4']:
                # Post-simulation options
                print("\nPost-simulation options:")
                print("p. 📈 Plot results")
                print("v. 📡 Real-time visualization")
                print("e. 💾 Export results")
                print("c. 🔄 Continue with new simulation")
                
                post_choice = input("Enter choice (p/v/e/c) or press Enter to continue: ").strip()
                
                if post_choice == 'p':
                    try:
                        demo.plot_results(network)
                    except Exception as e:
                        print(f"Could not display plots: {e}")
                
                elif post_choice == 'v':
                    try:
                        print("🎨 Starting post-simulation visualization...")
                        print("⚠️  Close the visualization window to return to menu")
                        visualizer = RealTimeNetworkVisualizer(network)
                        visualizer.run(enable_simulation=False)  # Static visualization of final state
                    except Exception as e:
                        print(f"Visualization error: {e}")
                
                elif post_choice == 'e':
                    demo.export_results(network)
        
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
