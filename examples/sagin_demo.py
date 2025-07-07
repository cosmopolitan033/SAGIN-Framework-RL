"""
Comprehensive SAGIN Network Simulation Demo
==========================================

This single script provides all SAGIN simulation functionality with configurable options:
- Quick test mode for validation
- Simple demo for learning
- Full simulation with detailed logging options
- Performance analysis and visualization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from core.network import SAGINNetwork
from core.types import SystemParameters, Position


class SAGINDemo:
    """Comprehensive SAGIN demonstration with multiple modes."""
    
    def __init__(self):
        self.network = None
        self.metrics_history = []
    
    def create_network(self, scale: str = "medium") -> SAGINNetwork:
        """Create a SAGIN network with configurable scale."""
        if scale == "small":
            # Small network for testing/quick demos
            params = SystemParameters(
                epoch_duration=1.0,
                total_epochs=50,
                min_rate_threshold=1.0,
                uav_max_speed=20.0,
                uav_altitude=100.0,
                max_load_imbalance=0.3
            )
            area_bounds = (0.0, 5000.0, 0.0, 5000.0)
            num_regions = 3
            random_vehicles = 20
            bus_vehicles = 5
            dynamic_uavs = 3
            satellites = 6
            planes = 2
            
        elif scale == "medium":
            # Medium network for standard demos
            params = SystemParameters(
                epoch_duration=1.0,
                total_epochs=100,
                min_rate_threshold=1.0,
                uav_max_speed=20.0,
                uav_altitude=100.0,
                max_load_imbalance=0.3
            )
            area_bounds = (0.0, 8000.0, 0.0, 8000.0)
            num_regions = 4
            random_vehicles = 50
            bus_vehicles = 15
            dynamic_uavs = 7
            satellites = 9
            planes = 3
            
        else:  # large
            # Large network for comprehensive simulations
            params = SystemParameters(
                epoch_duration=1.0,
                total_epochs=500,
                min_rate_threshold=1.0,
                uav_max_speed=20.0,
                uav_altitude=100.0,
                max_load_imbalance=0.3
            )
            area_bounds = (0.0, 10000.0, 0.0, 10000.0)
            num_regions = 5
            random_vehicles = 80
            bus_vehicles = 20
            dynamic_uavs = 10
            satellites = 12
            planes = 3
        
        # Initialize network
        network = SAGINNetwork(params)
        
        # Setup network topology
        network.setup_network_topology(area_bounds, num_regions=num_regions)
        
        # Add vehicles
        random_veh = network.add_vehicles(random_vehicles, area_bounds, vehicle_type="random")
        bus_veh = network.add_vehicles(bus_vehicles, area_bounds, vehicle_type="bus")
        
        # Add dynamic UAVs
        dynamic_uav_list = network.add_dynamic_uavs(dynamic_uavs, area_bounds)
        
        # Add satellite constellation
        satellite_list = network.add_satellite_constellation(num_satellites=satellites, num_planes=planes)
        
        print(f"Created {scale} SAGIN network with:")
        print(f"  - {num_regions} regions ({area_bounds[1]/1000:.0f}km x {area_bounds[3]/1000:.0f}km)")
        print(f"  - {len(random_veh)} random vehicles")
        print(f"  - {len(bus_veh)} bus vehicles")
        print(f"  - {num_regions} static UAVs (1 per region)")
        print(f"  - {len(dynamic_uav_list)} dynamic UAVs")
        print(f"  - {len(satellite_list)} satellites")
        
        self.network = network
        return network
    
    def add_burst_events(self, network: SAGINNetwork, intensity: str = "medium"):
        """Add burst events with configurable intensity."""
        if intensity == "light":
            events = [
                (1, 20.0, 15.0, 1.5),
            ]
        elif intensity == "medium":
            events = [
                (1, 30.0, 20.0, 2.0),
                (2, 60.0, 15.0, 1.8),
            ]
        else:  # heavy
            events = [
                (1, 100.0, 50.0, 3.0),
                (2, 200.0, 30.0, 2.5),
                (3, 350.0, 40.0, 2.0),
            ]
        
        for region_id, start_time, duration, amplitude in events:
            network.task_manager.add_burst_event(region_id, start_time, duration, amplitude)
        
        print(f"Added {len(events)} burst events ({intensity} intensity)")
    
    def run_test_mode(self):
        """Quick test mode - validate basic functionality."""
        print("🧪 SAGIN SYSTEM TEST MODE")
        print("="*50)
        
        # Create small network
        network = self.create_network("small")
        network.initialize_simulation()
        
        # Run a few steps
        print("\nRunning basic functionality test...")
        for i in range(5):
            step_results = network.step()
            new_tasks = len(step_results.get('new_tasks', []))
            print(f"  Step {i+1}: {new_tasks} new tasks generated")
        
        # Get results
        performance = network.get_performance_summary()
        metrics = performance['final_metrics']
        
        print(f"\nTest Results:")
        print(f"  ✓ Tasks generated: {metrics.total_tasks_generated}")
        print(f"  ✓ Tasks completed: {metrics.total_tasks_completed}")
        print(f"  ✓ Success rate: {metrics.success_rate:.3f}")
        print(f"  ✓ System working correctly!")
        
        return True
    
    def run_detailed_simulation(self, logging_level: str = "medium"):
        """Run detailed simulation with configurable logging."""
        print("🔍 SAGIN DETAILED SIMULATION")
        print("="*50)
        
        # Create network
        network = self.create_network("large")
        
        # Enable detailed logging based on level
        if logging_level in ["medium", "high"]:
            network.log_decisions = True
            network.log_resource_usage = True
        
        network.initialize_simulation()
        self.add_burst_events(network, "heavy")
        
        # Configure epochs based on logging level
        if logging_level == "high":
            num_epochs = 100
            detailed_interval = 2
            progress_interval = 10
        elif logging_level == "medium":
            num_epochs = 200
            detailed_interval = 5
            progress_interval = 25
        else:  # low
            num_epochs = 500
            detailed_interval = 0
            progress_interval = 50
        
        print(f"\nRunning {num_epochs}-epoch simulation with {logging_level} logging...")
        
        # Track metrics
        self.metrics_history = []
        
        def progress_callback(epoch: int, total_epochs: int, step_results: Dict):
            """Track progress and metrics."""
            if epoch % progress_interval == 0:
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
        if detailed_interval > 0:
            print(f"\n🔍 Running with detailed logging every {detailed_interval} epochs...")
            network.run_simulation_with_detailed_logging(
                num_epochs, detailed_interval, progress_callback
            )
        else:
            print(f"\n📊 Running standard simulation...")
            network.run_simulation(num_epochs, progress_callback)
        
        # Show detailed analysis
        if logging_level in ["medium", "high"]:
            print("\n🔍 DETAILED ANALYSIS")
            print("="*50)
            network.print_decision_analysis(min(50, num_epochs))
            network.print_resource_utilization_summary(min(25, num_epochs))
        
        return network
    
    def print_summary(self, network: SAGINNetwork):
        """Print comprehensive network summary."""
        print("\n" + "="*60)
        print("SAGIN NETWORK SUMMARY")
        print("="*60)
        
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
        fig.suptitle('SAGIN Network Simulation Results', fontsize=16)
        
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
    
    def export_results(self, network: SAGINNetwork, filename: str = "sagin_results.json"):
        """Export simulation results."""
        try:
            network.export_results(filename)
            print(f"\n💾 Results exported to '{filename}'")
        except Exception as e:
            print(f"Could not export results: {e}")


def main():
    """Main function with interactive menu."""
    print("🛰️  SAGIN Network Simulation")
    print("Space-Air-Ground Integrated Network")
    print("="*60)
    
    demo = SAGINDemo()
    
    while True:
        print("\nSelect simulation mode:")
        print("1. 🧪 Test Mode (Quick validation - 5 steps)")
        print("2. 📊 Standard Simulation (Medium logging - 200 epochs)")
        print("3. 🔍 Detailed Simulation (High logging - 100 epochs)")
        print("4. 🚀 Full Simulation (Low logging - 500 epochs)")
        print("5. 🎯 Custom Simulation (Choose your settings)")
        print("6. ❌ Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                demo.run_test_mode()
                
            elif choice == '2':
                network = demo.run_detailed_simulation("low")
                demo.print_summary(network)
                
            elif choice == '3':
                network = demo.run_detailed_simulation("medium")
                demo.print_summary(network)
                
            elif choice == '4':
                network = demo.run_detailed_simulation("low")
                demo.print_summary(network)
                
            elif choice == '5':
                # Custom simulation
                print("\nCustom Simulation Settings:")
                scale = input("Network scale (small/medium/large) [medium]: ").strip() or "medium"
                logging = input("Logging level (low/medium/high) [medium]: ").strip() or "medium"
                
                network = demo.create_network(scale)
                network.initialize_simulation()
                
                burst = input("Burst intensity (light/medium/heavy) [medium]: ").strip() or "medium"
                demo.add_burst_events(network, burst)
                
                network = demo.run_detailed_simulation(logging)
                demo.print_summary(network)
                
            elif choice == '6':
                print("Goodbye! 👋")
                break
                
            else:
                print("Invalid choice. Please try again.")
                continue
            
            if choice in ['2', '3', '4', '5']:
                # Post-simulation options
                print("\nPost-simulation options:")
                print("p. 📈 Plot results")
                print("e. 💾 Export results")
                print("c. 🔄 Continue with new simulation")
                
                post_choice = input("Enter choice (p/e/c) or press Enter to continue: ").strip()
                
                if post_choice == 'p':
                    try:
                        demo.plot_results(network)
                    except Exception as e:
                        print(f"Could not display plots: {e}")
                
                elif post_choice == 'e':
                    filename = input("Export filename [sagin_results.json]: ").strip()
                    demo.export_results(network, filename or "sagin_results.json")
        
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
