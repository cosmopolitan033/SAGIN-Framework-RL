"""
Detailed logging demonstration for SAGIN network simulation.
This script showcases the comprehensive process logging capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network import SAGINNetwork
from core.types import SystemParameters, Position


def create_small_network() -> SAGINNetwork:
    """Create a small SAGIN network for detailed logging demonstration."""
    # Create system parameters optimized for detailed logging
    params = SystemParameters(
        epoch_duration=2.0,  # 2 seconds per epoch for more detailed observation
        total_epochs=50,
        min_rate_threshold=1.0,
        uav_max_speed=15.0,
        uav_altitude=100.0,
        max_load_imbalance=0.4
    )
    
    # Initialize network
    network = SAGINNetwork(params)
    
    # Enable detailed logging
    network.log_decisions = True
    network.log_resource_usage = True
    
    # Define smaller area (5km x 5km) for easier tracking
    area_bounds = (0.0, 5000.0, 0.0, 5000.0)
    
    # Setup network topology with 3 regions
    network.setup_network_topology(area_bounds, num_regions=3)
    
    # Add fewer vehicles for easier tracking
    random_vehicles = network.add_vehicles(30, area_bounds, vehicle_type="random")
    bus_vehicles = network.add_vehicles(10, area_bounds, vehicle_type="bus")
    
    # Add fewer dynamic UAVs
    dynamic_uavs = network.add_dynamic_uavs(5, area_bounds)
    
    # Add satellite constellation
    satellites = network.add_satellite_constellation(num_satellites=6, num_planes=2)
    
    print(f"Created detailed logging demo network with:")
    print(f"  - 3 regions")
    print(f"  - {len(random_vehicles)} random vehicles")
    print(f"  - {len(bus_vehicles)} bus vehicles")
    print(f"  - 3 static UAVs (1 per region)")
    print(f"  - {len(dynamic_uavs)} dynamic UAVs")
    print(f"  - {len(satellites)} satellites")
    
    return network


def add_demo_burst_events(network: SAGINNetwork):
    """Add burst events designed for demonstration."""
    # Add smaller, more frequent burst events
    network.task_manager.add_burst_event(
        region_id=1, start_time=10.0, duration=20.0, amplitude=2.0
    )
    
    network.task_manager.add_burst_event(
        region_id=2, start_time=30.0, duration=15.0, amplitude=1.5
    )
    
    network.task_manager.add_burst_event(
        region_id=3, start_time=60.0, duration=25.0, amplitude=2.5
    )
    
    print("Added demonstration burst events:")
    print("  - Region 1: t=10-30s, amplitude=2.0")
    print("  - Region 2: t=30-45s, amplitude=1.5") 
    print("  - Region 3: t=60-85s, amplitude=2.5")


def run_detailed_logging_demo():
    """Run the detailed logging demonstration."""
    print("🔍 SAGIN Detailed Logging Demonstration")
    print("="*60)
    
    # Create network
    network = create_small_network()
    
    # Initialize simulation
    network.initialize_simulation()
    
    # Add burst events
    add_demo_burst_events(network)
    
    print(f"\n🚀 Starting detailed simulation...")
    print("This will show comprehensive logging for each decision made.")
    print("="*60)
    
    # Run simulation with detailed logging every epoch
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"🎯 EPOCH {epoch} ANALYSIS")
        print(f"{'='*70}")
        
        # Execute step with detailed logging
        step_results = network.step_with_detailed_logging()
        
        # Show decision and resource analysis every 5 epochs
        if epoch > 0 and epoch % 5 == 0:
            print(f"\n📊 PERIODIC ANALYSIS (Every 5 epochs)")
            print("-" * 50)
            network.print_decision_analysis(5)
            network.print_resource_utilization_summary(5)
        
        # Show system state every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            network.print_system_state_report(detailed=True)
        
        # Brief pause for readability (remove in production)
        if epoch < 10:  # Only pause for first 10 epochs
            try:
                input(f"\nPress Enter to continue to epoch {epoch + 1} (or Ctrl+C to skip pauses)...")
            except KeyboardInterrupt:
                print("\nContinuing without pauses...")
                break
    
    print(f"\n✅ Detailed logging demonstration completed!")
    print("="*60)
    
    # Final comprehensive analysis
    print("\n🎯 FINAL COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Show final decision statistics
    network.print_decision_analysis(50)
    
    # Show final resource utilization
    network.print_resource_utilization_summary(25)
    
    # Show final system state
    network.print_system_state_report(detailed=True)
    
    # Export detailed results
    try:
        network.export_results("detailed_logging_demo_results.json")
        print("\n💾 Detailed results exported to 'detailed_logging_demo_results.json'")
    except Exception as e:
        print(f"Could not export results: {e}")
    
    print("\n🎉 Demonstration completed successfully!")
    print("This showcases the comprehensive decision-making visibility in SAGIN simulation.")


def run_quick_demo():
    """Run a quick version of the demo without pauses."""
    print("🔍 SAGIN Quick Detailed Logging Demo")
    print("="*50)
    
    # Create network
    network = create_small_network()
    
    # Initialize simulation
    network.initialize_simulation()
    
    # Add burst events
    add_demo_burst_events(network)
    
    # Run simulation with detailed logging every 2 epochs
    print(f"\n🚀 Running quick detailed simulation (25 epochs)...")
    network.run_simulation_with_detailed_logging(
        num_epochs=25,
        detailed_interval=2
    )
    
    print("\n✅ Quick demo completed!")


def main():
    """Main function for the detailed logging demonstration."""
    print("SAGIN Detailed Logging Demonstration")
    print("="*50)
    
    print("\nSelect demo mode:")
    print("1. Full detailed demo (interactive, with pauses)")
    print("2. Quick demo (non-interactive, faster)")
    print("3. Exit")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            run_detailed_logging_demo()
        elif choice == '2':
            run_quick_demo()
        elif choice == '3':
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Running quick demo...")
            run_quick_demo()
            
    except (EOFError, KeyboardInterrupt):
        print("\nRunning quick demo...")
        run_quick_demo()


if __name__ == "__main__":
    main()
