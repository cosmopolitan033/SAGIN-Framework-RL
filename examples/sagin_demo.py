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

# RL Integration
try:
    from src.rl.rl_integration import RLModelManager, EnhancedSAGINRLEnvironment
    from src.rl.trainers import HierarchicalRLTrainer
    RL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RL components not available: {e}")
    RL_AVAILABLE = False


class SAGINDemo:
    """SAGIN demonstration using comprehensive configuration system."""
    
    def __init__(self):
        self.network = None
        self.metrics_history = []
        self.current_config = None
        self.orchestration_mode = "heuristic"  # "heuristic" or "rl"
        self.selected_rl_model = None
        
        # Initialize RL manager if available
        if RL_AVAILABLE:
            self.rl_manager = RLModelManager()
        else:
            self.rl_manager = None
    
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
        network.setup_network_topology_with_grid(config.grid, config.tasks, config.satellites)
        
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
            num_satellites=config.satellites.num_satellites
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
    
    def configure_task_proportions(self, network: SAGINNetwork) -> bool:
        """Interactive configuration of task type proportions.
        
        Args:
            network: The SAGIN network to configure
            
        Returns:
            True if user configured proportions, False if using defaults
        """
        print("\n🎯 Task Type Proportion Configuration")
        print("=" * 50)
        
        # Show current proportions
        current_proportions = network.get_task_type_proportions()
        print("Current task type proportions:")
        for task_type, proportion in current_proportions.items():
            print(f"  - {task_type.replace('_', ' ').title()}: {proportion:.1%}")
        
        print("\nOptions:")
        print("1. 🏭 High Computation (60% computation-intensive, 20% normal, 15% data, 5% latency)")
        print("2. 📊 Data Heavy (50% data-intensive, 30% normal, 10% computation, 10% latency)")
        print("3. ⚡ Real-time Critical (40% latency-sensitive, 30% normal, 20% computation, 10% data)")
        print("4. ⚖️  Balanced Mix (25% each type)")
        print("5. 🎛️  Custom proportions")
        print("6. ✅ Keep current proportions")
        
        while True:
            choice = input("\nSelect task proportion scenario (1-6): ").strip()
            
            if choice == '1':
                proportions = {
                    'normal': 0.2,
                    'computation_intensive': 0.6,
                    'data_intensive': 0.15,
                    'latency_sensitive': 0.05
                }
                network.set_task_type_proportions(proportions)
                print("✅ High Computation scenario applied")
                return True
                
            elif choice == '2':
                proportions = {
                    'normal': 0.3,
                    'computation_intensive': 0.1,
                    'data_intensive': 0.5,
                    'latency_sensitive': 0.1
                }
                network.set_task_type_proportions(proportions)
                print("✅ Data Heavy scenario applied")
                return True
                
            elif choice == '3':
                proportions = {
                    'normal': 0.3,
                    'computation_intensive': 0.2,
                    'data_intensive': 0.1,
                    'latency_sensitive': 0.4
                }
                network.set_task_type_proportions(proportions)
                print("✅ Real-time Critical scenario applied")
                return True
                
            elif choice == '4':
                proportions = {
                    'normal': 0.25,
                    'computation_intensive': 0.25,
                    'data_intensive': 0.25,
                    'latency_sensitive': 0.25
                }
                network.set_task_type_proportions(proportions)
                print("✅ Balanced Mix scenario applied")
                return True
                
            elif choice == '5':
                return self._configure_custom_proportions(network)
                
            elif choice == '6':
                print("✅ Keeping current task proportions")
                return False
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
    
    def _configure_custom_proportions(self, network: SAGINNetwork) -> bool:
        """Configure custom task type proportions."""
        print("\n🎛️  Custom Task Type Proportions")
        print("Enter proportions (as decimals, must sum to 1.0):")
        
        try:
            normal = float(input("Normal tasks (current: 60%): ") or "0.6")
            computation = float(input("Computation-intensive tasks (current: 20%): ") or "0.2")
            data = float(input("Data-intensive tasks (current: 15%): ") or "0.15")
            latency = float(input("Latency-sensitive tasks (current: 5%): ") or "0.05")
            
            # Validate proportions
            total = normal + computation + data + latency
            if abs(total - 1.0) > 0.001:
                print(f"❌ Error: Proportions sum to {total:.3f}, must sum to 1.0")
                return False
            
            proportions = {
                'normal': normal,
                'computation_intensive': computation,
                'data_intensive': data,
                'latency_sensitive': latency
            }
            
            network.set_task_type_proportions(proportions)
            print("✅ Custom proportions applied")
            return True
            
        except ValueError:
            print("❌ Invalid input. Please enter valid decimal numbers.")
            return False
    
    def select_orchestration_mode(self) -> bool:
        """Interactive selection of orchestration mode.
        
        Returns:
            True if user wants to proceed, False to cancel
        """
        if not RL_AVAILABLE:
            print("⚠️  RL components not available, using heuristic mode")
            self.orchestration_mode = "heuristic"
            return True
        
        print("\n🧠 Select Orchestration Mode:")
        print("=" * 40)
        print("1. 🔧 Heuristic (Original Algorithm)")
        print("2. 🤖 Reinforcement Learning")
        print("3. 🚀 Train New RL Model")
        print("0. ❌ Cancel")
        
        while True:
            try:
                choice = input("\nEnter choice (0-3): ").strip()
                
                if choice == '1':
                    self.orchestration_mode = "heuristic"
                    print("✅ Selected: Heuristic orchestration")
                    return True
                    
                elif choice == '2':
                    if self.rl_manager:
                        selected_model = self.rl_manager.interactive_model_selection()
                        if selected_model is None:
                            continue  # User cancelled model selection
                        elif selected_model == "heuristic":
                            self.orchestration_mode = "heuristic"
                            print("✅ Selected: Heuristic orchestration")
                        else:
                            self.orchestration_mode = "rl"
                            self.selected_rl_model = selected_model
                            print(f"✅ Selected: RL orchestration with model '{selected_model}'")
                        return True
                    else:
                        print("❌ RL manager not available")
                        continue
                        
                elif choice == '3':
                    success = self._train_new_rl_model()
                    if success:
                        return self.select_orchestration_mode()  # Re-select after training
                    else:
                        continue
                        
                elif choice == '0':
                    print("❌ Cancelled")
                    return False
                    
                else:
                    print("❌ Invalid choice. Please enter 0-3")
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input or interrupted")
                return False
    
    def _train_new_rl_model(self) -> bool:
        """Train a new RL model interactively.
        
        Returns:
            True if training successful, False otherwise
        """
        if not RL_AVAILABLE:
            print("❌ RL components not available for training")
            return False
        
        print("\n🚀 Train New RL Model")
        print("=" * 30)
        
        try:
            # Get model name
            model_name = input("Enter model name: ").strip()
            if not model_name:
                print("❌ Model name cannot be empty")
                return False
            
            # Get training configuration
            print("\nTraining Configuration:")
            try:
                episodes = int(input("Number of training episodes (default: 500): ").strip() or "500")
                config_name = input("Base configuration (default: medium_demo): ").strip() or "medium_demo"
                description = input("Model description: ").strip() or f"RL model trained on {config_name}"
            except ValueError:
                print("❌ Invalid input")
                return False
            
            print(f"\n🔧 Starting training of '{model_name}' for {episodes} episodes...")
            print("⚠️  Training may take several minutes. Please wait...")
            
            # Create training environment
            config = get_sagin_config(config_name)
            network = self.create_network(config_name)
            
            # Setup RL environment with both network and config
            rl_config = {
                'load_imbalance_weight': 0.5,
                'energy_penalty_weight': 1.0,
                'min_energy_threshold': 0.2,
                'central_action_interval': 5
            }
            
            # Create trainer config with training parameters
            trainer_config = {
                'env_config': rl_config,
                'num_episodes': episodes,
                'central_update_frequency': 5,
                'results_dir': "models/rl"
            }
            
            # Create trainer
            trainer = HierarchicalRLTrainer(network, trainer_config)
            
            # Train the model
            print("🏃 Training in progress...")
            start_time = time.time()
            
            central_agent, shared_static_uav_agent, training_stats = trainer.train()
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time:.1f} seconds")
            
            # Save the model
            model_info = {
                'description': description,
                'config': config_name,
                'episodes': episodes,
                'training_time': training_time,
                'performance': {
                    'final_reward': training_stats.get('final_average_reward', 0.0),
                    'success_rate': training_stats.get('success_rate', 0.0),
                    'avg_latency': training_stats.get('avg_latency', 0.0)
                }
            }
            
            self.rl_manager.save_model(model_name, central_agent, shared_static_uav_agent, model_info)
            print(f"💾 Model '{model_name}' saved successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def manage_rl_models(self):
        """RL model management interface."""
        if not RL_AVAILABLE or not self.rl_manager:
            print("❌ RL components not available")
            return
        
        while True:
            print("\n🤖 RL Model Management")
            print("=" * 40)
            print("1. 📋 List available RL models")
            print("2. 🚀 Train new RL model")
            print("3. ⚙️  Evaluate RL model")
            print("0. ⬅️  Back to main menu")
            
            choice = input("\nEnter choice (0-3): ").strip()
            
            if choice == '1':
                models = self.rl_manager.list_models()
                if not models:
                    print("❌ No trained models available")
                    continue
                    
                print("\n📋 Available RL Models:")
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
            
            elif choice == '2':
                self._train_new_rl_model()
                
            elif choice == '3':
                # Model evaluation
                model_name = self.rl_manager.interactive_model_selection()
                if model_name is None or model_name == "heuristic":
                    continue
                    
                # Get config for evaluation
                print("\nSelect configuration for evaluation:")
                configs = list_available_configs()
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                
                selection = input("\nEnter configuration name or number: ").strip()
                try:
                    if selection.isdigit():
                        config_idx = int(selection) - 1
                        if 0 <= config_idx < len(configs):
                            config_name = configs[config_idx]
                        else:
                            print("❌ Invalid selection")
                            continue
                    else:
                        config_name = selection
                    
                    # Run evaluation
                    print(f"🔍 Evaluating model '{model_name}' on {config_name}...")
                    self.orchestration_mode = "rl"
                    self.selected_rl_model = model_name
                    network = self.run_simulation(config_name)
                    self.print_summary(network)
                    
                    # Option to plot results
                    if input("\nShow plots? (y/n): ").lower().startswith('y'):
                        self.plot_results(network)
                        
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    continue
                
            elif choice == '0':
                return
                
            else:
                print("❌ Invalid choice. Please enter 0-3")
    
    def run_simulation(self, config_name: str = "medium_demo", epochs: int = None):
        """Run simulation using the specified configuration.
        
        Args:
            config_name: Name of the configuration to use
            epochs: Number of epochs to run (overrides config default if provided)
        """
        print(f"🚀 SAGIN SIMULATION: {config_name.upper()}")
        print("="*60)
        
        # Start timing
        self._simulation_start_time = time.time()
        
        # Create network with configuration
        network = self.create_network(config_name)
        config = self.current_config
        
        # Set orchestration mode on the network
        if self.orchestration_mode == "rl" and self.selected_rl_model and self.rl_manager:
            try:
                central_agent, shared_static_uav_agent = self.rl_manager.load_model(self.selected_rl_model, network)
                network.set_rl_orchestration(central_agent, shared_static_uav_agent)
                print(f"🤖 Using RL orchestration with model: {self.selected_rl_model}")
            except Exception as e:
                print(f"❌ Failed to load RL model '{self.selected_rl_model}': {e}")
                print("🔧 Falling back to heuristic orchestration")
                network.set_heuristic_orchestration()
        else:
            network.set_heuristic_orchestration()
            print("🔧 Using heuristic orchestration")
        
        # Configure logging based on config
        if config.simulation.logging_level in ["medium", "high"]:
            network.log_decisions = config.simulation.log_decisions
            network.log_resource_usage = config.simulation.log_resource_usage
        
        network.initialize_simulation()
        
        # Use custom epochs if provided, otherwise use config default
        simulation_epochs = epochs if epochs is not None else config.simulation.total_epochs
        
        print(f"\nRunning {simulation_epochs}-epoch simulation with {config.simulation.logging_level} logging...")
        
        # Track metrics
        self.metrics_history = []
        
        def progress_callback(epoch: int, total_epochs: int, step_results: Dict):
            """Track progress and metrics."""
            # Capture metrics every epoch for smooth plotting curves
            metrics = network.metrics
            self.metrics_history.append({
                'epoch': epoch,
                'success_rate': metrics.success_rate,
                'average_latency': metrics.average_latency,
                'load_imbalance': metrics.load_imbalance,
                'uav_utilization': metrics.uav_utilization,
                'satellite_utilization': metrics.satellite_utilization,
                'tasks_generated': metrics.total_tasks_generated,
                'tasks_completed': metrics.total_tasks_completed,
                'energy_consumption': getattr(metrics, 'energy_consumption', 0.0),
                'coverage_percentage': getattr(metrics, 'coverage_percentage', 100.0)
            })
            
            # Print progress only at intervals to avoid spam
            if epoch % config.simulation.progress_interval == 0:
                print(f"Epoch {epoch:3d}: Success: {metrics.success_rate:.3f}, "
                      f"Latency: {metrics.average_latency:.3f}s, "
                      f"Load: {metrics.load_imbalance:.3f}")
        
        # Run simulation
        if config.simulation.detailed_interval > 0 and config.simulation.logging_level == "high":
            print(f"\n🔍 Running with detailed logging every {config.simulation.detailed_interval} epochs...")
            network.run_simulation_with_detailed_logging(
                simulation_epochs, config.simulation.detailed_interval, progress_callback
            )
        else:
            print(f"\n📊 Running standard simulation...")
            network.run_simulation(simulation_epochs, progress_callback)
        
        # Store simulation time
        self._simulation_time = time.time() - self._simulation_start_time
        
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
        
        # Show task type distribution
        task_stats = network.task_manager.task_generator.get_generation_statistics()
        total_generated = task_stats['total_generated']
        if total_generated > 0:
            print(f"  Task Type Distribution:")
            for task_type, count in task_stats['by_type'].items():
                percentage = (count / total_generated) * 100
                print(f"    - {task_type.value.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Show configured proportions
        current_proportions = network.get_task_type_proportions()
        print(f"  Configured Task Proportions:")
        for task_type, proportion in current_proportions.items():
            print(f"    - {task_type.replace('_', ' ').title()}: {proportion:.1%}")
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
        """Plot comprehensive simulation results with detailed statistics."""
        if not self.metrics_history:
            print("No metrics history to plot")
            return
        
        print(f"📊 Plotting {len(self.metrics_history)} data points from simulation...")
        
        # Extract data from metrics history
        epochs = [m['epoch'] for m in self.metrics_history]
        success_rates = [m['success_rate'] for m in self.metrics_history]
        avg_latencies = [m['average_latency'] for m in self.metrics_history]
        load_imbalances = [m['load_imbalance'] for m in self.metrics_history]
        uav_utilizations = [m['uav_utilization'] for m in self.metrics_history]
        satellite_utilizations = [m['satellite_utilization'] for m in self.metrics_history]
        
        # Create comprehensive figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig.suptitle(f'SAGIN {self.orchestration_mode.upper()} Simulation Results - {self.current_config.name} - {timestamp}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Success Rate with moving average
        ax1 = axes[0, 0]
        ax1.plot(epochs, success_rates, alpha=0.6, color='blue', label='Success Rate')
        
        # Add moving average (window=10)
        if len(success_rates) > 10:
            moving_avg = np.convolve(success_rates, np.ones(10)/10, mode='valid')
            ax1.plot(epochs[9:], moving_avg, color='red', linewidth=2, label='Moving Average (10)')
        
        ax1.set_title('Task Success Rate')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add statistics text
        final_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
        ax1.text(0.02, 0.98, f'Final Avg (100): {final_success:.3f}\nMax: {max(success_rates):.3f}\nMin: {min(success_rates):.3f}', 
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Average Latency
        ax2 = axes[0, 1]
        ax2.plot(epochs, avg_latencies, color='red', linewidth=1.5)
        ax2.set_title('Average Task Latency')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Latency (s)')
        ax2.grid(True, alpha=0.3)
        
        # Add latency statistics
        final_latency = np.mean(avg_latencies[-50:]) if len(avg_latencies) >= 50 else np.mean(avg_latencies)
        ax2.text(0.02, 0.98, f'Final Avg (50): {final_latency:.3f}s\nMax: {max(avg_latencies):.3f}s\nMin: {min(avg_latencies):.3f}s', 
                transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # 3. Load Imbalance
        ax3 = axes[0, 2]
        ax3.plot(epochs, load_imbalances, color='green', linewidth=1.5)
        ax3.set_title('Load Imbalance')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Load Imbalance')
        ax3.grid(True, alpha=0.3)
        
        # Add load imbalance statistics
        final_load = np.mean(load_imbalances[-50:]) if len(load_imbalances) >= 50 else np.mean(load_imbalances)
        ax3.text(0.02, 0.98, f'Final Avg (50): {final_load:.3f}\nMax: {max(load_imbalances):.3f}\nMin: {min(load_imbalances):.3f}', 
                transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 4. Performance Trend Analysis
        ax4 = axes[1, 0]
        if len(success_rates) > 20:
            # Calculate improvement rate over epochs
            window = 20
            improvement = []
            for i in range(window, len(success_rates)):
                recent_avg = np.mean(success_rates[i-window:i])
                past_avg = np.mean(success_rates[max(0, i-2*window):i-window])
                improvement.append(recent_avg - past_avg)
            
            ax4.plot(epochs[window:], improvement, color='purple', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Performance Improvement Trend')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Success Rate Improvement')
            ax4.grid(True, alpha=0.3)
            
            # Add improvement statistics
            recent_trend = np.mean(improvement[-20:]) if len(improvement) >= 20 else np.mean(improvement) if improvement else 0
            positive_epochs = sum(1 for x in improvement if x > 0)
            ax4.text(0.02, 0.98, f'Recent Trend: {recent_trend:.4f}\nPositive Epochs: {positive_epochs}/{len(improvement)}', 
                    transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor trend analysis\n(need >20 epochs)', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Performance Improvement Trend')
        
        # 5. Resource Utilization Comparison
        ax5 = axes[1, 1]
        ax5.plot(epochs, uav_utilizations, label='UAV Utilization', color='magenta', alpha=0.7)
        ax5.plot(epochs, satellite_utilizations, label='Satellite Utilization', color='cyan', alpha=0.7)
        ax5.set_title('Resource Utilization Comparison')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Utilization')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        ax5.legend()
        
        # Calculate utilization efficiency
        final_uav_util = np.mean(uav_utilizations[-50:]) if len(uav_utilizations) >= 50 else np.mean(uav_utilizations)
        final_sat_util = np.mean(satellite_utilizations[-50:]) if len(satellite_utilizations) >= 50 else np.mean(satellite_utilizations)
        ax5.text(0.02, 0.98, f'UAV Util: {final_uav_util:.3f}\nSat Util: {final_sat_util:.3f}\nBalance: {abs(final_uav_util-final_sat_util):.3f}', 
                transform=ax5.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 6. Comprehensive Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')  # Turn off axis for text display
        
        # Calculate comprehensive statistics
        total_epochs = len(self.metrics_history)
        simulation_time = getattr(self, '_simulation_time', 0)
        
        # Performance metrics
        initial_performance = np.mean(success_rates[:10]) if len(success_rates) >= 10 else success_rates[0] if success_rates else 0
        final_performance = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
        improvement = final_performance - initial_performance
        
        # Stability analysis
        if len(success_rates) >= 50:
            stability = 1.0 / (1.0 + np.std(success_rates[-50:]))  # Higher is more stable
        else:
            stability = 0.5
        
        # Performance consistency
        if len(success_rates) >= 20:
            recent_variance = np.var(success_rates[-20:])
            consistency = "High" if recent_variance < 0.01 else "Medium" if recent_variance < 0.05 else "Low"
        else:
            consistency = "Unknown"
        
        # Get final network metrics
        final_metrics = network.get_performance_summary()['final_metrics']
        
        # Task type distribution
        task_stats = network.task_manager.task_generator.get_generation_statistics()
        task_diversity = len(task_stats['by_type']) if task_stats.get('by_type') else 0
        
        summary_text = f"""SIMULATION SUMMARY
════════════════════════
Configuration: {self.current_config.name}
Orchestration: {self.orchestration_mode.upper()}
Total Epochs: {total_epochs}
Simulation Time: {simulation_time:.1f}s
Epochs/sec: {total_epochs/max(simulation_time, 1):.2f}

PERFORMANCE METRICS
═══════════════════════
Initial Success Rate: {initial_performance:.3f}
Final Success Rate: {final_performance:.3f}
Performance Change: {improvement:+.3f}
Stability Score: {stability:.3f}
Consistency: {consistency}

NETWORK PERFORMANCE
══════════════════════
Tasks Generated: {final_metrics.total_tasks_generated}
Tasks Completed: {final_metrics.total_tasks_completed}
Final Success Rate: {final_metrics.success_rate:.1%}
Avg Latency: {final_metrics.average_latency:.3f}s
Load Imbalance: {final_metrics.load_imbalance:.3f}

RESOURCE EFFICIENCY
══════════════════════
UAV Utilization: {final_metrics.uav_utilization:.1%}
Satellite Utilization: {final_metrics.satellite_utilization:.1%}
Coverage: {final_metrics.coverage_percentage:.1f}%
Task Diversity: {task_diversity} types

OPTIMIZATION METRICS
═══════════════════════
Best Success Rate: {max(success_rates):.3f}
Worst Success Rate: {min(success_rates):.3f}
Best Latency: {min(avg_latencies):.3f}s
Success Rate Range: {max(success_rates)-min(success_rates):.3f}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive plot
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        comprehensive_filename = f"{results_dir}/comprehensive_{self.orchestration_mode}_results_{timestamp}.png"
        plt.savefig(comprehensive_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Comprehensive results saved to: {comprehensive_filename}")
        
        plt.show()
        plt.close()  # Close instead of showing
        
        # Print final summary to console
        print(f"\n🎯 FINAL SIMULATION SUMMARY")
        print("="*60)
        print(f"Configuration: {self.current_config.name}")
        print(f"Orchestration: {self.orchestration_mode.upper()}")
        print(f"Total Epochs: {total_epochs}")
        print(f"Simulation Time: {simulation_time:.2f} seconds")
        print(f"Performance Change: {improvement:+.3f} (from {initial_performance:.3f} to {final_performance:.3f})")
        print(f"Final Success Rate: {final_metrics.success_rate:.1%}")
        print(f"Final Average Latency: {final_metrics.average_latency:.3f}s")
        print(f"Resource Utilization: UAV {final_metrics.uav_utilization:.1%}, Satellite {final_metrics.satellite_utilization:.1%}")
        print(f"Stability Score: {stability:.3f}/1.0")
        print(f"Performance Consistency: {consistency}")
        print("="*60)
    
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
        if RL_AVAILABLE:
            print("6. 🤖 RL Management (train/view models)")
        print("0. ❌ Exit")
        print("=" * 60)
        
        try:
            max_choice = "6" if RL_AVAILABLE else "5"
            choice = input(f"\nEnter choice (0-{max_choice}): ").strip()
            
            if choice == '1':
                print_config_summary()
                
            elif choice == '2':
                config_name = input("Enter configuration name: ").strip()
                print_config_summary(config_name)
                
            elif choice == '3':
                # Select orchestration mode first
                if not demo.select_orchestration_mode():
                    continue  # User cancelled orchestration selection
                
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
                    
                    # Ask for number of epochs
                    try:
                        epochs_input = input("Number of simulation epochs (press Enter for config default): ").strip()
                        epochs = int(epochs_input) if epochs_input else None
                    except ValueError:
                        print("Invalid number, using config default")
                        epochs = None
                    
                    network = demo.run_simulation(config_name, epochs)
                    demo.print_summary(network)
                    
                except (ValueError, IndexError):
                    print("Invalid selection")
                    continue
                
            elif choice == '4':
                # Select orchestration mode first
                if not demo.select_orchestration_mode():
                    continue  # User cancelled orchestration selection
                    
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
                        # Use the configuration's total epochs for visualization
                        max_epochs = demo.current_config.simulation.total_epochs
                        visualizer.run(max_epochs=max_epochs)
                    except Exception as e:
                        print(f"Visualization error: {e}")
                        print("Make sure matplotlib is properly installed.")
                    
                except (ValueError, IndexError):
                    print("Invalid selection")
                    continue
                    
            elif choice == '6' and RL_AVAILABLE:
                demo.manage_rl_models()
                
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
