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
        print("2. 📍 Baseline (Fixed UAV Positions)")
        print("3. 🤖 Reinforcement Learning")
        print("4. 🚀 Train New RL Model")
        print("0. ❌ Cancel")
        
        while True:
            try:
                choice = input("\nEnter choice (0-4): ").strip()
                
                if choice == '1':
                    self.orchestration_mode = "heuristic"
                    print("✅ Selected: Heuristic orchestration")
                    return True
                    
                elif choice == '2':
                    self.orchestration_mode = "baseline"
                    print("✅ Selected: Baseline orchestration (fixed UAV positions)")
                    return True
                    
                elif choice == '3':
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
                        
                elif choice == '4':
                    success = self._train_new_rl_model()
                    if success:
                        return self.select_orchestration_mode()  # Re-select after training
                    else:
                        continue
                        
                elif choice == '0':
                    print("❌ Cancelled")
                    return False
                    
                else:
                    print("❌ Invalid choice. Please enter 0-4")
                    
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
        elif self.orchestration_mode == "baseline":
            network.set_baseline_orchestration()
            print("📍 Using baseline orchestration (fixed UAV positions)")
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
        
        # Add warm-up epochs to eliminate initial transients and get stable metrics
        warmup_epochs = 30
        total_epochs_with_warmup = simulation_epochs + warmup_epochs
        
        print(f"\n🔥 Running {total_epochs_with_warmup}-epoch simulation ({warmup_epochs} warm-up + {simulation_epochs} recorded epochs)")
        print(f"   First {warmup_epochs} epochs will be discarded to eliminate initial transients...")
        
        # Track metrics
        self.metrics_history = []
        
        def progress_callback(epoch: int, total_epochs: int, step_results: Dict):
            """Track progress and metrics."""
            # Capture metrics every epoch for smooth plotting curves
            metrics = network.metrics
            
            # Only record metrics AFTER warm-up period
            if epoch > warmup_epochs:
                # Adjust epoch number to start from 1 after warm-up
                recorded_epoch = epoch - warmup_epochs
                self.metrics_history.append({
                    'epoch': recorded_epoch,
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
                if epoch <= warmup_epochs:
                    print(f"Warm-up {epoch:2d}/{warmup_epochs}: Success: {metrics.success_rate:.3f}, "
                          f"Latency: {metrics.average_latency:.3f}s, "
                          f"Load: {metrics.load_imbalance:.3f}")
                else:
                    recorded_epoch = epoch - warmup_epochs
                    print(f"Epoch {recorded_epoch:3d}: Success: {metrics.success_rate:.3f}, "
                          f"Latency: {metrics.average_latency:.3f}s, "
                          f"Load: {metrics.load_imbalance:.3f}")
        
        # Run simulation with warm-up epochs
        if config.simulation.detailed_interval > 0 and config.simulation.logging_level == "high":
            print(f"\n🔍 Running with detailed logging every {config.simulation.detailed_interval} epochs...")
            network.run_simulation_with_detailed_logging(
                total_epochs_with_warmup, config.simulation.detailed_interval, progress_callback
            )
        else:
            print(f"\n📊 Running standard simulation...")
            network.run_simulation(total_epochs_with_warmup, progress_callback)
        
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
        
        # Create optimized figure with comprehensive layout - METRICS + CONFIG + SUMMARY + UAV TIMELINE
        fig = plt.figure(figsize=(24, 16))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig.suptitle(f'SAGIN {self.orchestration_mode.upper()} Performance Analysis - {self.current_config.name} - {timestamp}', 
                    fontsize=20, fontweight='bold', y=0.96)
        
        # Create enhanced grid layout: 2x2 metrics + config panel + UAV timeline
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)
        
        # 1. Success Rate without moving average (top-left)
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax1.plot(epochs, success_rates, alpha=0.8, color='blue', linewidth=2, label='Success Rate')
        
        ax1.set_title('Task Success Rate', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add compact statistics text
        final_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
        ax1.text(0.98, 0.98, f'Final: {final_success:.3f} | Max: {max(success_rates):.3f} | Min: {min(success_rates):.3f}', 
                transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # 2. Average Latency (top-right)
        ax2 = fig.add_subplot(gs[0, 3:6])
        ax2.plot(epochs, avg_latencies, color='red', linewidth=2)
        ax2.set_title('Average Task Latency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Latency (s)')
        ax2.grid(True, alpha=0.3)
        
        # Add compact latency statistics
        final_latency = np.mean(avg_latencies[-50:]) if len(avg_latencies) >= 50 else np.mean(avg_latencies)
        ax2.text(0.98, 0.98, f'Final: {final_latency:.3f}s | Max: {max(avg_latencies):.3f}s | Min: {min(avg_latencies):.3f}s', 
                transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
        
        # 3. Load Imbalance (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0:3])
        ax3.plot(epochs, load_imbalances, color='green', linewidth=2)
        ax3.set_title('Load Imbalance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Load Imbalance')
        ax3.grid(True, alpha=0.3)
        
        # Add compact load imbalance statistics
        final_load = np.mean(load_imbalances[-50:]) if len(load_imbalances) >= 50 else np.mean(load_imbalances)
        ax3.text(0.98, 0.98, f'Final: {final_load:.3f} | Max: {max(load_imbalances):.3f} | Min: {min(load_imbalances):.3f}', 
                transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # 4. Resource Utilization Comparison (bottom-right)
        ax4 = fig.add_subplot(gs[1, 3:6])
        ax4.plot(epochs, uav_utilizations, label='UAV Utilization', color='purple', linewidth=2)
        ax4.plot(epochs, satellite_utilizations, label='Satellite Utilization', color='orange', linewidth=2)
        ax4.set_title('Resource Utilization Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Utilization Rate')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        # Add compact utilization statistics
        final_uav_util = np.mean(uav_utilizations[-50:]) if len(uav_utilizations) >= 50 else np.mean(uav_utilizations)
        final_sat_util = np.mean(satellite_utilizations[-50:]) if len(satellite_utilizations) >= 50 else np.mean(satellite_utilizations)
        ax4.text(0.98, 0.98, f'UAV: {final_uav_util:.1%} | Satellite: {final_sat_util:.1%}', 
                transform=ax4.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontsize=10)
        
        # 5. Detailed Network Configuration Panel (left side)
        ax5 = fig.add_subplot(gs[2, 0:2])
        try:
            # Get comprehensive network configuration details
            network_state = network.get_network_state()
            performance = network.get_performance_summary()
            config = self.current_config
            
            # Create compact configuration table
            config_text = f"""📋 NETWORK CONFIG
{'='*25}
🌐 TOPOLOGY
Grid: {config.grid.grid_rows}×{config.grid.grid_cols} ({config.grid.total_regions} regions)
Area: {config.grid.area_bounds[1]/1000:.1f}×{config.grid.area_bounds[3]/1000:.1f}km

🚗 VEHICLES ({len(network.vehicle_manager.vehicles)} total)
Random: {config.vehicles.random_vehicles}
Buses: {config.vehicles.bus_vehicles}
Density: {len(network.vehicle_manager.vehicles)/config.grid.total_regions:.1f}/region

🚁 UAVs ({len(network.uav_manager.static_uavs) + len(network.uav_manager.dynamic_uavs)} total)
Static: {len(network.uav_manager.static_uavs)}
Dynamic: {len(network.uav_manager.dynamic_uavs)}

🛰️ SATELLITES
Count: {len(network.satellite_constellation.satellites)}
Delay: {getattr(config.satellites, 'processing_delay', 'N/A')}s

⚙️ SIMULATION
Mode: {self.orchestration_mode.upper()}
Epochs: {config.simulation.total_epochs}
Log: {config.simulation.logging_level.upper()}
Bursts: {len(config.tasks.burst_events) if config.tasks.burst_events else 0}"""

            # Add RL-specific configuration if in RL mode
            if self.orchestration_mode == 'rl' and hasattr(config, 'rl'):
                rl_config_text = f"""

🧠 RL TRAINING CONFIG
{'='*25}
Learning Rate: {getattr(config.rl, 'learning_rate', 'N/A')}
Batch Size: {getattr(config.rl, 'batch_size', 'N/A')}
Gamma: {getattr(config.rl, 'gamma', 'N/A')}
Epsilon Start: {getattr(config.rl, 'epsilon_start', 'N/A')}
Epsilon Decay: {getattr(config.rl, 'epsilon_decay', 'N/A')}
Buffer Size: {getattr(config.rl, 'buffer_size', 'N/A')}
Hidden Dim: {getattr(config.rl, 'hidden_dim', 'N/A')}
Actor-Critic: {getattr(config.rl, 'use_actor_critic', 'N/A')}"""
                config_text += rl_config_text
            
            # Display the configuration with better positioning
            ax5.text(0.02, 0.98, config_text, 
                    transform=ax5.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontsize=8, fontfamily='monospace', wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
            
            ax5.set_title('📊 Network Configuration Details', fontsize=12, fontweight='bold')
            ax5.axis('off')
                
        except Exception as e:
            # Fallback display
            ax5.text(0.5, 0.5, f'Network Configuration\nDetails\n\nError: {str(e)[:50]}...', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax5.set_title('📊 Network Configuration - Error', fontsize=12, fontweight='bold')
            ax5.axis('off')
        
        # 6. UAV Repositioning Timeline (center-wide)
        ax6 = fig.add_subplot(gs[2, 2:6])
        try:
            # Show UAV repositioning decisions as a timeline visualization
            dynamic_uavs = network.uav_manager.dynamic_uavs
            
            if dynamic_uavs and len(epochs) > 0:
                uav_ids = list(dynamic_uavs.keys())
                
                # Collect all repositioning events in chronological order
                repositioning_events = []
                
                # Check if UAV manager has repositioning history
                if hasattr(network.uav_manager, 'repositioning_history'):
                    for uav_id in uav_ids:
                        uav_history = network.uav_manager.repositioning_history.get(uav_id, {})
                        for epoch, data in uav_history.items():
                            region_id = data['region_id']
                            status = data.get('status', 'hovering')
                            if status == 'moving':
                                from_region = data.get('old_region_id', '?')
                                repositioning_events.append((epoch, uav_id, from_region, region_id))
                
                # Sort events by epoch
                repositioning_events.sort(key=lambda x: x[0])
                
                if repositioning_events:
                    # Create visual timeline 
                    epochs_with_events = [event[0] for event in repositioning_events]
                    uav_positions = [event[1] for event in repositioning_events]
                    
                    # Plot timeline as scatter plot
                    colors = plt.cm.Set3(np.linspace(0, 1, len(set(uav_positions))))
                    uav_color_map = {uav_id: colors[i] for i, uav_id in enumerate(set(uav_positions))}
                    
                    for i, (epoch, uav_id, from_region, to_region) in enumerate(repositioning_events):
                        color = uav_color_map[uav_id]
                        ax6.scatter(epoch, uav_id, s=100, c=[color], alpha=0.8, 
                                   label=f'UAV{uav_id}' if i == 0 or uav_id != repositioning_events[i-1][1] else "")
                        
                        # Add arrow annotation for movement
                        ax6.annotate(f'R{from_region}→R{to_region}', 
                                   xy=(epoch, uav_id), xytext=(epoch, uav_id + 0.3),
                                   ha='center', va='bottom', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                    
                    ax6.set_title('🚁 UAV Repositioning Timeline - Chronological View', fontsize=12, fontweight='bold')
                    ax6.set_xlabel('Epoch')
                    ax6.set_ylabel('UAV ID')
                    ax6.grid(True, alpha=0.3)
                    ax6.set_xlim(min(epochs_with_events) - 1, max(epochs_with_events) + 1)
                    
                    # Add summary text
                    summary_text = f"Total Events: {len(repositioning_events)} | "
                    summary_text += f"UAVs Active: {len(set(uav_positions))} | "
                    summary_text += f"Avg per UAV: {len(repositioning_events)/len(uav_ids):.1f}"
                    
                    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
                    
                else:
                    # No repositioning events
                    ax6.text(0.5, 0.5, '🚁 No UAV repositioning events occurred during simulation\n\n' +
                           'This could indicate:\n• Balanced load distribution\n• Short simulation duration\n• Low task generation rate', 
                           transform=ax6.transAxes, ha='center', va='center', fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax6.set_title('🚁 UAV Repositioning Timeline', fontsize=12, fontweight='bold')
                    ax6.axis('off')
            else:
                # No dynamic UAVs configured
                ax6.text(0.5, 0.5, '🚁 No dynamic UAVs configured in this simulation\n\n' +
                       'Dynamic UAVs are required for repositioning events', 
                       transform=ax6.transAxes, ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax6.set_title('🚁 UAV Repositioning Timeline', fontsize=12, fontweight='bold')
                ax6.axis('off')
                            
        except Exception as e:
            # Error fallback
            ax6.text(0.5, 0.5, f'🚁 UAV Repositioning Timeline\n\nError: {str(e)[:100]}...', 
                    transform=ax6.transAxes, ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax6.set_title('🚁 UAV Repositioning Timeline - Error', fontsize=12, fontweight='bold')
            ax6.axis('off')
        

        # Calculate comprehensive statistics for display
        total_epochs = len(self.metrics_history)
        simulation_time = getattr(self, '_simulation_time', 0)
        
        # Performance metrics
        initial_performance = np.mean(success_rates[:10]) if len(success_rates) >= 10 else success_rates[0] if success_rates else 0
        final_performance = np.mean(success_rates[-10:]) if len(success_rates) >= 10 else np.mean(success_rates)
        improvement = final_performance - initial_performance
        
        # Get final network metrics
        final_metrics = network.get_performance_summary()['final_metrics']
        
        # Add compact summary info as figure text
        summary_info = f"✨ {self.current_config.name} | {self.orchestration_mode.upper()} Mode | "
        summary_info += f"{total_epochs} Epochs ({simulation_time:.1f}s) | "
        summary_info += f"Success: {final_performance:.1%} ({improvement:+.3f}) | "
        summary_info += f"Completion: {(final_metrics.total_tasks_completed/final_metrics.total_tasks_generated*100):.1f}% | "
        summary_info += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        fig.text(0.5, 0.005, summary_info, ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.9))
        
        # Adjust layout for the enhanced grid
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        # Calculate comprehensive and detailed statistics for console output
        total_epochs = len(self.metrics_history)
        simulation_time = getattr(self, '_simulation_time', 0)
        
        # Enhanced performance metrics
        initial_performance = np.mean(success_rates[:10]) if len(success_rates) >= 10 else success_rates[0] if success_rates else 0
        final_performance = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
        improvement = final_performance - initial_performance
        peak_performance = max(success_rates) if success_rates else 0
        
        # Detailed stability analysis
        if len(success_rates) >= 50:
            stability = 1.0 / (1.0 + np.std(success_rates[-50:]))
            performance_trend = "Improving" if improvement > 0.02 else "Declining" if improvement < -0.02 else "Stable"
        else:
            stability = 0.5
            performance_trend = "Unknown"
        
        # Enhanced performance consistency
        if len(success_rates) >= 20:
            recent_variance = np.var(success_rates[-20:])
            consistency = "High" if recent_variance < 0.01 else "Medium" if recent_variance < 0.05 else "Low"
            volatility = np.std(success_rates) / np.mean(success_rates) if np.mean(success_rates) > 0 else 0
        else:
            consistency = "Unknown"
            volatility = 0
        
        # Get final network metrics with additional calculations
        final_metrics = network.get_performance_summary()['final_metrics']
        
        # Enhanced task statistics
        task_stats = network.task_manager.task_generator.get_generation_statistics()
        task_diversity = len(task_stats['by_type']) if task_stats.get('by_type') else 0
        completion_rate = (final_metrics.total_tasks_completed / final_metrics.total_tasks_generated * 100) if final_metrics.total_tasks_generated > 0 else 0
        
        # Resource efficiency calculations
        avg_uav_util = np.mean(uav_utilizations) if uav_utilizations else 0
        avg_sat_util = np.mean(satellite_utilizations) if satellite_utilizations else 0
        resource_balance = abs(avg_uav_util - avg_sat_util)
        
        # Latency analysis
        latency_improvement = min(avg_latencies) - max(avg_latencies) if len(avg_latencies) > 1 else 0
        avg_latency_overall = np.mean(avg_latencies) if avg_latencies else 0
        
        # Load balancing effectiveness
        avg_load_imbalance = np.mean(load_imbalances) if load_imbalances else 0
        load_stability = 1.0 / (1.0 + np.std(load_imbalances)) if load_imbalances else 0
        
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

    def _get_best_rl_model(self):
        """Automatically select the best available RL model based on performance metrics."""
        if not RL_AVAILABLE or not self.rl_manager:
            return None
        
        models = self.rl_manager.list_models()
        if not models:
            return None
        
        # Sort models by success rate if available, otherwise use most recent
        models_with_performance = [m for m in models if 'success_rate' in m.get('performance', {})]
        
        if models_with_performance:
            # Select model with highest success rate
            best_model = max(models_with_performance, 
                           key=lambda m: m['performance']['success_rate'])
            print(f"🎯 Selected best RL model: {best_model['name']} (Success Rate: {best_model['performance']['success_rate']:.1%})")
            return best_model['name']
        else:
            # Fallback to most recent model
            latest_model = models[-1]
            print(f"📅 Selected latest RL model: {latest_model['name']} (no performance data available)")
            return latest_model['name']

    def compare_orchestration_methods(self, config_name: str):
        """Compare all three orchestration methods (baseline, heuristic, RL) on the same configuration."""
        print(f"\n🔬 ORCHESTRATION METHOD COMPARISON")
        print(f"Configuration: {config_name}")
        print("="*60)
        
        # Interactive RL model selection if RL is available
        best_rl_model = None
        if RL_AVAILABLE and self.rl_manager:
            models = self.rl_manager.list_models()
            if models:
                print("\n🤖 Select RL model for comparison:")
                selected_model = self.rl_manager.interactive_model_selection()
                if selected_model and selected_model != "heuristic":
                    best_rl_model = selected_model
                    print(f"✅ Selected RL model: {best_rl_model}")
                else:
                    print("⚠️ No RL model selected, skipping RL comparison")
            else:
                print("⚠️ No RL models available, skipping RL comparison")
        
        # Interactive epochs selection
        config = get_sagin_config(config_name)
        default_epochs = config.simulation.total_epochs
        print(f"\n⏱️ Simulation Duration Configuration")
        print(f"Default epochs for {config_name}: {default_epochs}")
        
        while True:
            try:
                epochs_input = input(f"Enter number of epochs to run (default: {default_epochs}): ").strip()
                if not epochs_input:
                    comparison_epochs = default_epochs
                    break
                else:
                    comparison_epochs = int(epochs_input)
                    if comparison_epochs <= 0:
                        print("❌ Number of epochs must be positive")
                        continue
                    break
            except ValueError:
                print("❌ Invalid input. Please enter a valid number.")
                continue
        
        print(f"✅ Running comparison with {comparison_epochs} epochs")
        
        # Store results for each method
        comparison_results = {
            'baseline': {'metrics_history': [], 'final_metrics': None},
            'heuristic': {'metrics_history': [], 'final_metrics': None},
            'rl': {'metrics_history': [], 'final_metrics': None}
        }
        
        # Determine which methods to run
        methods_to_run = ['baseline', 'heuristic']
        if best_rl_model:
            methods_to_run.append('rl')
        else:
            print("⚠️ Running baseline and heuristic comparison only")
        
        for method in methods_to_run:
            print(f"\n🚀 Running {method.upper()} orchestration...")
            
            # Set orchestration mode and RL model if needed
            self.orchestration_mode = method
            if method == 'rl':
                self.selected_rl_model = best_rl_model
            
            # Clear previous metrics
            self.metrics_history = []
            
            try:
                # Run simulation for this method with specified epochs
                network = self.run_simulation(config_name, epochs=comparison_epochs)
                
                # Store results
                comparison_results[method]['metrics_history'] = self.metrics_history.copy()
                comparison_results[method]['final_metrics'] = network.get_performance_summary()['final_metrics']
                
                # Get final success rate
                if self.metrics_history:
                    final_success = self.metrics_history[-1]['success_rate']
                    print(f"✅ {method.upper()} completed - Final Success Rate: {final_success:.1%}")
                else:
                    print(f"⚠️ {method.upper()} completed - No metrics recorded")
                    
            except Exception as e:
                print(f"❌ {method.upper()} failed: {str(e)}")
                continue
        
        # Create comparison plot
        self.plot_comparison_results(comparison_results, config_name)
        
        return comparison_results

    def plot_comparison_results(self, comparison_results: dict, config_name: str):
        """Plot comparison of all three orchestration methods."""
        print(f"\n📊 Creating comparison visualization...")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig.suptitle(f'SAGIN Orchestration Methods Comparison - {config_name} - {timestamp}', 
                    fontsize=16, fontweight='bold')
        
        # Colors for each method
        colors = {
            'baseline': '#FF8C00',    # Orange
            'heuristic': '#32CD32',   # Green  
            'rl': '#8A2BE2'           # Purple
        }
        
        # Labels for better display
        labels = {
            'baseline': 'Baseline (Fixed)',
            'heuristic': 'Heuristic (Adaptive)', 
            'rl': 'RL (Intelligent)'
        }
        
        # 1. Success Rate Comparison
        for method, data in comparison_results.items():
            if data['metrics_history']:
                epochs = [m['epoch'] for m in data['metrics_history']]
                success_rates = [m['success_rate'] for m in data['metrics_history']]
                ax1.plot(epochs, success_rates, 
                        color=colors[method], linewidth=2.5, 
                        label=labels[method], alpha=0.8)
        
        ax1.set_title('Task Success Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # 2. Average Latency Comparison  
        for method, data in comparison_results.items():
            if data['metrics_history']:
                epochs = [m['epoch'] for m in data['metrics_history']]
                latencies = [m['average_latency'] for m in data['metrics_history']]
                ax2.plot(epochs, latencies,
                        color=colors[method], linewidth=2.5,
                        label=labels[method], alpha=0.8)
        
        ax2.set_title('Average Latency Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Latency (s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Load Imbalance Comparison
        for method, data in comparison_results.items():
            if data['metrics_history']:
                epochs = [m['epoch'] for m in data['metrics_history']]
                load_imbalances = [m['load_imbalance'] for m in data['metrics_history']]
                ax3.plot(epochs, load_imbalances,
                        color=colors[method], linewidth=2.5,
                        label=labels[method], alpha=0.8)
        
        ax3.set_title('Load Imbalance Comparison', fontsize=14, fontweight='bold') 
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Load Imbalance')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Performance Summary Bar Chart
        methods = []
        final_success_rates = []
        final_latencies = []
        
        for method, data in comparison_results.items():
            if data['final_metrics']:
                methods.append(labels[method])
                final_success_rates.append(data['final_metrics'].success_rate)
                final_latencies.append(data['final_metrics'].average_latency)
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            # Success rate bars
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(x - width/2, final_success_rates, width, 
                           label='Success Rate', alpha=0.8, color='lightblue')
            bars2 = ax4_twin.bar(x + width/2, final_latencies, width,
                                label='Avg Latency (s)', alpha=0.8, color='lightcoral')
            
            ax4.set_title('Final Performance Summary', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Orchestration Method')
            ax4.set_ylabel('Success Rate', color='blue')
            ax4_twin.set_ylabel('Average Latency (s)', color='red')
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods, rotation=15)
            ax4.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars1, final_success_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            for bar, value in zip(bars2, final_latencies):
                height = bar.get_height()
                ax4_twin.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
            
            # Legends
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save comparison plot
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        comparison_filename = f"{results_dir}/orchestration_comparison_{config_name}_{timestamp}.png"
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison results saved to: {comparison_filename}")
        
        plt.show()
        
        # Print comparison summary
        print(f"\n🎯 ORCHESTRATION COMPARISON SUMMARY")
        print("="*60)
        for method, data in comparison_results.items():
            if data['final_metrics']:
                metrics = data['final_metrics']
                print(f"{labels[method]:20}: Success {metrics.success_rate:.1%} | "
                      f"Latency {metrics.average_latency:.3f}s | "
                      f"Tasks {metrics.total_tasks_completed}/{metrics.total_tasks_generated}")
        print("="*60)


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
            print("7. 🔬 Compare orchestration methods")
        else:
            print("6. 🔬 Compare orchestration methods")
        print("0. ❌ Exit")
        print("=" * 60)
        
        try:
            max_choice = "7" if RL_AVAILABLE else "6"
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
                
            elif choice == '7' and RL_AVAILABLE:
                # Orchestration methods comparison
                print("\nAvailable configurations:")
                configs = list_available_configs()
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                
                try:
                    selection = input("\nEnter configuration name or number for comparison: ").strip()
                    if selection.isdigit():
                        config_idx = int(selection) - 1
                        if 0 <= config_idx < len(configs):
                            config_name = configs[config_idx]
                        else:
                            print("Invalid selection")
                            continue
                    else:
                        config_name = selection
                    
                    print(f"\n🔬 This will run all three orchestration methods on '{config_name}'")
                    print("   This may take several minutes to complete...")
                    confirm = input("Continue? (y/n): ").strip().lower()
                    
                    if confirm.startswith('y'):
                        comparison_results = demo.compare_orchestration_methods(config_name)
                    else:
                        print("Comparison cancelled.")
                        
                except (ValueError, IndexError):
                    print("Invalid selection")
                    continue
                    
            elif choice == '6' and not RL_AVAILABLE:
                # Orchestration methods comparison (when RL not available)
                print("\nAvailable configurations:")
                configs = list_available_configs()
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                
                try:
                    selection = input("\nEnter configuration name or number for comparison: ").strip()
                    if selection.isdigit():
                        config_idx = int(selection) - 1
                        if 0 <= config_idx < len(configs):
                            config_name = configs[config_idx]
                        else:
                            print("Invalid selection")
                            continue
                    else:
                        config_name = selection
                    
                    print(f"\n🔬 This will run baseline and heuristic methods on '{config_name}'")
                    print("   (RL not available - install torch for full comparison)")
                    print("   This may take several minutes to complete...")
                    confirm = input("Continue? (y/n): ").strip().lower()
                    
                    if confirm.startswith('y'):
                        # Run limited comparison without RL
                        print("⚠️  Running limited comparison (baseline + heuristic only)")
                        comparison_results = demo.compare_orchestration_methods(config_name)
                    else:
                        print("Comparison cancelled.")
                        
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
