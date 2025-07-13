"""
Real-Time Network Topology Visualizer for SAGIN

Features:
- Color-coded grid regions based on load
- Animated vehicle movements
- UAV positions and dynamic coverage zones
- Communication links, color-coded by link quality

Usage:
    visualizer = RealTimeNetworkVisualizer(network)
    visualizer.run()
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.types import NodeType
from src.core.uavs import UAVStatus


class RealTimeNetworkVisualizer:
    def __init__(self, network):
        self.network = network
        
        # Initialize simulation if not already running
        if not network.is_running:
            network.initialize_simulation()
            
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.grid = network.grid
        self.simulation_running = False
        self.current_epoch = 0
        self.max_epochs = 100  # Default, can be configured
        
        # Initialize data structures (no longer needed since we use ax.clear())
        
        # Color schemes
        self.vehicle_colors = {'random': 'blue', 'bus': 'orange'}
        self.vehicle_markers = {'random': 'o', 'bus': 's'}  # circles for cars, squares for buses
        self.vehicle_sizes = {'random': 25, 'bus': 35}     # different sizes
        self.uav_colors = {'static': 'green', 'dynamic': 'red'}
        
        # Setup the plot
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize the plot layout."""
        self.ax.set_xlim(self.grid.area_bounds[0], self.grid.area_bounds[1])
        self.ax.set_ylim(self.grid.area_bounds[2], self.grid.area_bounds[3])
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('SAGIN Real-Time Network Topology')
        self.ax.grid(True, alpha=0.3)

    def _get_region_load(self, region_id):
        """Get normalized load for a region."""
        if region_id in self.network.regions:
            region = self.network.regions[region_id]
            # Try to get static UAV load for this region
            static_uav = None
            for uav in self.network.uav_manager.static_uavs:
                if hasattr(uav, 'assigned_region') and uav.assigned_region == region_id:
                    static_uav = uav
                    break
            
            if static_uav:
                # Normalize by CPU capacity
                load = static_uav.total_workload / static_uav.cpu_capacity
                return min(1.0, load)
        return 0.0

    def _draw_grid(self):
        """Draw the grid regions with color-coded loads."""
        # No need to manually remove patches since we use ax.clear() in _update
        
        for row in range(self.grid.grid_rows):
            for col in range(self.grid.grid_cols):
                region_id = self.grid.get_region_id(row, col)
                load = self._get_region_load(region_id)
                color = plt.cm.Reds(load)
                
                center_x, center_y = self.grid.get_region_center(row, col)
                rect = patches.Rectangle(
                    (center_x - self.grid.region_width/2, center_y - self.grid.region_height/2),
                    self.grid.region_width, self.grid.region_height,
                    linewidth=1, edgecolor='gray', facecolor=color, alpha=0.6
                )
                self.ax.add_patch(rect)
                
                # Add region ID text in top-left corner
                text_x = center_x - self.grid.region_width/2 + 50  # Small offset from left edge
                text_y = center_y + self.grid.region_height/2 - 50  # Small offset from top edge
                self.ax.text(text_x, text_y, str(region_id), 
                           ha='left', va='top', fontsize=8, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    def _draw_vehicles(self):
        """Draw all vehicles with different markers and colors for different types."""
        if not self.network.vehicle_manager.vehicles:
            return
        
        # Separate vehicles by type for different visual representation
        bus_positions = []
        car_positions = []
        
        for vehicle in self.network.vehicle_manager.vehicles.values():
            position = [vehicle.position.x, vehicle.position.y]
            if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'bus':
                bus_positions.append(position)
            else:
                car_positions.append(position)
        
        # Draw cars (random vehicles) as blue circles
        if car_positions:
            car_positions = np.array(car_positions)
            self.ax.scatter(
                car_positions[:, 0], car_positions[:, 1], 
                c='blue', s=25, marker='o', alpha=0.8, 
                label='Cars', zorder=3, edgecolors='darkblue', linewidths=0.5
            )
        
        # Draw buses as orange squares
        if bus_positions:
            bus_positions = np.array(bus_positions)
            self.ax.scatter(
                bus_positions[:, 0], bus_positions[:, 1], 
                c='orange', s=35, marker='s', alpha=0.8, 
                label='Buses', zorder=3, edgecolors='darkorange', linewidths=0.5
            )

    def _draw_uavs(self):
        """Draw UAVs with coverage zones, preventing visual overlap."""
        # No need to manually remove coverage circles since we use ax.clear() in _update
        
        # Draw static UAVs first (lower layer)
        static_positions = []
        for uav in self.network.uav_manager.static_uavs.values():
            static_positions.append([uav.position.x, uav.position.y])
            # Draw coverage zone
            coverage = patches.Circle(
                (uav.position.x, uav.position.y), uav.communication_range,
                color='green', alpha=0.1, zorder=1
            )
            self.ax.add_patch(coverage)
        
        # Plot static UAVs with smaller size and lower z-order
        if static_positions:
            static_positions = np.array(static_positions)
            self.ax.scatter(
                static_positions[:, 0], static_positions[:, 1],
                c='green', s=50, marker='^', label='Static UAVs', 
                zorder=4, edgecolors='black', linewidths=1
            )
        
        # Draw dynamic UAVs second (higher layer) with offsets to prevent overlap
        dynamic_positions = []
        dynamic_colors = []
        dynamic_sizes = []
        
        for uav in self.network.uav_manager.dynamic_uavs.values():
            # Add small visual offset to prevent exact overlap with static UAVs
            offset_x = 15  # Small offset in meters
            offset_y = 15
            
            # Adjust position with offset for visual separation
            visual_x = uav.position.x + offset_x
            visual_y = uav.position.y + offset_y
            dynamic_positions.append([visual_x, visual_y])
            
            # Color and size based on status
            if uav.status.value == "flying":
                dynamic_colors.append('orange')  # Orange for flying UAVs
                dynamic_sizes.append(80)         # Larger size for flying UAVs
            else:
                dynamic_colors.append('red')     # Red for active UAVs
                dynamic_sizes.append(70)         # Standard size for active UAVs
            
            # Only draw coverage zone for available (active) UAVs
            # Use original position (not offset) for coverage zone
            if uav.is_available:
                coverage = patches.Circle(
                    (uav.position.x, uav.position.y), uav.communication_range,
                    color='red', alpha=0.1, zorder=1
                )
                self.ax.add_patch(coverage)
            # Flying UAVs get no coverage zone (they can't communicate)
        
        # Plot dynamic UAVs with higher z-order and variable sizes
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            self.ax.scatter(
                dynamic_positions[:, 0], dynamic_positions[:, 1],
                c=dynamic_colors, s=dynamic_sizes, marker='^', label='Dynamic UAVs', 
                zorder=5, edgecolors='white', linewidths=1.5, alpha=0.9
            )

    def _draw_communication_links(self):
        """Draw communication links between network elements with different colors for UAV types."""
        # No need to manually remove link lines since we use ax.clear() in _update
        
        # Get UAVs separated by type
        static_uavs = [uav for uav in self.network.uav_manager.static_uavs.values() 
                      if uav.is_available]
        dynamic_uavs = [uav for uav in self.network.uav_manager.dynamic_uavs.values() 
                       if uav.is_available and uav.status.value != "flying"]
        
        # Sample some communication links for visualization
        # For performance, we'll only show a subset of active links
        links_drawn = 0
        max_links = 50  # Limit for performance
        
        # Draw vehicle-to-UAV links
        vehicles_list = list(self.network.vehicle_manager.vehicles.values())
        for vehicle in vehicles_list[:20]:  # Sample first 20 vehicles
            if links_drawn >= max_links:
                break
                
            # Find closest static UAV
            closest_static_uav = None
            closest_dynamic_uav = None
            min_static_distance = float('inf')
            min_dynamic_distance = float('inf')
            
            # Check static UAVs
            for uav in static_uavs:
                distance = vehicle.position.distance_to(uav.position)
                if distance < min_static_distance and distance < uav.communication_range:
                    min_static_distance = distance
                    closest_static_uav = uav
            
            # Check dynamic UAVs
            for uav in dynamic_uavs:
                distance = vehicle.position.distance_to(uav.position)
                if distance < min_dynamic_distance and distance < uav.communication_range:
                    min_dynamic_distance = distance
                    closest_dynamic_uav = uav
            
            # Choose the closest UAV overall and draw link with appropriate color
            if closest_static_uav and closest_dynamic_uav:
                if min_static_distance <= min_dynamic_distance:
                    chosen_uav = closest_static_uav
                    chosen_distance = min_static_distance
                    link_color_base = 'green'  # Green tones for static UAV links
                else:
                    chosen_uav = closest_dynamic_uav
                    chosen_distance = min_dynamic_distance
                    link_color_base = 'red'    # Red tones for dynamic UAV links
            elif closest_static_uav:
                chosen_uav = closest_static_uav
                chosen_distance = min_static_distance
                link_color_base = 'green'      # Green tones for static UAV links
            elif closest_dynamic_uav:
                chosen_uav = closest_dynamic_uav
                chosen_distance = min_dynamic_distance
                link_color_base = 'red'        # Red tones for dynamic UAV links
            else:
                continue
            
            # Calculate link quality and apply color based on UAV type
            quality = max(0.1, 1.0 - chosen_distance / chosen_uav.communication_range)
            
            if link_color_base == 'green':
                # Use green colormap for static UAV links
                color = plt.cm.Greens(0.4 + 0.6 * quality)  # Range from light to dark green
            else:
                # Use red colormap for dynamic UAV links  
                color = plt.cm.Reds(0.4 + 0.6 * quality)   # Range from light to dark red
            
            self.ax.plot(
                [vehicle.position.x, chosen_uav.position.x],
                [vehicle.position.y, chosen_uav.position.y],
                color=color, linewidth=1.5, alpha=0.7, zorder=2
            )
            links_drawn += 1

    def _update_network_state(self):
        """Update the network by one simulation step."""
        if self.simulation_running and self.current_epoch < self.max_epochs:
            # Run one simulation step
            try:
                # Show repositioning logs every 10 epochs (when repositioning happens)
                verbose_mode = (self.current_epoch % 10 == 9)  # Epoch 10, 20, 30, etc.
                step_results = self.network.step(verbose=verbose_mode)
                self.current_epoch += 1
                
                # Debug: Log dynamic UAV positions and status
                if self.current_epoch % 10 == 0:  # Log every 10 epochs
                    print(f"\n🚁 Dynamic UAV Status (Epoch {self.current_epoch}):")
                    for uav_id, uav in self.network.uav_manager.dynamic_uavs.items():
                        if hasattr(uav, 'target_region_id') and uav.target_region_id is not None:
                            target_info = f" → {uav.target_region_id}"
                        else:
                            target_info = ""
                        print(f"  UAV {uav_id}: pos=({uav.position.x:.1f}, {uav.position.y:.1f}), "
                              f"status={uav.status.value}, available={uav.is_available}, "
                              f"region={uav.assigned_region_id}{target_info}, "
                              f"energy={uav.current_energy:.1f}")
                    
                    # Show total UAV count
                    total_dynamic = len(self.network.uav_manager.dynamic_uavs)
                    available_dynamic = len([uav for uav in self.network.uav_manager.dynamic_uavs.values() 
                                           if uav.status == UAVStatus.ACTIVE])
                    flying_dynamic = len([uav for uav in self.network.uav_manager.dynamic_uavs.values() 
                                        if uav.status == UAVStatus.FLYING])
                    print(f"  Total: {total_dynamic}, Active: {available_dynamic}, Flying: {flying_dynamic}")
                
                # Update title with current epoch
                self.ax.set_title(f'SAGIN Real-Time Network - Epoch {self.current_epoch}/{self.max_epochs}')
                
            except Exception as e:
                print(f"Simulation step error: {e}")
                self.simulation_running = False

    def _update(self, frame):
        """Animation update function."""
        # Clear the plot
        self.ax.clear()
        
        # Update network state
        if self.simulation_running:
            self._update_network_state()
        
        # Redraw all elements
        self._setup_plot()
        self._draw_grid()
        self._draw_vehicles()
        self._draw_uavs()
        self._draw_communication_links()
        
        # Add enhanced legend with vehicle types and communication link info
        legend_elements = []
        
        # Vehicle types
        if len(self.network.vehicle_manager.vehicles) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Cars', markeredgecolor='darkblue'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=6, label='Buses', markeredgecolor='darkorange'))
        
        # UAV types
        if len(self.network.uav_manager.static_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Static UAVs'))
        if len(self.network.uav_manager.dynamic_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Dynamic UAVs (Active)'))
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='Dynamic UAVs (Flying)'))
        
        # Communication link types
        legend_elements.append(plt.Line2D([0], [0], color='green', linewidth=2, label='Links to Static UAVs'))
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Links to Dynamic UAVs'))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
        
        # Add metrics text
        if hasattr(self.network, 'metrics'):
            metrics_text = (
                f"Success Rate: {self.network.metrics.success_rate:.3f}\n"
                f"Avg Latency: {self.network.metrics.average_latency:.3f}s\n"
                f"UAV Util: {self.network.metrics.uav_utilization:.3f}\n"
                f"Load Imbal: {self.network.metrics.load_imbalance:.3f}"
            )
            self.ax.text(0.02, 0.98, metrics_text, transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def run(self, interval=1000, enable_simulation=True, max_epochs=100):
        """
        Run the real-time visualization.
        
        Args:
            interval: Update interval in milliseconds
            enable_simulation: Whether to run simulation steps
            max_epochs: Maximum number of simulation epochs
        """
        self.simulation_running = enable_simulation
        self.max_epochs = max_epochs
        
        print(f"🎨 Starting real-time visualization...")
        if enable_simulation:
            print(f"   Running simulation for {max_epochs} epochs")
        else:
            print("   Static visualization mode")
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, self._update, 
            interval=interval, repeat=False
        )
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        return ani
