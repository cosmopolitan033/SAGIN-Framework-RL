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


class RealTimeNetworkVisualizer:
    def __init__(self, network):
        self.network = network
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.grid = network.grid
        self.simulation_running = False
        self.current_epoch = 0
        self.max_epochs = 100  # Default, can be configured
        
        # Initialize data structures (no longer needed since we use ax.clear())
        
        # Color schemes
        self.vehicle_colors = {'random': 'blue', 'bus': 'orange'}
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
        
        # Add colorbar for region loads
        self.cbar_ax = self.fig.add_axes([0.92, 0.1, 0.02, 0.8])
        self.cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), cax=self.cbar_ax)
        self.cbar.set_label('Region Load (Normalized)', rotation=270, labelpad=20)

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
        """Draw all vehicles with different colors for different types."""
        if not self.network.vehicle_manager.vehicles:
            return
            
        positions = []
        colors = []
        
        for vehicle in self.network.vehicle_manager.vehicles.values():
            positions.append([vehicle.position.x, vehicle.position.y])
            # Color by vehicle type
            if hasattr(vehicle, 'vehicle_type'):
                colors.append(self.vehicle_colors.get(vehicle.vehicle_type, 'blue'))
            else:
                colors.append('blue')
        
        positions = np.array(positions)
        
        # No need to remove previous scatter since we use ax.clear() in _update
        self.ax.scatter(
            positions[:, 0], positions[:, 1], 
            c=colors, s=25, alpha=0.8, label='Vehicles', zorder=3
        )

    def _draw_uavs(self):
        """Draw UAVs with coverage zones."""
        # No need to manually remove coverage circles since we use ax.clear() in _update
        
        # Draw static UAVs
        static_positions = []
        for uav in self.network.uav_manager.static_uavs.values():
            static_positions.append([uav.position.x, uav.position.y])
            # Draw coverage zone
            coverage = patches.Circle(
                (uav.position.x, uav.position.y), uav.communication_range,
                color='green', alpha=0.1, zorder=1
            )
            self.ax.add_patch(coverage)
        
        # Draw dynamic UAVs
        dynamic_positions = []
        for uav in self.network.uav_manager.dynamic_uavs.values():
            if uav.is_available:  # Use is_available property instead of is_active
                dynamic_positions.append([uav.position.x, uav.position.y])
                # Draw coverage zone
                coverage = patches.Circle(
                    (uav.position.x, uav.position.y), uav.communication_range,
                    color='red', alpha=0.1, zorder=1
                )
                self.ax.add_patch(coverage)
        
        # Plot static UAVs
        if static_positions:
            static_positions = np.array(static_positions)
            self.ax.scatter(
                static_positions[:, 0], static_positions[:, 1],
                c='green', s=60, marker='^', label='Static UAVs', 
                zorder=4, edgecolors='black'
            )
        
        # Plot dynamic UAVs
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            self.ax.scatter(
                dynamic_positions[:, 0], dynamic_positions[:, 1],
                c='red', s=60, marker='^', label='Dynamic UAVs', 
                zorder=4, edgecolors='black'
            )

    def _draw_communication_links(self):
        """Draw communication links between network elements."""
        # No need to manually remove link lines since we use ax.clear() in _update
        
        # Sample some communication links for visualization
        # For performance, we'll only show a subset of active links
        links_drawn = 0
        max_links = 50  # Limit for performance
        
        # Draw some vehicle-to-UAV links
        vehicles_list = list(self.network.vehicle_manager.vehicles.values())
        for vehicle in vehicles_list[:20]:  # Sample first 20 vehicles
            if links_drawn >= max_links:
                break
                
            # Find closest UAV
            closest_uav = None
            min_distance = float('inf')
            
            for uav in list(self.network.uav_manager.static_uavs.values()) + list(self.network.uav_manager.dynamic_uavs.values()):
                if not uav.is_available:
                    continue
                distance = vehicle.position.distance_to(uav.position)
                if distance < min_distance and distance < uav.communication_range:
                    min_distance = distance
                    closest_uav = uav
            
            if closest_uav:
                # Calculate link quality (simplified)
                quality = max(0.1, 1.0 - min_distance / closest_uav.communication_range)
                color = plt.cm.viridis(quality)
                
                self.ax.plot(
                    [vehicle.position.x, closest_uav.position.x],
                    [vehicle.position.y, closest_uav.position.y],
                    color=color, linewidth=1, alpha=0.5, zorder=2
                )
                links_drawn += 1

    def _update_network_state(self):
        """Update the network by one simulation step."""
        if self.simulation_running and self.current_epoch < self.max_epochs:
            # Run one simulation step
            try:
                step_results = self.network.step(verbose=False)
                self.current_epoch += 1
                
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
        
        # Add legend
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
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
