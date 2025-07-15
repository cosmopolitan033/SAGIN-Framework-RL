"""
Real-Time 3D Network Topology Visualizer for SAGIN

Features:
- 3D visualization showing spatial relationships:
  * Satellites at high altitude (z-axis top)
  * UAVs at medium altitude (z-axis middle)
  * Vehicles at ground level (z-axis bottom)
- Color-coded grid regions based on load
- Animated vehicle movements
- UAV positions and coverage zones
- Communication links in 3D space

Usage:
    visualizer = RealTimeNetworkVisualizer(network)
    visualizer.run()
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
            
        # Create 3D plot
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.grid = network.grid
        self.simulation_running = False
        self.current_epoch = 0
        
        # Auto-detect max epochs from network's system parameters
        self.max_epochs = getattr(network.system_params, 'total_epochs', 100)  # Default fallback to 100
        
        # Define altitude levels for 3D visualization
        self.ground_level = 0      # Vehicles at ground level
        self.uav_level = 150       # UAVs at 150m altitude
        self.satellite_level = 550 # Satellites at 550km (scaled down for visualization)
        
        # Color schemes
        self.vehicle_colors = {'random': 'blue', 'bus': 'orange'}
        self.vehicle_markers = {'random': 'o', 'bus': 's'}  # circles for cars, squares for buses
        self.vehicle_sizes = {'random': 25, 'bus': 35}     # different sizes
        self.uav_colors = {'static': 'green', 'dynamic': 'red'}
        
        # Setup the plot
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize the 3D plot layout."""
        self.ax.set_xlim(self.grid.area_bounds[0], self.grid.area_bounds[1])
        self.ax.set_ylim(self.grid.area_bounds[2], self.grid.area_bounds[3])
        self.ax.set_zlim(0, self.satellite_level + 50)  # Z-axis from ground to above satellites
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Altitude (meters)')
        self.ax.set_title('SAGIN Real-Time 3D Network Topology')
        
        # Set viewing angle for better 3D perspective
        self.ax.view_init(elev=25, azim=45)
        
        # Add grid
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
        """Draw the grid regions at ground level with color-coded loads."""        
        for row in range(self.grid.grid_rows):
            for col in range(self.grid.grid_cols):
                region_id = self.grid.get_region_id(row, col)
                load = self._get_region_load(region_id)
                color = plt.cm.Reds(load)
                
                center_x, center_y = self.grid.get_region_center(row, col)
                
                # Create a 3D rectangular patch at ground level
                x_coords = [center_x - self.grid.region_width/2, center_x + self.grid.region_width/2,
                           center_x + self.grid.region_width/2, center_x - self.grid.region_width/2]
                y_coords = [center_y - self.grid.region_height/2, center_y - self.grid.region_height/2,
                           center_y + self.grid.region_height/2, center_y + self.grid.region_height/2]
                z_coords = [self.ground_level] * 4
                
                # Create vertices for the region rectangle
                vertices = list(zip(x_coords, y_coords, z_coords))
                
                # Draw the region as a colored surface
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly = Poly3DCollection([vertices], alpha=0.6, facecolor=color, edgecolor='gray', linewidth=0.5)
                self.ax.add_collection3d(poly)
                
                # Add region ID text at ground level
                text_x = center_x - self.grid.region_width/2 + 50  
                text_y = center_y + self.grid.region_height/2 - 50  
                self.ax.text(text_x, text_y, self.ground_level + 5, str(region_id), 
                           fontsize=8, alpha=0.8)

    def _draw_vehicles(self):
        """Draw all vehicles at ground level with different markers and colors for different types."""
        if not self.network.vehicle_manager.vehicles:
            return
        
        # Separate vehicles by type for different visual representation
        bus_positions = []
        car_positions = []
        
        for vehicle in self.network.vehicle_manager.vehicles.values():
            position = [vehicle.position.x, vehicle.position.y, self.ground_level]
            if hasattr(vehicle, 'vehicle_type') and vehicle.vehicle_type == 'bus':
                bus_positions.append(position)
            else:
                car_positions.append(position)
        
        # Draw cars (random vehicles) as blue circles at ground level
        if car_positions:
            car_positions = np.array(car_positions)
            self.ax.scatter(
                car_positions[:, 0], car_positions[:, 1], car_positions[:, 2],
                c='blue', s=25, marker='o', alpha=0.8, 
                label='Cars', depthshade=True, edgecolors='darkblue', linewidths=0.5
            )
        
        # Draw buses as orange squares at ground level
        if bus_positions:
            bus_positions = np.array(bus_positions)
            self.ax.scatter(
                bus_positions[:, 0], bus_positions[:, 1], bus_positions[:, 2],
                c='orange', s=35, marker='s', alpha=0.8, 
                label='Buses', depthshade=True, edgecolors='darkorange', linewidths=0.5
            )

    def _draw_uavs(self):
        """Draw UAVs at medium altitude with simplified coverage indication."""
        # Draw static UAVs first
        static_positions = []
        for uav in self.network.uav_manager.static_uavs.values():
            static_positions.append([uav.position.x, uav.position.y, self.uav_level])
            
            # Draw a simple vertical line to show coverage projection to ground
            self.ax.plot(
                [uav.position.x, uav.position.x],
                [uav.position.y, uav.position.y],
                [self.ground_level, self.uav_level],
                color='green', alpha=0.3, linewidth=2
            )
        
        # Plot static UAVs at UAV altitude
        if static_positions:
            static_positions = np.array(static_positions)
            self.ax.scatter(
                static_positions[:, 0], static_positions[:, 1], static_positions[:, 2],
                c='green', s=80, marker='^', label='Static UAVs', 
                depthshade=True, edgecolors='black', linewidths=1
            )
        
        # Draw dynamic UAVs at UAV altitude
        dynamic_positions = []
        dynamic_colors = []
        dynamic_sizes = []
        
        for uav in self.network.uav_manager.dynamic_uavs.values():
            dynamic_positions.append([uav.position.x, uav.position.y, self.uav_level])
            
            # Color and size based on status
            if uav.status.value == "flying":
                dynamic_colors.append('orange')  # Orange for flying UAVs
                dynamic_sizes.append(120)        # Larger size for flying UAVs
            else:
                dynamic_colors.append('red')     # Red for active UAVs
                dynamic_sizes.append(100)        # Standard size for active UAVs
            
            # Only draw coverage line for available (active) UAVs
            if uav.is_available:
                self.ax.plot(
                    [uav.position.x, uav.position.x],
                    [uav.position.y, uav.position.y],
                    [self.ground_level, self.uav_level],
                    color='red', alpha=0.3, linewidth=2
                )
        
        # Plot dynamic UAVs at UAV altitude
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            self.ax.scatter(
                dynamic_positions[:, 0], dynamic_positions[:, 1], dynamic_positions[:, 2],
                c=dynamic_colors, s=dynamic_sizes, marker='^', label='Dynamic UAVs', 
                depthshade=True, edgecolors='white', linewidths=1.5, alpha=0.9
            )
            self.ax.scatter(
                dynamic_positions[:, 0], dynamic_positions[:, 1],
                c=dynamic_colors, s=dynamic_sizes, marker='^', label='Dynamic UAVs', 
                zorder=5, edgecolors='white', linewidths=1.5, alpha=0.9
            )

    def _draw_communication_links(self):
        """Draw 3D communication links between network elements with different colors for UAV types."""
        # Get UAVs separated by type
        static_uavs = [uav for uav in self.network.uav_manager.static_uavs.values() 
                      if uav.is_available]
        dynamic_uavs = [uav for uav in self.network.uav_manager.dynamic_uavs.values() 
                       if uav.is_available and uav.status.value != "flying"]
        
        # Sample some communication links for visualization
        # For performance, we'll only show a subset of active links
        links_drawn = 0
        max_links = 30  # Reduced for 3D performance
        
        # Draw vehicle-to-UAV links
        vehicles_list = list(self.network.vehicle_manager.vehicles.values())
        for vehicle in vehicles_list[::2]:  # Sample every 2nd vehicle
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
            
            # Choose the closest UAV overall and draw 3D link with appropriate color
            chosen_uav = None
            chosen_distance = 0
            link_color_base = None
            
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
            
            # Draw 3D line from vehicle (ground) to UAV (altitude)
            self.ax.plot(
                [vehicle.position.x, chosen_uav.position.x],
                [vehicle.position.y, chosen_uav.position.y],
                [self.ground_level, self.uav_level],
                color=color, linewidth=1.5, alpha=0.7
            )
            links_drawn += 1

    def _draw_satellites(self):
        """Draw satellites at high altitude as simple points without coverage zones."""
        if not hasattr(self.network, 'satellite_constellation') or not self.network.satellite_constellation.satellites:
            return
        
        satellite_positions = []
        satellite_colors = []
        satellite_sizes = []
        
        # Get area bounds for satellite positioning
        min_x, max_x, min_y, max_y = self.grid.area_bounds
        area_center_x = (min_x + max_x) / 2
        area_center_y = (min_y + max_y) / 2
        area_width = max_x - min_x
        area_height = max_y - min_y
        
        # Position satellites in a distributed pattern above the simulation area
        for sat_id, satellite in self.network.satellite_constellation.satellites.items():
            # Create a distributed pattern for satellites
            num_satellites = len(self.network.satellite_constellation.satellites)
            
            if num_satellites == 1:
                # Single satellite at center
                sat_x, sat_y = area_center_x, area_center_y
            elif num_satellites <= 4:
                # Arrange in corners/edges for small numbers
                positions = [
                    (area_center_x - area_width * 0.3, area_center_y - area_height * 0.3),
                    (area_center_x + area_width * 0.3, area_center_y - area_height * 0.3),
                    (area_center_x - area_width * 0.3, area_center_y + area_height * 0.3),
                    (area_center_x + area_width * 0.3, area_center_y + area_height * 0.3)
                ]
                sat_x, sat_y = positions[sat_id % len(positions)]
            else:
                # Arrange in a circle for larger numbers
                angle = (sat_id * 2 * np.pi) / num_satellites
                orbit_radius = min(area_width, area_height) * 0.4
                sat_x = area_center_x + orbit_radius * np.cos(angle)
                sat_y = area_center_y + orbit_radius * np.sin(angle)
            
            satellite_positions.append([sat_x, sat_y, self.satellite_level])
            
            # Color based on satellite utilization
            utilization = min(1.0, satellite.current_workload / satellite.cpu_capacity)
            if utilization < 0.3:
                satellite_colors.append('lightblue')     # Low utilization
                satellite_sizes.append(100)
            elif utilization < 0.7:
                satellite_colors.append('skyblue')       # Medium utilization  
                satellite_sizes.append(120)
            else:
                satellite_colors.append('darkblue')      # High utilization
                satellite_sizes.append(140)
            
            # Add satellite ID label at satellite level
            self.ax.text(sat_x, sat_y, self.satellite_level + 20, f'SAT-{sat_id}', 
                        fontsize=9, alpha=0.9, ha='center')
        
        # Plot satellites at high altitude
        if satellite_positions:
            satellite_positions = np.array(satellite_positions)
            self.ax.scatter(
                satellite_positions[:, 0], satellite_positions[:, 1], satellite_positions[:, 2],
                c=satellite_colors, s=satellite_sizes, marker='*', 
                label='Satellites', depthshade=True, edgecolors='navy', linewidths=2, alpha=0.9
            )

    def _draw_satellite_links(self):
        """Draw simplified 3D communication links from UAVs to satellites."""
        if not hasattr(self.network, 'satellite_constellation') or not self.network.satellite_constellation.satellites:
            return
        
        # Sample some satellite links for visualization (limited for 3D performance)
        links_drawn = 0
        max_satellite_links = 10  # Reduced for 3D clarity
        
        # Get satellite positions (same calculation as in _draw_satellites)
        min_x, max_x, min_y, max_y = self.grid.area_bounds
        area_center_x = (min_x + max_x) / 2
        area_center_y = (min_y + max_y) / 2
        area_width = max_x - min_x
        area_height = max_y - min_y
        
        satellite_positions = {}
        for sat_id, satellite in self.network.satellite_constellation.satellites.items():
            num_satellites = len(self.network.satellite_constellation.satellites)
            
            if num_satellites == 1:
                sat_x, sat_y = area_center_x, area_center_y
            elif num_satellites <= 4:
                positions = [
                    (area_center_x - area_width * 0.3, area_center_y - area_height * 0.3),
                    (area_center_x + area_width * 0.3, area_center_y - area_height * 0.3),
                    (area_center_x - area_width * 0.3, area_center_y + area_height * 0.3),
                    (area_center_x + area_width * 0.3, area_center_y + area_height * 0.3)
                ]
                sat_x, sat_y = positions[sat_id % len(positions)]
            else:
                angle = (sat_id * 2 * np.pi) / num_satellites
                orbit_radius = min(area_width, area_height) * 0.4
                sat_x = area_center_x + orbit_radius * np.cos(angle)
                sat_y = area_center_y + orbit_radius * np.sin(angle)
                
            satellite_positions[sat_id] = (sat_x, sat_y)
        
        # Draw some UAV-to-satellite links (since satellites primarily communicate with UAVs)
        all_uavs = list(self.network.uav_manager.static_uavs.values()) + \
                   [uav for uav in self.network.uav_manager.dynamic_uavs.values() if uav.is_available]
        
        for i, uav in enumerate(all_uavs[::2]):  # Sample every 2nd UAV
            if links_drawn >= max_satellite_links:
                break
            
            # Choose nearest satellite conceptually
            best_sat_id = None
            min_distance = float('inf')
            
            for sat_id, (sat_x, sat_y) in satellite_positions.items():
                distance = np.sqrt((uav.position.x - sat_x)**2 + (uav.position.y - sat_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_sat_id = sat_id
            
            if best_sat_id is not None:
                sat_x, sat_y = satellite_positions[best_sat_id]
                
                # Draw 3D line from UAV to satellite
                self.ax.plot(
                    [uav.position.x, sat_x],
                    [uav.position.y, sat_y],
                    [self.uav_level, self.satellite_level],
                    color='cyan', linewidth=1, alpha=0.5, linestyle=':'
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
        """Animation update function for 3D visualization."""
        # Clear the plot
        self.ax.clear()
        
        # Update network state
        if self.simulation_running:
            self._update_network_state()
        
        # Redraw all elements in proper order
        self._setup_plot()
        self._draw_grid()           # Ground plane with region loads
        self._draw_vehicles()       # Vehicles at ground level
        self._draw_uavs()          # UAVs at medium altitude
        self._draw_satellites()     # Satellites at high altitude
        self._draw_communication_links()  # Vehicle-UAV links
        self._draw_satellite_links()       # UAV-satellite links
        
        # Add enhanced legend for 3D visualization
        legend_elements = []
        
        # Vehicle types (ground level)
        if len(self.network.vehicle_manager.vehicles) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Cars (Ground)', markeredgecolor='darkblue'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=6, label='Buses (Ground)', markeredgecolor='darkorange'))
        
        # UAV types (medium altitude)
        if len(self.network.uav_manager.static_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='Static UAVs (150m)'))
        if len(self.network.uav_manager.dynamic_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Dynamic UAVs (Active)'))
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='Dynamic UAVs (Flying)'))
        
        # Communication link types
        legend_elements.append(plt.Line2D([0], [0], color='green', linewidth=2, label='Links to Static UAVs'))
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Links to Dynamic UAVs'))
        
        # Satellite elements (high altitude)
        if hasattr(self.network, 'satellite_constellation') and self.network.satellite_constellation.satellites:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lightblue', markersize=10, label='Satellites (550km)', markeredgecolor='navy'))
            legend_elements.append(plt.Line2D([0], [0], color='cyan', linewidth=1, linestyle=':', label='UAV-Satellite Links'))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=9)
        
        # Add metrics text in 3D-compatible location
        if hasattr(self.network, 'metrics'):
            metrics_text = (
                f"3D SAGIN Network (Epoch {self.current_epoch}/{self.max_epochs})\n"
                f"Success Rate: {self.network.metrics.success_rate:.3f}\n"
                f"Avg Latency: {self.network.metrics.average_latency:.3f}s\n"
                f"UAV Util: {self.network.metrics.uav_utilization:.3f}\n"
                f"Sat Util: {self.network.metrics.satellite_utilization:.3f}\n"
                f"Load Imbal: {self.network.metrics.load_imbalance:.3f}"
            )
            # Position text in upper left of the plot
            self.ax.text2D(0.02, 0.98, metrics_text, transform=self.ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def run(self, interval=1000, enable_simulation=True, max_epochs=None):
        """
        Run the real-time 3D visualization.
        
        Args:
            interval: Update interval in milliseconds
            enable_simulation: Whether to run simulation steps
            max_epochs: Maximum number of simulation epochs (if None, uses network's config)
        """
        self.simulation_running = enable_simulation
        
        # Use provided max_epochs or default to network's configuration
        if max_epochs is not None:
            self.max_epochs = max_epochs
        # else: keep the auto-detected value from __init__
        
        print(f"🎨 Starting real-time 3D visualization...")
        if enable_simulation:
            print(f"   Running simulation for {self.max_epochs} epochs")
        print("   3D View: Satellites (top) → UAVs (middle) → Vehicles (ground)")
        print("   Use mouse to rotate and zoom the 3D view")
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, self._update, 
            interval=interval, repeat=False
        )
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        return ani
