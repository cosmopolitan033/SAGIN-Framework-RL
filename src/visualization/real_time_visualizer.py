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
        
        # Define altitude levels for 3D visualization with proper hierarchy
        self.ground_level = 0      # Vehicles at ground level
        self.uav_level = network.system_params.static_uav_altitude       # Static UAVs from config
        self.dynamic_uav_level = network.system_params.dynamic_uav_altitude  # Dynamic UAVs at 2x static altitude
        self.flying_uav_level = self.dynamic_uav_level + 40   # Flying UAVs slightly higher
        self.satellite_level = 500 # Satellites at 500km (scaled down for visualization)
        
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
        
        # Draw dynamic UAVs at different altitudes based on status
        dynamic_positions = []
        dynamic_colors = []
        dynamic_sizes = []
        
        for uav in self.network.uav_manager.dynamic_uavs.values():
            # Set altitude based on UAV status for clear visual separation
            if uav.status.value == "flying":
                # Flying UAVs at highest altitude
                uav_altitude = self.flying_uav_level
            else:
                # Active UAVs at standard dynamic UAV level
                uav_altitude = self.dynamic_uav_level
            
            dynamic_positions.append([uav.position.x, uav.position.y, uav_altitude])
            
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
                    [self.ground_level, uav_altitude],
                    color='red', alpha=0.3, linewidth=2
                )
        
        # Plot dynamic UAVs at proper altitude (single plotting)
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            self.ax.scatter(
                dynamic_positions[:, 0], dynamic_positions[:, 1], dynamic_positions[:, 2],
                c=dynamic_colors, s=dynamic_sizes, marker='^', label='Dynamic UAVs', 
                depthshade=True, edgecolors='white', linewidths=1.5, alpha=0.9
            )

    def _draw_communication_links(self):
        """Draw 3D communication links showing actual task assignments and active offloading."""
        # Get UAVs separated by type
        static_uavs = [uav for uav in self.network.uav_manager.static_uavs.values() 
                      if uav.is_available]
        dynamic_uavs = [uav for uav in self.network.uav_manager.dynamic_uavs.values() 
                       if uav.is_available and uav.status.value != "flying"]
        
        links_drawn = 0
        max_vehicle_links = 20  # Limit vehicle links for performance
        max_offload_links = 15  # Limit offloading links
        
        # 1. VEHICLES → STATIC UAVs (Primary connections - always present)
        vehicles_list = list(self.network.vehicle_manager.vehicles.values())
        for vehicle in vehicles_list[::3]:  # Sample every 3rd vehicle
            if links_drawn >= max_vehicle_links:
                break
                
            # Find closest static UAV (vehicles always connect to static UAVs first)
            closest_static_uav = None
            min_distance = float('inf')
            
            for uav in static_uavs:
                distance = vehicle.position.distance_to(uav.position)
                if distance < min_distance and distance < uav.communication_range:
                    min_distance = distance
                    closest_static_uav = uav
            
            if closest_static_uav:
                # Draw vehicle-to-static-UAV link
                quality = max(0.1, 1.0 - min_distance / closest_static_uav.communication_range)
                color = plt.cm.Blues(0.4 + 0.6 * quality)  # Blue for vehicle-to-static links
                
                self.ax.plot(
                    [vehicle.position.x, closest_static_uav.position.x],
                    [vehicle.position.y, closest_static_uav.position.y],
                    [self.ground_level, self.uav_level],
                    color=color, linewidth=1.5, alpha=0.8
                )
                links_drawn += 1
        
        # 2. ACTIVE TASK OFFLOADING LINKS (Based on actual current task assignments)
        offload_links_drawn = 0
        
        # Check each static UAV for active task offloading
        for static_uav in static_uavs:
            if offload_links_drawn >= max_offload_links:
                break
            
            # Check if static UAV has tasks assigned to dynamic UAVs or satellites
            if hasattr(static_uav, 'processing_tasks') and hasattr(static_uav, 'task_queue'):
                current_load = static_uav.total_workload / static_uav.cpu_capacity
                
                # Show offloading to dynamic UAVs when static UAV is handling high load
                if current_load > 0.6:  # Above 60% capacity indicates active offloading scenario
                    # Find dynamic UAVs in same region that are actively processing tasks
                    region_dynamic_uavs = [
                        uav for uav in dynamic_uavs 
                        if (uav.assigned_region_id == static_uav.assigned_region_id and 
                            uav.total_workload > 0)  # Only show if actually processing tasks
                    ]
                    
                    if region_dynamic_uavs:
                        # Connect to the most loaded dynamic UAV (showing actual task processing)
                        target_dynamic_uav = max(region_dynamic_uavs, 
                                               key=lambda u: u.total_workload)
                        
                        # Determine the correct altitude for the dynamic UAV
                        if target_dynamic_uav.status.value == "flying":
                            dynamic_altitude = self.flying_uav_level
                        else:
                            dynamic_altitude = self.dynamic_uav_level
                        
                        # Draw static-to-dynamic UAV active offloading link
                        link_intensity = min(1.0, target_dynamic_uav.total_workload / target_dynamic_uav.cpu_capacity)
                        color = plt.cm.Oranges(0.5 + 0.5 * link_intensity)  # Orange intensity based on dynamic UAV load
                        
                        self.ax.plot(
                            [static_uav.position.x, target_dynamic_uav.position.x],
                            [static_uav.position.y, target_dynamic_uav.position.y],
                            [self.uav_level, dynamic_altitude],
                            color=color, linewidth=2.5, alpha=0.9, linestyle='--'
                        )
                        offload_links_drawn += 1
                
                # Show satellite offloading links when UAV has high load AND satellites are actively processing tasks
                # This indicates actual offloading is happening
                uav_load_ratio = static_uav.total_workload / static_uav.cpu_capacity
                show_satellite_link = (uav_load_ratio > 0.6 and offload_links_drawn < max_offload_links)
                
                if show_satellite_link:
                    if (hasattr(self.network, 'satellite_constellation') and 
                        self.network.satellite_constellation.satellites):
                        
                        satellites = list(self.network.satellite_constellation.satellites.values())
                        
                        # Find the most loaded satellite (likely receiving tasks)
                        if satellites:
                            target_satellite = max(satellites, key=lambda s: s.total_workload)
                            
                            # Only show link if satellite actually has workload
                            if target_satellite.total_workload > 0:
                                # Get satellite position from our visualization layout
                                sat_positions = self._get_satellite_positions()
                                sat_index = list(self.network.satellite_constellation.satellites.keys()).index(target_satellite.id)
                                
                                if sat_index < len(sat_positions):
                                    sat_pos = sat_positions[sat_index]
                                    
                                    # Draw ACTIVE UAV-to-satellite offloading link
                                    # Color intensity based on satellite utilization
                                    sat_utilization = target_satellite.total_workload / target_satellite.cpu_capacity
                                    link_intensity = min(1.0, sat_utilization)
                                    color = plt.cm.Purples(0.6 + 0.4 * link_intensity)  # Purple for satellite offloading
                                    
                                    # Dotted line for satellite offloading links
                                    self.ax.plot(
                                        [static_uav.position.x, sat_pos[0]],
                                        [static_uav.position.y, sat_pos[1]], 
                                        [self.uav_level, self.satellite_level],
                                        color=color, linewidth=2.5, alpha=0.9, linestyle=':'
                                    )
                                    offload_links_drawn += 1
    
    def _get_satellite_positions(self):
        """Get satellite positions for visualization (helper method)."""
        if not hasattr(self.network, 'satellite_constellation') or not self.network.satellite_constellation.satellites:
            return []
        
        satellite_positions = []
        min_x, max_x, min_y, max_y = self.grid.area_bounds
        area_center_x = (min_x + max_x) / 2
        area_center_y = (min_y + max_y) / 2
        area_width = max_x - min_x
        area_height = max_y - min_y
        
        num_satellites = len(self.network.satellite_constellation.satellites)
        
        for sat_id in range(num_satellites):
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
            
            satellite_positions.append([sat_x, sat_y, self.satellite_level])
        
        return satellite_positions

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
        self._draw_communication_links()  # Hierarchical communication links
        
        # Add enhanced legend for 3D visualization
        legend_elements = []
        
        # Vehicle types (ground level)
        if len(self.network.vehicle_manager.vehicles) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Cars (Ground)', markeredgecolor='darkblue'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=6, label='Buses (Ground)', markeredgecolor='darkorange'))
        
        # UAV types (medium altitude with better separation)
        if len(self.network.uav_manager.static_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label=f'Static UAVs ({self.uav_level}m)', markeredgecolor='black'))
        if len(self.network.uav_manager.dynamic_uavs) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label=f'Dynamic UAVs ({self.dynamic_uav_level}m)', markeredgecolor='white'))
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=9, label=f'Flying UAVs ({self.flying_uav_level}m)', markeredgecolor='white'))
        
        # Communication link types (hierarchical)
        legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=2, label='Vehicle → Static UAV'))
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Static → Dynamic UAV (Offloading)'))
        legend_elements.append(plt.Line2D([0], [0], color='purple', linewidth=2.5, linestyle=':', label='UAV → Satellite (Active Offloading)'))
        
        # Satellite elements (high altitude)
        if hasattr(self.network, 'satellite_constellation') and self.network.satellite_constellation.satellites:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lightblue', markersize=10, label=f'Satellites ({self.satellite_level}km)', markeredgecolor='navy'))
        
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
        print("   Communication Flow: Vehicle → Static UAV → Dynamic UAV → Satellite")
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
