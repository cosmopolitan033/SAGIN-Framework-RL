"""
Static Network Visualizer for SAGIN

Features:
- Network topology snapshot visualization
- Grid layout with region information
- Vehicle and UAV positions
- Communication coverage areas

Usage:
    visualizer = NetworkVisualizer(network)
    visualizer.show()
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class NetworkVisualizer:
    def __init__(self, network):
        self.network = network
        self.grid = network.grid
        
    def show(self, figsize=(12, 8), save_path=None):
        """Show static network visualization."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw grid regions
        self._draw_grid(ax)
        
        # Draw vehicles
        self._draw_vehicles(ax)
        
        # Draw UAVs
        self._draw_uavs(ax)
        
        # Setup plot
        ax.set_xlim(self.grid.area_bounds[0], self.grid.area_bounds[1])
        ax.set_ylim(self.grid.area_bounds[2], self.grid.area_bounds[3])
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('SAGIN Network Topology Snapshot')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add network statistics
        stats_text = self._get_network_stats()
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
    def _draw_grid(self, ax):
        """Draw grid regions."""
        for row in range(self.grid.grid_rows):
            for col in range(self.grid.grid_cols):
                region_id = self.grid.get_region_id(row, col)
                center_x, center_y = self.grid.get_region_center(row, col)
                
                rect = patches.Rectangle(
                    (center_x - self.grid.region_width/2, center_y - self.grid.region_height/2),
                    self.grid.region_width, self.grid.region_height,
                    linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.3
                )
                ax.add_patch(rect)
                
                # Add region ID
                ax.text(center_x, center_y, str(region_id), 
                       ha='center', va='center', fontsize=8, alpha=0.7)
    
    def _draw_vehicles(self, ax):
        """Draw vehicles."""
        if not self.network.vehicle_manager.vehicles:
            return
            
        positions = np.array([[v.position.x, v.position.y] for v in self.network.vehicle_manager.vehicles])
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='blue', s=20, alpha=0.7, label=f'Vehicles ({len(positions)})')
    
    def _draw_uavs(self, ax):
        """Draw UAVs with coverage zones."""
        # Static UAVs
        static_positions = []
        for uav in self.network.uav_manager.static_uavs:
            static_positions.append([uav.position.x, uav.position.y])
            # Coverage zone
            coverage = patches.Circle(
                (uav.position.x, uav.position.y), uav.communication_range,
                color='green', alpha=0.1, zorder=1
            )
            ax.add_patch(coverage)
        
        if static_positions:
            static_positions = np.array(static_positions)
            ax.scatter(static_positions[:, 0], static_positions[:, 1],
                      c='green', s=60, marker='^', label=f'Static UAVs ({len(static_positions)})',
                      edgecolors='black', zorder=3)
        
        # Dynamic UAVs
        dynamic_positions = []
        for uav in self.network.uav_manager.dynamic_uavs:
            if uav.is_active:
                dynamic_positions.append([uav.position.x, uav.position.y])
                # Coverage zone
                coverage = patches.Circle(
                    (uav.position.x, uav.position.y), uav.communication_range,
                    color='red', alpha=0.1, zorder=1
                )
                ax.add_patch(coverage)
        
        if dynamic_positions:
            dynamic_positions = np.array(dynamic_positions)
            ax.scatter(dynamic_positions[:, 0], dynamic_positions[:, 1],
                      c='red', s=60, marker='^', label=f'Dynamic UAVs ({len(dynamic_positions)})',
                      edgecolors='black', zorder=3)
    
    def _get_network_stats(self):
        """Get network statistics text."""
        stats = []
        stats.append(f"Grid: {self.grid.grid_rows}×{self.grid.grid_cols}")
        stats.append(f"Area: {self.grid.area_bounds[1]/1000:.1f}×{self.grid.area_bounds[3]/1000:.1f}km")
        stats.append(f"Vehicles: {len(self.network.vehicle_manager.vehicles)}")
        stats.append(f"Static UAVs: {len(self.network.uav_manager.static_uavs)}")
        stats.append(f"Dynamic UAVs: {len(self.network.uav_manager.dynamic_uavs)}")
        stats.append(f"Satellites: {len(self.network.satellite_constellation.satellites)}")
        
        if hasattr(self.network, 'metrics'):
            stats.append(f"Success Rate: {self.network.metrics.success_rate:.3f}")
            stats.append(f"UAV Utilization: {self.network.metrics.uav_utilization:.3f}")
        
        return "\n".join(stats)
