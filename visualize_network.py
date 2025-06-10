#!/usr/bin/env python3
# visualize_network.py - Visualization tool for SAGIN three-tier architecture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

def visualize_sagin_network(env, timestep):
    """
    Visualize the current state of the SAGIN network including:
    - Ground vehicles and their connections
    - UAVs and their coverage
    - Satellites and their connections
    """
    fig = plt.figure(figsize=(12, 9))
    
    # Creating a 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Grid dimensions
    grid_size_x = env.X * 100
    grid_size_y = env.Y * 100
    max_altitude = 600  # For visualization scaling
    
    # Set axis limits
    ax.set_xlim(0, grid_size_x)
    ax.set_ylim(0, grid_size_y)
    ax.set_zlim(0, max_altitude)
    
    # Plot grid lines
    for i in range(env.X + 1):
        x = i * 100
        ax.plot([x, x], [0, grid_size_y], [0, 0], 'k--', alpha=0.3)
    
    for j in range(env.Y + 1):
        y = j * 100
        ax.plot([0, grid_size_x], [y, y], [0, 0], 'k--', alpha=0.3)
    
    # Plot ground vehicles
    for v_id, vehicle in env.vehicles.items():
        x, y, z = vehicle.position
        # Plot vehicle as a scatter point
        ax.scatter(x, y, z, color='green', marker='o', s=50, label=f'Vehicle {v_id}' if v_id == 0 else "")
        
        # Show V2V connections
        for neighbor_id in vehicle.neighbor_vehicles:
            neighbor = env.vehicles[neighbor_id]
            nx, ny, nz = neighbor.position
            ax.plot([x, nx], [y, ny], [z, nz], 'g-', alpha=0.3)
        
        # Show V2U connection if connected
        if vehicle.connected_uav:
            uav_x, uav_y = vehicle.connected_uav
            ux, uy, uz = uav_x * 100 + 50, uav_y * 100 + 50, 100
            ax.plot([x, ux], [y, uy], [z, uz], 'c-', alpha=0.5)
    
    # Plot UAVs
    for (x, y), uav in env.uavs.items():
        ux = x * 100 + 50
        uy = y * 100 + 50
        uz = 100  # UAV altitude
        
        # Plot UAV as a scatter point
        ax.scatter(ux, uy, uz, color='blue', marker='^', s=100, label=f'UAV ({x},{y})' if (x,y) == (0,0) else "")
        
        # Show UAV-to-Satellite connection
        if (x, y) in env.connected_uavs:
            for sat in env.sats:
                if (x, y) in env.satellite_slots.get(sat.sat_id, []):
                    sx, sy, sz = sat.position
                    ax.plot([ux, sx], [uy, sy], [uz, sz], 'r--', alpha=0.5)
                    break
    
    # Plot Satellites
    for sat in env.sats:
        sx, sy, sz = sat.position
        # Reduce z-coordinate for visualization (actual altitude would be too high to visualize)
        vis_z = max_altitude - 50  # Just below the top of our visualization
        
        # Plot satellite as a scatter point
        ax.scatter(sx, sy, vis_z, color='red', marker='*', s=200, label=f'Satellite {sat.sat_id}' if sat.sat_id == 0 else "")
    
    # Set labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'SAGIN Network - Timestep {timestep}')
    
    # Add a legend (only showing one of each type)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Add stats text
    stats_text = (
        f"Vehicles: {len(env.vehicles)}\n"
        f"UAVs: {len(env.uavs)}\n"
        f"Satellites: {len(env.sats)}\n"
        f"V2V Links: {sum(len(v.neighbor_vehicles) for v in env.vehicles.values())}\n"
        f"V2U Links: {len(env.vehicle_uav_assignments)}\n"
        f"U2S Links: {len(env.connected_uavs)}"
    )
    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"sagin_network_t{timestep}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved as 'sagin_network_t{timestep}.png'")

def visualize_network_metrics(env, timestep):
    """
    Visualize key network metrics:
    - Task distribution by tier
    - Success rates
    - Content distribution
    - Energy consumption
    """
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Task distribution by tier (Pie chart)
    ax1 = axs[0, 0]
    
    # Calculate task counts
    gv_tasks = sum(stats['generated'] for stats in env.task_stats['ground_vehicle'].values())
    uav_tasks = sum(stats['generated'] for stats in env.task_stats['uav'].values())
    sat_tasks = sum(stats['received'] for stats in env.task_stats['satellite'].values())
    
    labels = ['Ground Vehicles', 'UAVs', 'Satellites']
    sizes = [gv_tasks, uav_tasks, sat_tasks]
    
    # Only include non-zero values in the pie chart
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
    non_zero_sizes = [size for size in sizes if size > 0]
    
    if non_zero_sizes:
        ax1.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%', startangle=90,
               colors=['green', 'blue', 'red'])
        ax1.set_title('Task Distribution by Tier')
    else:
        ax1.text(0.5, 0.5, 'No tasks yet', horizontalalignment='center', verticalalignment='center')
        ax1.set_title('Task Distribution by Tier (No Data)')
    
    # 2. Success rates (Bar chart)
    ax2 = axs[0, 1]
    
    # Calculate success rates
    tiers = ['Vehicles', 'UAVs', 'Satellites']
    completion_rates = []
    success_rates = []
    
    # Ground vehicles
    total_gv_gen = sum(stats['generated'] for stats in env.task_stats['ground_vehicle'].values())
    total_gv_comp = sum(stats['completed'] for stats in env.task_stats['ground_vehicle'].values())
    total_gv_succ = sum(stats['successful'] for stats in env.task_stats['ground_vehicle'].values())
    
    gv_comp_rate = 100 * total_gv_comp / total_gv_gen if total_gv_gen > 0 else 0
    gv_succ_rate = 100 * total_gv_succ / total_gv_comp if total_gv_comp > 0 else 0
    
    # UAVs
    total_uav_gen = sum(stats['generated'] for stats in env.task_stats['uav'].values())
    total_uav_comp = sum(stats['completed'] for stats in env.task_stats['uav'].values())
    total_uav_succ = sum(stats['successful'] for stats in env.task_stats['uav'].values())
    
    uav_comp_rate = 100 * total_uav_comp / total_uav_gen if total_uav_gen > 0 else 0
    uav_succ_rate = 100 * total_uav_succ / total_uav_comp if total_uav_comp > 0 else 0
    
    # Satellites
    total_sat_rec = sum(stats['received'] for stats in env.task_stats['satellite'].values())
    total_sat_comp = sum(stats['completed'] for stats in env.task_stats['satellite'].values())
    total_sat_succ = sum(stats['successful'] for stats in env.task_stats['satellite'].values())
    
    sat_comp_rate = 100 * total_sat_comp / total_sat_rec if total_sat_rec > 0 else 0
    sat_succ_rate = 100 * total_sat_succ / total_sat_comp if total_sat_comp > 0 else 0
    
    completion_rates = [gv_comp_rate, uav_comp_rate, sat_comp_rate]
    success_rates = [gv_succ_rate, uav_succ_rate, sat_succ_rate]
    
    x = np.arange(len(tiers))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, completion_rates, width, label='Completion Rate')
    bars2 = ax2.bar(x + width/2, success_rates, width, label='Success Rate')
    
    ax2.set_ylabel('Percentage')
    ax2.set_title('Task Completion and Success Rates')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.legend()
    
    # Set y limit a bit above 100 to ensure we can see 100% values
    ax2.set_ylim(0, 110)
    
    # Add labels above bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax2.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # 3. Content distribution (Stacked bar chart)
    ax3 = axs[1, 0]
    
    # Count content by type in different tiers
    content_types = ['sensor', 'image', 'video', 'map']
    colors = ['skyblue', 'lightgreen', 'salmon', 'violet']
    
    # Vehicle content
    vehicle_content = {}
    for v in env.vehicles.values():
        for content in list(v.cache_storage.values()) + list(v.aggregated_content.values()):
            content_type = content.get('type', 'unknown')
            vehicle_content[content_type] = vehicle_content.get(content_type, 0) + 1
    
    # UAV content
    uav_content = {}
    for uav in env.uavs.values():
        for content in list(uav.cache_storage.values()) + list(uav.aggregated_content.values()):
            content_type = content.get('type', 'unknown')
            uav_content[content_type] = uav_content.get(content_type, 0) + 1
    
    # Satellite content
    sat_content = {}
    for content in env.global_satellite_content_pool.values():
        content_type = content.get('type', 'unknown')
        sat_content[content_type] = sat_content.get(content_type, 0) + 1
    
    # Prepare data for stacked bar chart
    bottom = np.zeros(3)  # For vehicles, UAVs, satellites
    
    for i, content_type in enumerate(content_types):
        counts = [
            vehicle_content.get(content_type, 0),
            uav_content.get(content_type, 0),
            sat_content.get(content_type, 0)
        ]
        
        if sum(counts) > 0:  # Only plot if there's data
            ax3.bar(tiers, counts, bottom=bottom, label=content_type.capitalize(), color=colors[i])
            bottom += counts
    
    ax3.set_ylabel('Content Count')
    ax3.set_title('Content Distribution by Type and Tier')
    
    # Only show legend if we have content
    if sum(bottom) > 0:
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No content data yet', horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes)
    
    # 4. Energy consumption scatter plot
    ax4 = axs[1, 1]
    
    # Vehicle energy
    vehicle_energy = []
    for v in env.vehicles.values():
        initial_energy = 10000  # This should match the initial energy in main.py
        energy_consumption = 100 * (1 - v.energy / initial_energy)
        vehicle_energy.append((v.vehicle_id, energy_consumption))
    
    # UAV energy
    uav_energy = []
    for (x, y), uav in env.uavs.items():
        initial_energy = 10000  # This should match the initial energy in main.py
        energy_consumption = 100 * (1 - uav.energy / initial_energy)
        uav_energy.append(((x, y), energy_consumption))
    
    # Plot energy consumption
    if vehicle_energy:
        v_ids, v_energy = zip(*vehicle_energy)
        ax4.scatter(v_ids, v_energy, color='green', label='Vehicles', alpha=0.7, s=50)
    
    if uav_energy:
        uav_ids = [f"({x},{y})" for (x,y), _ in uav_energy]
        uav_e = [e for _, e in uav_energy]
        ax4.scatter(range(len(uav_ids)), uav_e, color='blue', label='UAVs', alpha=0.7, s=80, marker='s')
        
        # Add UAV coordinates as x-tick labels
        ax4.set_xticks(range(len(uav_ids)))
        ax4.set_xticklabels(uav_ids, rotation=45)
    
    # Set labels and title
    ax4.set_ylabel('Energy Consumption (%)')
    ax4.set_title('Energy Consumption by Node')
    
    # Add legend only if we have data
    if vehicle_energy or uav_energy:
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No energy data yet', horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes)
    
    # Set title for overall figure
    fig.suptitle(f'SAGIN Network Metrics - Timestep {timestep}', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(f"sagin_metrics_t{timestep}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network metrics visualization saved as 'sagin_metrics_t{timestep}.png'")

# This function can be called from main.py to generate visualizations
def create_visualizations(env, timestep):
    """Generate all visualizations for the current timestep"""
    visualize_sagin_network(env, timestep)
    visualize_network_metrics(env, timestep)
