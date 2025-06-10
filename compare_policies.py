# compare_policies_clean_baseline.py
import numpy as np
import matplotlib.pyplot as plt
from sagin_env import SAGINEnv
from sagin_env import SystemDownException

# from ppo_gru_agent import GRUPPOAgent


from sagin_env import SystemDownException

def run_policy(env, episodes=5, visualize=True):
    logs = {
        'success_log': [], 'energy': [], 'cache_hits': [], 'task_log': [], 'dropped': 0
    }

    try:
        for ep in range(episodes):
            print('episode:', ep)
            for timestep in range(50):
                # Print progress indicator every 10 steps
                if timestep % 10 == 0:
                    print(f"  Step {timestep}/50")
                
                # Core simulation steps
                env.collect_iot_data()
                env.allocate_ofdm_slots()
                for sat in env.sats:
                    sat.update_coverage(timestep)
                env.upload_to_satellites()
                env.sync_satellites()
                env.generate_and_offload_tasks()
                env.step()  # <- may raise SystemDownException
                
                # Generate visualization at specific intervals if requested
                if visualize and timestep > 0 and timestep % 25 == 0:
                    try:
                        from visualize_network import create_visualizations
                        print(f"Creating visualizations for timestep {timestep}...")
                        create_visualizations(env, timestep)
                    except Exception as e:
                        print(f"Visualization error: {e}")

    except SystemDownException as e:
        print(e)
        print("Simulation ended due to energy depletion.")

    logs['success_log'] = env.success_log
    logs['task_log'] = env.task_log
    logs['dropped'] = env.dropped_tasks
    return logs



def summarize_and_plot(baseline_log, env=None):
    def get_success_rate(log):
        total = sum(e['completed'] for e in log)
        ok = sum(e['successful'] for e in log)
        return 100 * ok / total if total else 0

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Performance metrics
    ax1 = fig.add_subplot(2, 2, 1)
    labels = ['Success Rate (%)', 'Avg Cache Hit', 'Avg Energy Used', 'Dropped Tasks']
    baseline_vals = [
        get_success_rate(baseline_log['success_log']),
        np.mean(baseline_log['cache_hits']),
        np.mean(baseline_log['energy']),
        baseline_log['dropped']
    ]

    x = np.arange(len(labels))
    width = 0.4
    ax1.bar(x, baseline_vals, width, label='Baseline (Pop-based Caching)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Overall Performance Metrics")
    ax1.legend()
    
    # Only add the following plots if we have environment data
    if env is not None:
        # 2. Task completion by tier
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Collect tier statistics
        tiers = ['Ground Vehicles', 'UAVs', 'Satellites']
        
        # Ground vehicles
        total_gv_gen = sum(stats['generated'] for stats in env.task_stats['ground_vehicle'].values())
        total_gv_comp = sum(stats['completed'] for stats in env.task_stats['ground_vehicle'].values())
        total_gv_succ = sum(stats['successful'] for stats in env.task_stats['ground_vehicle'].values())
        
        # UAVs
        total_uav_gen = sum(stats['generated'] for stats in env.task_stats['uav'].values())
        total_uav_comp = sum(stats['completed'] for stats in env.task_stats['uav'].values())
        total_uav_succ = sum(stats['successful'] for stats in env.task_stats['uav'].values())
        
        # Satellites
        total_sat_rec = sum(stats['received'] for stats in env.task_stats['satellite'].values())
        total_sat_comp = sum(stats['completed'] for stats in env.task_stats['satellite'].values())
        total_sat_succ = sum(stats['successful'] for stats in env.task_stats['satellite'].values())
        
        # Total tasks by tier
        tier_tasks = [total_gv_gen, total_uav_gen, total_sat_rec]
        tier_completed = [total_gv_comp, total_uav_comp, total_sat_comp]
        tier_successful = [total_gv_succ, total_uav_succ, total_sat_succ]
        
        width = 0.25
        x = np.arange(len(tiers))
        
        ax2.bar(x - width, tier_tasks, width, label='Generated/Received')
        ax2.bar(x, tier_completed, width, label='Completed')
        ax2.bar(x + width, tier_successful, width, label='Successful')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(tiers)
        ax2.set_title("Task Handling by Tier")
        ax2.legend()
        
        # 3. Vehicle distribution
        ax3 = fig.add_subplot(2, 2, 3)
        
        vehicles_per_grid = {}
        for v_id, vehicle in env.vehicles.items():
            grid_pos = vehicle.get_position_as_grid(env.X, env.Y)
            vehicles_per_grid[grid_pos] = vehicles_per_grid.get(grid_pos, 0) + 1
        
        if vehicles_per_grid:
            grid_coords = list(vehicles_per_grid.keys())
            vehicle_counts = [vehicles_per_grid[pos] for pos in grid_coords]
            grid_labels = [f"({x},{y})" for x, y in grid_coords]
            
            ax3.bar(grid_labels, vehicle_counts)
            ax3.set_title("Vehicle Distribution Across Grid")
            ax3.set_xlabel("Grid Cell")
            ax3.set_ylabel("Number of Vehicles")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
        # 4. V2V and V2U connectivity
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Calculate average number of V2V and V2U links per vehicle
        total_v2v = sum(len(v.neighbor_vehicles) for v in env.vehicles.values())
        avg_v2v = total_v2v / len(env.vehicles) if env.vehicles else 0
        
        total_v2u = len(env.vehicle_uav_assignments)
        avg_v2u = total_v2u / len(env.vehicles) if env.vehicles else 0
        
        # Count U2S links
        u2s_links = len(env.connected_uavs)
        
        link_types = ['Avg. V2V per Vehicle', 'V2U Links', 'U2S Links']
        link_counts = [avg_v2v, total_v2u, u2s_links]
        
        ax4.bar(link_types, link_counts, color=['green', 'blue', 'red'])
        ax4.set_title("Connectivity in SAGIN Network")
        ax4.set_ylabel("Number of Links")
        
    plt.tight_layout()
    plt.savefig("sagin_baseline_performance.png", dpi=300, bbox_inches='tight')
    print("Performance visualization saved as 'sagin_baseline_performance.png'")
    plt.show()


if __name__ == "__main__":
    X, Y = 10, 10
    cache_size = 2048
    compute_power_uav = 30
    compute_power_sat = 200
    compute_power_gv = 20  # Compute power for ground vehicles
    energy = 540000
    max_queue = 10
    num_sats = 2
    num_vehicles = 15  # Number of ground vehicles
    num_iot_per_region = 50
    max_active_iot = 25
    ofdm_slots = 9
    duration = 300

    env = SAGINEnv(X, Y, duration, cache_size, compute_power_uav, compute_power_sat, compute_power_gv, 
                   energy, max_queue, num_sats, num_iot_per_region, max_active_iot, ofdm_slots, 
                   num_vehicles=num_vehicles)

    logs_baseline = run_policy(env, episodes=1)
    
    # Print ground vehicle stats
    print("\n=== Ground Vehicle Task Stats ===")
    if env.task_stats['ground_vehicle']:
        for v_id, stats in env.task_stats['ground_vehicle'].items():
            print(
                f"Vehicle {v_id}: Generated={stats['generated']}, Completed={stats['completed']}, Successful={stats['successful']}")
    else:
        print("No ground vehicle task statistics available")

    # Print UAV stats
    print("\n=== UAV Task Stats ===")
    for coord, stats in env.task_stats['uav'].items():
        print(
            f"UAV {coord}: Generated={stats['generated']}, Completed={stats['completed']}, Successful={stats['successful']}")

    # Print satellite stats
    print("\n=== Satellite Task Stats ===")
    for sid, stats in env.task_stats['satellite'].items():
        print(
            f"Satellite {sid}: Received={stats['received']}, Completed={stats['completed']}, Successful={stats['successful']}")

    # Vehicle mobility summary
    print("\n=== Ground Vehicle Distribution ===")
    vehicles_per_grid = {}
    for v_id, vehicle in env.vehicles.items():
        grid_pos = vehicle.get_position_as_grid(env.X, env.Y)
        vehicles_per_grid[grid_pos] = vehicles_per_grid.get(grid_pos, 0) + 1
    
    for grid_pos, count in vehicles_per_grid.items():
        print(f"Grid cell {grid_pos}: {count} vehicles")

    # Print summary visualization
    print("\n=== Generating Performance Visualization ===")
    summarize_and_plot(logs_baseline, env=env)

    # === Ready for PPO Extension Later ===
    # ppo_agent = GRUPPOAgent(obs_dim=..., num_contents=...)
    # logs_ppo = run_policy(env, agent=ppo_agent, use_ppo=True)
    # summarize_and_plot(logs_baseline, logs_ppo)
