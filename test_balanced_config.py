#!/usr/bin/env python3
"""Test script for balanced configuration."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from examples.sagin_demo import SAGINDemo
from config.grid_config import get_sagin_config

def test_balanced_config():
    """Test the balanced configuration."""
    print("Testing realistic resource pressure:")
    print("- Task complexity: 300M cycles (3s processing time)")
    print("- UAV CPU capacity: 100M cycles/s")
    print("- Task deadlines: 2.5±0.5s")
    print("- Expected: Moderate success rate with measurable latency")
    print()
    
    # Create demo instance
    demo = SAGINDemo()
    
    # Override configuration for testing
    config = get_sagin_config('large_simulation')
    config.simulation.total_epochs = 15
    config.simulation.logging_level = 'low'
    config.simulation.progress_interval = 5
    
    print(f"Running simulation with:")
    print(f"  - Task complexity: {config.tasks.cpu_cycles_mean:e} cycles")
    print(f"  - Task rate: {config.tasks.base_task_rate} tasks/sec/region")
    print(f"  - CPU capacity: {config.uavs.uav_cpu_capacity:e} cycles/s")
    print(f"  - Deadline: {config.tasks.deadline_mean}±{config.tasks.deadline_std}s")
    print()
    
    # Run simulation
    results = demo.run_simulation(config.name)
    
    # Extract metrics
    metrics = results['final_metrics']
    task_metrics = results.get('task_metrics', {})
    
    print("Final Results:")
    print(f"  - Success rate: {metrics.success_rate:.1%}")
    print(f"  - Average latency: {metrics.average_latency:.3f}s")
    print(f"  - Tasks generated: {task_metrics.get('total_generated', 0)}")
    print(f"  - Tasks completed: {task_metrics.get('total_completed', 0)}")
    print(f"  - Tasks failed: {task_metrics.get('total_failed', 0)}")
    print(f"  - UAV utilization: {metrics.uav_utilization:.1%}")
    print(f"  - Satellite utilization: {metrics.satellite_utilization:.1%}")
    print()
    
    print("Analysis:")
    expected_processing_time = config.tasks.cpu_cycles_mean / config.uavs.uav_cpu_capacity
    print(f"  - Expected processing time: {expected_processing_time:.1f}s")
    print(f"  - Average deadline: {config.tasks.deadline_mean}s")
    print(f"  - Deadline buffer: {config.tasks.deadline_mean - expected_processing_time:+.1f}s")
    
    if metrics.average_latency > 0.1:
        print(f"  - Latency measurement: ✅ Working ({metrics.average_latency:.3f}s)")
    else:
        print(f"  - Latency measurement: ❌ Still broken ({metrics.average_latency:.3f}s)")
    
    if 0.3 <= metrics.success_rate <= 0.7:
        print(f"  - Resource pressure: ✅ Realistic ({metrics.success_rate:.1%})")
    else:
        print(f"  - Resource pressure: ⚠️ May need adjustment ({metrics.success_rate:.1%})")

if __name__ == "__main__":
    test_balanced_config()
