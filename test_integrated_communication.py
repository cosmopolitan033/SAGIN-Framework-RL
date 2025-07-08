#!/usr/bin/env python3
"""
Test script for the integrated communication model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.types import Position, NodeType, SystemParameters
from src.models.communication import CommunicationModel, LoadBalancingMetrics, ShannonCapacityModel


def test_integrated_communication_model():
    """Test the integrated communication model with advanced features."""
    print("Testing Integrated Communication Model")
    print("=" * 50)
    
    # Create system parameters
    system_params = SystemParameters(
        min_rate_threshold=1.0,  # Mbps
        propagation_speed=3e8    # m/s
    )
    
    # Create communication model
    comm_model = CommunicationModel(system_params)
    
    # Define test positions
    vehicle_pos = Position(0, 0, 0)
    uav_pos = Position(100, 100, 50)
    satellite_pos = Position(1000, 1000, 35786000)  # GEO satellite
    
    print("1. Testing basic communication link (Vehicle to UAV)")
    print("-" * 40)
    
    # Test basic data rate calculation
    data_rate = comm_model.calculate_data_rate(
        vehicle_pos, uav_pos, NodeType.VEHICLE, NodeType.DYNAMIC_UAV
    )
    print(f"Data rate: {data_rate:.2f} Mbps")
    
    # Test advanced data rate calculation
    advanced_result = comm_model.calculate_advanced_data_rate(
        vehicle_pos, uav_pos, NodeType.VEHICLE, NodeType.DYNAMIC_UAV
    )
    print(f"Advanced data rate: {advanced_result['data_rate_mbps']:.2f} Mbps")
    print(f"SINR: {advanced_result['sinr_db']:.2f} dB")
    print(f"Fading factor: {advanced_result['fading_factor']:.3f}")
    print(f"Modulation order: {advanced_result['modulation_order']}")
    print(f"Coding rate: {advanced_result['coding_rate']:.2f}")
    
    print("\n2. Testing link quality metrics")
    print("-" * 40)
    
    # Test link quality metrics
    metrics = comm_model.get_link_quality_metrics(
        vehicle_pos, uav_pos, NodeType.VEHICLE, NodeType.DYNAMIC_UAV
    )
    print(f"Distance: {metrics['distance']:.2f} m")
    print(f"Channel gain: {metrics['channel_gain_db']:.2f} dB")
    print(f"SINR: {metrics['sinr_db']:.2f} dB")
    print(f"Link available: {metrics['is_available']}")
    
    print("\n3. Testing interference modeling")
    print("-" * 40)
    
    # Add interference sources
    comm_model.add_interference_source(Position(50, 50, 10), NodeType.DYNAMIC_UAV, 5.0)
    comm_model.add_interference_source(Position(150, 150, 20), NodeType.DYNAMIC_UAV, 3.0)
    
    # Test with interference
    advanced_result_with_interference = comm_model.calculate_advanced_data_rate(
        vehicle_pos, uav_pos, NodeType.VEHICLE, NodeType.DYNAMIC_UAV
    )
    print(f"Data rate with interference: {advanced_result_with_interference['data_rate_mbps']:.2f} Mbps")
    print(f"Interference power: {advanced_result_with_interference['interference_power_w']:.6f} W")
    
    print("\n4. Testing satellite communication")
    print("-" * 40)
    
    # Test satellite link
    sat_data_rate = comm_model.calculate_data_rate(
        uav_pos, satellite_pos, NodeType.DYNAMIC_UAV, NodeType.SATELLITE
    )
    print(f"UAV to Satellite data rate: {sat_data_rate:.2f} Mbps")
    
    # Test satellite propagation delay
    prop_delay = comm_model.calculate_propagation_delay(uav_pos, satellite_pos)
    print(f"Satellite propagation delay: {prop_delay*1000:.2f} ms")
    
    print("\n5. Testing link reliability")
    print("-" * 40)
    
    # Test link reliability
    reliability = comm_model.calculate_link_reliability(
        vehicle_pos, uav_pos, NodeType.VEHICLE, NodeType.DYNAMIC_UAV, 5.0
    )
    print(f"Link reliability (5 Mbps requirement): {reliability:.2f}")
    
    print("\n6. Testing load balancing metrics")
    print("-" * 40)
    
    # Test load balancing
    load_metrics = LoadBalancingMetrics()
    node_workloads = [0.2, 0.8, 0.5, 0.9, 0.3]
    load_info = load_metrics.calculate_comprehensive_load_metrics(node_workloads)
    
    print(f"Load imbalance coefficient: {load_info['load_imbalance_coefficient']:.3f}")
    print(f"Fairness index: {load_info['fairness_index']:.3f}")
    print(f"Peak-to-average ratio: {load_info['peak_to_average_ratio']:.3f}")
    
    print("\n7. Testing Shannon capacity model")
    print("-" * 40)
    
    # Test Shannon capacity model
    shannon_model = ShannonCapacityModel(system_params)
    capacity_info = shannon_model.calculate_practical_capacity(20.0, 15.0)
    
    print(f"Theoretical capacity: {capacity_info['theoretical_capacity_mbps']:.2f} Mbps")
    print(f"Practical capacity: {capacity_info['practical_capacity_mbps']:.2f} Mbps")
    print(f"Efficiency: {capacity_info['efficiency']:.2f}")
    print(f"Modulation scheme: {capacity_info['modulation_scheme']}")
    print(f"Coding rate: {capacity_info['coding_rate']:.2f}")
    
    print("\n" + "=" * 50)
    print("Integration test completed successfully!")


if __name__ == "__main__":
    test_integrated_communication_model()
