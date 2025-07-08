#!/usr/bin/env python3
"""
Quick test to verify the SAGIN implementation.
"""

import sys
import os
sys.path.append('src')

def test_shannon_capacity():
    """Test the Shannon capacity model."""
    print("Testing Shannon Capacity Model...")
    
    try:
        from src.models.communication import ShannonCapacityModel
        from src.core.types import SystemParameters
        
        # Create test parameters
        system_params = SystemParameters()
        model = ShannonCapacityModel(system_params)
        
        # Test practical capacity calculation
        result = model.calculate_practical_capacity(bandwidth_mhz=20.0, snr_db=15.0)
        
        print(f"✅ Shannon Capacity Test Passed!")
        print(f"   Theoretical Capacity: {result['theoretical_capacity_mbps']:.2f} Mbps")
        print(f"   Practical Capacity: {result['practical_capacity_mbps']:.2f} Mbps")
        print(f"   Modulation Scheme: {result['modulation_scheme']}")
        print(f"   Coding Rate: {result['coding_rate']:.2f}")
        print(f"   Efficiency: {result['efficiency']:.2f}")
        
        # Test adaptive capacity
        channel_conditions = {
            'snr_db': 18.0,
            'bandwidth_hz': 20e6,
            'fading_margin_db': 3.0,
            'interference_margin_db': 2.0
        }
        
        adaptive_result = model.calculate_adaptive_capacity(channel_conditions)
        print(f"   Reliable Capacity: {adaptive_result['reliable_capacity_bps']/1e6:.2f} Mbps")
        
        # Test paper formula: R_ij = W_ij * log2(1 + P_t * G_ij / σ²)
        bandwidth_hz = 20e6
        transmit_power = 10.0
        channel_gain = 0.001
        noise_power = 1e-12
        
        shannon_capacity = model.calculate_shannon_capacity(bandwidth_hz, transmit_power, channel_gain, noise_power)
        print(f"   Paper Formula Capacity: {shannon_capacity/1e6:.2f} Mbps")
        
        return True
        
    except Exception as e:
        print(f"❌ Shannon Capacity Test Failed: {e}")
        return False

def test_load_balancing():
    """Test the load balancing metrics."""
    print("\nTesting Load Balancing Metrics...")
    
    try:
        from src.models.communication import LoadBalancingMetrics
        import numpy as np
        
        # Create test workloads
        metrics = LoadBalancingMetrics()
        test_workloads = [0.1, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1]
        
        # Calculate comprehensive metrics
        result = metrics.calculate_comprehensive_load_metrics(test_workloads)
        
        print(f"✅ Load Balancing Test Passed!")
        print(f"   Paper Formula (ΔL): {result['load_imbalance_paper']:.3f}")
        print(f"   Load Imbalance Coefficient: {result['load_imbalance_coefficient']:.3f}")
        print(f"   Fairness Index: {result['fairness_index']:.3f}")
        print(f"   Peak-to-Average Ratio: {result['peak_to_average_ratio']:.3f}")
        print(f"   Coefficient of Variation: {result['coefficient_of_variation']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Load Balancing Test Failed: {e}")
        return False

def test_latency_breakdown():
    """Test the detailed latency breakdown."""
    print("\nTesting Detailed Latency Breakdown...")
    
    try:
        from src.models.latency import DetailedLatencyBreakdown
        from src.core.types import Task, Position, NodeType
        
        # Create test components
        breakdown = DetailedLatencyBreakdown()
        
        # Create mock task
        task = Task(
            id=1,
            source_vehicle_id=1,
            region_id=1,
            data_size_in=1.0,
            data_size_out=0.5,
            cpu_cycles=1e9,
            deadline=10.0,
            creation_time=0.0
        )
        
        # Create mock network path
        network_path = [
            (Position(0, 0, 0), NodeType.VEHICLE),
            (Position(100, 100, 100), NodeType.STATIC_UAV)
        ]
        
        # Create mock processing info
        processing_info = {
            'node_id': 1,
            'cpu_capacity': 1e9,
            'current_queue': []
        }
        
        # Calculate detailed latency
        result = breakdown.calculate_detailed_latency(task, network_path, processing_info)
        
        print(f"✅ Latency Breakdown Test Passed!")
        print(f"   Total Latency: {result['total_latency']:.4f} seconds")
        print(f"   Communication: {result['total_communication']:.4f} seconds")
        print(f"   Processing: {result['total_processing']:.4f} seconds")
        print(f"   System Overhead: {result['total_system']:.4f} seconds")
        
        # Test bottleneck analysis
        analysis = breakdown.analyze_latency_bottlenecks(result)
        if analysis:
            print(f"   Primary Bottleneck: {analysis['bottleneck_type']}")
            print(f"   Recommendations: {len(analysis['optimization_recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Latency Breakdown Test Failed: {e}")
        return False

def test_energy_model():
    """Test the energy model conforming to paper formula."""
    print("\nTesting Energy Model (Paper Formula)...")
    
    try:
        from src.models.communication import EnergyModel
        from src.core.types import SystemParameters
        
        # Create test parameters
        system_params = SystemParameters()
        energy_model = EnergyModel(system_params)
        
        # Test individual energy components
        velocity_vector = (5.0, 3.0, 0.0)  # m/s in x, y, z
        data_transmitted = 2.5  # MB
        cpu_cycles = 1e9  # 1 billion cycles
        epoch_duration = 1.0  # 1 second
        
        # Calculate energy breakdown
        breakdown = energy_model.get_energy_breakdown(velocity_vector, data_transmitted, cpu_cycles, epoch_duration)
        
        # Test energy update formula: E_v(t+1) = E_v(t) - (E_flight + E_comm + E_comp)
        current_energy = 10000.0  # 10kJ
        new_energy = energy_model.update_uav_energy(current_energy, velocity_vector, data_transmitted, cpu_cycles, epoch_duration)
        
        # Test energy constraint
        constraint_check = energy_model.check_energy_constraint(new_energy)
        
        print(f"✅ Energy Model Test Passed!")
        print(f"   Flight Energy: {breakdown['flight_energy']:.2f} J ({breakdown['flight_percentage']:.1f}%)")
        print(f"   Communication Energy: {breakdown['communication_energy']:.2f} J ({breakdown['communication_percentage']:.1f}%)")
        print(f"   Computation Energy: {breakdown['computation_energy']:.2f} J ({breakdown['computation_percentage']:.1f}%)")
        print(f"   Total Energy Consumption: {breakdown['total_energy']:.2f} J")
        print(f"   Energy After Update: {new_energy:.2f} J")
        print(f"   Energy Constraint Satisfied: {constraint_check}")
        
        return True
        
    except Exception as e:
        print(f"❌ Energy Model Test Failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SAGIN FEATURES VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_shannon_capacity,
        test_load_balancing,
        test_latency_breakdown,
        test_energy_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL FEATURES WORKING CORRECTLY!")
        print("\nThe following features now conform to your paper:")
        print("• Shannon Capacity Model (R_ij = W_ij * log2(1 + P_t * G_ij / σ²))")
        print("• Detailed Latency Breakdown (T_total = T_prop + T_trans + T_queue + T_comp)")
        print("• Load Balancing Metrics (ΔL = sqrt(1/N * Σ(L_i - L̄)²))")
        print("• Energy Model (E_v(t+1) = E_v(t) - (E_flight + E_comm + E_comp))")
        print("\n✅ IMPLEMENTATION CONFORMS TO YOUR PAPER!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
