"""
Latency models for end-to-end task processing in the SAGIN system.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.types import Task, Position, NodeType, SystemParameters
from .communication import CommunicationModel


@dataclass
class LatencyComponents:
    """Components of end-to-end latency."""
    propagation_delay: float = 0.0  # Signal propagation time
    transmission_delay: float = 0.0  # Data transmission time
    queuing_delay: float = 0.0  # Time waiting in queue
    processing_delay: float = 0.0  # Computation time
    total_delay: float = 0.0  # Total end-to-end delay
    
    def __post_init__(self):
        """Calculate total delay."""
        self.total_delay = (self.propagation_delay + self.transmission_delay + 
                          self.queuing_delay + self.processing_delay)


@dataclass
class HopLatency:
    """Latency for a single hop in the communication path."""
    source_node: Tuple[Position, NodeType]
    destination_node: Tuple[Position, NodeType]
    data_size: float  # MB
    components: LatencyComponents
    success: bool = True


class LatencyModel:
    """Comprehensive latency model for SAGIN tasks."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.comm_model = CommunicationModel(system_params)
        
        # Processing delay parameters
        self.processing_overhead = 0.001  # Fixed processing overhead (seconds)
        self.context_switch_delay = 0.0001  # Context switching delay (seconds)
        
        # Queuing model parameters
        self.queue_service_discipline = "FIFO"  # First-In-First-Out
        
    def calculate_propagation_delay(self, source_pos: Position, dest_pos: Position) -> float:
        """Calculate propagation delay between two positions."""
        distance = source_pos.distance_to(dest_pos)
        return distance / self.system_params.propagation_speed
    
    def calculate_transmission_delay(self, data_size: float, source_pos: Position,
                                   dest_pos: Position, source_type: NodeType,
                                   dest_type: NodeType) -> float:
        """Calculate transmission delay for data transfer."""
        return self.comm_model.calculate_transmission_delay(
            data_size, source_pos, dest_pos, source_type, dest_type
        )
    
    def calculate_queuing_delay(self, task: Task, processing_node_id: int,
                              current_queue: List[Task], cpu_capacity: float) -> float:
        """Calculate queuing delay at a processing node."""
        if not current_queue:
            return 0.0
        
        # Calculate total workload ahead of this task
        total_workload_ahead = 0.0
        for queued_task in current_queue:
            if queued_task.creation_time < task.creation_time:
                total_workload_ahead += queued_task.cpu_cycles
        
        # Queuing delay = Workload ahead / Processing capacity
        queuing_delay = total_workload_ahead / cpu_capacity
        
        return queuing_delay
    
    def calculate_processing_delay(self, task: Task, cpu_capacity: float) -> float:
        """Calculate processing delay for task execution."""
        # Basic processing delay
        basic_delay = task.cpu_cycles / cpu_capacity
        
        # Add processing overhead
        total_delay = basic_delay + self.processing_overhead
        
        return total_delay
    
    def calculate_single_hop_latency(self, source_pos: Position, dest_pos: Position,
                                   source_type: NodeType, dest_type: NodeType,
                                   data_size: float) -> HopLatency:
        """Calculate latency for a single communication hop."""
        # Check if link is available
        if not self.comm_model.is_link_available(source_pos, dest_pos, source_type, dest_type):
            return HopLatency(
                source_node=(source_pos, source_type),
                destination_node=(dest_pos, dest_type),
                data_size=data_size,
                components=LatencyComponents(
                    propagation_delay=float('inf'),
                    transmission_delay=float('inf'),
                    queuing_delay=0.0,
                    processing_delay=0.0,
                    total_delay=float('inf')
                ),
                success=False
            )
        
        # Calculate individual components
        prop_delay = self.calculate_propagation_delay(source_pos, dest_pos)
        trans_delay = self.calculate_transmission_delay(
            data_size, source_pos, dest_pos, source_type, dest_type
        )
        
        components = LatencyComponents(
            propagation_delay=prop_delay,
            transmission_delay=trans_delay,
            queuing_delay=0.0,  # No queuing for communication
            processing_delay=0.0
        )
        
        return HopLatency(
            source_node=(source_pos, source_type),
            destination_node=(dest_pos, dest_type),
            data_size=data_size,
            components=components,
            success=True
        )
    
    def calculate_multi_hop_latency(self, task: Task, 
                                  communication_path: List[Tuple[Position, NodeType]]) -> List[HopLatency]:
        """Calculate latency for multi-hop communication path."""
        hop_latencies = []
        
        if len(communication_path) < 2:
            return hop_latencies
        
        # Calculate latency for each hop
        for i in range(len(communication_path) - 1):
            source_pos, source_type = communication_path[i]
            dest_pos, dest_type = communication_path[i + 1]
            
            # Use input data size for first hop, output data size for return
            if i == 0:
                data_size = task.data_size_in
            else:
                data_size = task.data_size_out
            
            hop_latency = self.calculate_single_hop_latency(
                source_pos, dest_pos, source_type, dest_type, data_size
            )
            
            hop_latencies.append(hop_latency)
        
        return hop_latencies
    
    def calculate_end_to_end_latency(self, task: Task, 
                                   communication_path: List[Tuple[Position, NodeType]],
                                   processing_node_info: Dict[str, Any]) -> LatencyComponents:
        """Calculate complete end-to-end latency for a task."""
        total_components = LatencyComponents()
        
        # 1. Calculate communication latency
        hop_latencies = self.calculate_multi_hop_latency(task, communication_path)
        
        for hop in hop_latencies:
            if not hop.success:
                # If any hop fails, return infinite latency
                return LatencyComponents(
                    propagation_delay=float('inf'),
                    transmission_delay=float('inf'),
                    queuing_delay=float('inf'),
                    processing_delay=float('inf'),
                    total_delay=float('inf')
                )
            
            total_components.propagation_delay += hop.components.propagation_delay
            total_components.transmission_delay += hop.components.transmission_delay
        
        # 2. Calculate processing latency
        processing_node_id = processing_node_info.get('node_id')
        current_queue = processing_node_info.get('current_queue', [])
        cpu_capacity = processing_node_info.get('cpu_capacity', 1e9)
        
        # Queuing delay
        total_components.queuing_delay = self.calculate_queuing_delay(
            task, processing_node_id, current_queue, cpu_capacity
        )
        
        # Processing delay
        total_components.processing_delay = self.calculate_processing_delay(
            task, cpu_capacity
        )
        
        # Calculate total delay
        total_components.total_delay = (
            total_components.propagation_delay +
            total_components.transmission_delay +
            total_components.queuing_delay +
            total_components.processing_delay
        )
        
        return total_components
    
    def estimate_latency_for_decision(self, task: Task, decision_type: str,
                                    network_state: Dict[str, Any]) -> LatencyComponents:
        """Estimate latency for different offloading decisions."""
        if decision_type == "local":
            return self._estimate_local_processing_latency(task, network_state)
        elif decision_type == "dynamic":
            return self._estimate_dynamic_uav_latency(task, network_state)
        elif decision_type == "satellite":
            return self._estimate_satellite_latency(task, network_state)
        else:
            return LatencyComponents(total_delay=float('inf'))
    
    def _estimate_local_processing_latency(self, task: Task, 
                                         network_state: Dict[str, Any]) -> LatencyComponents:
        """Estimate latency for local processing at static UAV."""
        static_uav_info = network_state.get('static_uav', {})
        
        # No communication delay for local processing
        components = LatencyComponents(
            propagation_delay=0.0,
            transmission_delay=0.0
        )
        
        # Calculate queuing and processing delays
        current_queue = static_uav_info.get('current_queue', [])
        cpu_capacity = static_uav_info.get('cpu_capacity', 1e9)
        
        components.queuing_delay = self.calculate_queuing_delay(
            task, static_uav_info.get('node_id'), current_queue, cpu_capacity
        )
        
        components.processing_delay = self.calculate_processing_delay(task, cpu_capacity)
        
        components.total_delay = (components.queuing_delay + components.processing_delay)
        
        return components
    
    def _estimate_dynamic_uav_latency(self, task: Task, 
                                    network_state: Dict[str, Any]) -> LatencyComponents:
        """Estimate latency for dynamic UAV processing."""
        static_uav_info = network_state.get('static_uav', {})
        dynamic_uav_info = network_state.get('best_dynamic_uav', {})
        
        if not dynamic_uav_info:
            return LatencyComponents(total_delay=float('inf'))
        
        # Communication path: Vehicle -> Static UAV -> Dynamic UAV
        vehicle_pos = Position(task.source_vehicle_id, 0, 0)  # Simplified
        static_uav_pos = static_uav_info.get('position', Position(0, 0, 100))
        dynamic_uav_pos = dynamic_uav_info.get('position', Position(0, 0, 100))
        
        communication_path = [
            (vehicle_pos, NodeType.VEHICLE),
            (static_uav_pos, NodeType.STATIC_UAV),
            (dynamic_uav_pos, NodeType.DYNAMIC_UAV)
        ]
        
        # Calculate end-to-end latency
        processing_info = {
            'node_id': dynamic_uav_info.get('node_id'),
            'current_queue': dynamic_uav_info.get('current_queue', []),
            'cpu_capacity': dynamic_uav_info.get('cpu_capacity', 1e9)
        }
        
        return self.calculate_end_to_end_latency(task, communication_path, processing_info)
    
    def _estimate_satellite_latency(self, task: Task, 
                                  network_state: Dict[str, Any]) -> LatencyComponents:
        """Estimate latency for satellite processing."""
        static_uav_info = network_state.get('static_uav', {})
        satellite_info = network_state.get('best_satellite', {})
        
        if not satellite_info:
            return LatencyComponents(total_delay=float('inf'))
        
        # Communication path: Vehicle -> Static UAV -> Satellite
        vehicle_pos = Position(task.source_vehicle_id, 0, 0)  # Simplified
        static_uav_pos = static_uav_info.get('position', Position(0, 0, 100))
        satellite_pos = satellite_info.get('position', Position(0, 0, 550000))
        
        communication_path = [
            (vehicle_pos, NodeType.VEHICLE),
            (static_uav_pos, NodeType.STATIC_UAV),
            (satellite_pos, NodeType.SATELLITE)
        ]
        
        # Calculate end-to-end latency
        processing_info = {
            'node_id': satellite_info.get('node_id'),
            'current_queue': satellite_info.get('current_queue', []),
            'cpu_capacity': satellite_info.get('cpu_capacity', 5e9)
        }
        
        return self.calculate_end_to_end_latency(task, communication_path, processing_info)
    
    def analyze_latency_breakdown(self, latency_components: LatencyComponents) -> Dict[str, float]:
        """Analyze latency breakdown as percentages."""
        total = latency_components.total_delay
        
        if total <= 0:
            return {}
        
        return {
            'propagation_percentage': (latency_components.propagation_delay / total) * 100,
            'transmission_percentage': (latency_components.transmission_delay / total) * 100,
            'queuing_percentage': (latency_components.queuing_delay / total) * 100,
            'processing_percentage': (latency_components.processing_delay / total) * 100
        }
    
    def predict_latency_violation(self, task: Task, estimated_latency: LatencyComponents,
                                current_time: float) -> bool:
        """Predict if task will violate its deadline."""
        completion_time = current_time + estimated_latency.total_delay
        return completion_time > task.deadline
    
    def calculate_latency_slack(self, task: Task, estimated_latency: LatencyComponents,
                              current_time: float) -> float:
        """Calculate slack time before deadline violation."""
        completion_time = current_time + estimated_latency.total_delay
        return task.deadline - completion_time
    
    def get_latency_statistics(self, completed_tasks: List[Task]) -> Dict[str, float]:
        """Calculate latency statistics for completed tasks."""
        if not completed_tasks:
            return {}
        
        latencies = []
        violations = 0
        
        for task in completed_tasks:
            if task.completion_time > 0 and task.creation_time > 0:
                latency = task.completion_time - task.creation_time
                latencies.append(latency)
                
                if task.completion_time > task.deadline:
                    violations += 1
        
        if not latencies:
            return {}
        
        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies),
            'percentile_95': np.percentile(latencies, 95),
            'percentile_99': np.percentile(latencies, 99),
            'violation_rate': violations / len(completed_tasks),
            'total_tasks': len(completed_tasks)
        }
    
    def optimize_communication_path(self, task: Task, available_nodes: List[Tuple[Position, NodeType]],
                                  processing_node: Tuple[Position, NodeType]) -> List[Tuple[Position, NodeType]]:
        """Find optimal communication path to minimize latency."""
        # Simple greedy approach - can be improved with more sophisticated algorithms
        source_pos = Position(task.source_vehicle_id, 0, 0)  # Simplified
        dest_pos, dest_type = processing_node
        
        # Try direct path first
        direct_hop = self.calculate_single_hop_latency(
            source_pos, dest_pos, NodeType.VEHICLE, dest_type, task.data_size_in
        )
        
        if direct_hop.success:
            return [(source_pos, NodeType.VEHICLE), (dest_pos, dest_type)]
        
        # Find best intermediate node
        best_path = None
        best_latency = float('inf')
        
        for inter_pos, inter_type in available_nodes:
            # Check first hop: source -> intermediate
            hop1 = self.calculate_single_hop_latency(
                source_pos, inter_pos, NodeType.VEHICLE, inter_type, task.data_size_in
            )
            
            if not hop1.success:
                continue
            
            # Check second hop: intermediate -> destination
            hop2 = self.calculate_single_hop_latency(
                inter_pos, dest_pos, inter_type, dest_type, task.data_size_out
            )
            
            if not hop2.success:
                continue
            
            # Calculate total communication latency
            total_latency = hop1.components.total_delay + hop2.components.total_delay
            
            if total_latency < best_latency:
                best_latency = total_latency
                best_path = [
                    (source_pos, NodeType.VEHICLE),
                    (inter_pos, inter_type),
                    (dest_pos, dest_type)
                ]
        
        return best_path or [(source_pos, NodeType.VEHICLE), (dest_pos, dest_type)]


class DetailedLatencyBreakdown:
    """Detailed latency breakdown for comprehensive analysis."""
    
    def __init__(self):
        self.breakdown_history = []
    
    def calculate_detailed_latency(self, task: Task, network_path: List[Tuple[Position, NodeType]],
                                 processing_node_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed latency breakdown with all components."""
        breakdown = {
            # Communication delays
            'uplink_propagation': 0.0,
            'uplink_transmission': 0.0,
            'downlink_propagation': 0.0,
            'downlink_transmission': 0.0,
            
            # Processing delays
            'queue_waiting': 0.0,
            'context_switch': 0.0,
            'computation': 0.0,
            'memory_access': 0.0,
            
            # System delays
            'protocol_overhead': 0.0,
            'error_handling': 0.0,
            'synchronization': 0.0,
            
            # Total components
            'total_communication': 0.0,
            'total_processing': 0.0,
            'total_system': 0.0,
            'total_latency': 0.0
        }
        
        # Calculate communication delays
        if len(network_path) >= 2:
            # Uplink: source to processing node
            source_pos, source_type = network_path[0]
            process_pos, process_type = network_path[-1]
            
            # Propagation delay
            distance = source_pos.distance_to(process_pos)
            breakdown['uplink_propagation'] = distance / 3e8  # Speed of light
            
            # Transmission delay (simplified)
            data_rate_mbps = 10.0  # Assume 10 Mbps for calculation
            breakdown['uplink_transmission'] = (task.data_size_in * 8) / data_rate_mbps
            
            # Downlink delays (for result transmission)
            breakdown['downlink_propagation'] = breakdown['uplink_propagation']
            breakdown['downlink_transmission'] = (task.data_size_out * 8) / data_rate_mbps
        
        # Calculate processing delays
        cpu_capacity = processing_node_info.get('cpu_capacity', 1e9)
        current_queue = processing_node_info.get('current_queue', [])
        
        # Queue waiting time
        total_queue_cycles = sum(t.cpu_cycles for t in current_queue)
        breakdown['queue_waiting'] = total_queue_cycles / cpu_capacity
        
        # Context switch overhead
        breakdown['context_switch'] = 0.0001  # 0.1ms
        
        # Computation time
        breakdown['computation'] = task.cpu_cycles / cpu_capacity
        
        # Memory access overhead (5% of computation time)
        breakdown['memory_access'] = breakdown['computation'] * 0.05
        
        # System delays
        breakdown['protocol_overhead'] = 0.001  # 1ms
        breakdown['error_handling'] = 0.0005   # 0.5ms
        breakdown['synchronization'] = 0.0002  # 0.2ms
        
        # Calculate totals
        breakdown['total_communication'] = (breakdown['uplink_propagation'] + 
                                          breakdown['uplink_transmission'] + 
                                          breakdown['downlink_propagation'] + 
                                          breakdown['downlink_transmission'])
        
        breakdown['total_processing'] = (breakdown['queue_waiting'] + 
                                       breakdown['context_switch'] + 
                                       breakdown['computation'] + 
                                       breakdown['memory_access'])
        
        breakdown['total_system'] = (breakdown['protocol_overhead'] + 
                                   breakdown['error_handling'] + 
                                   breakdown['synchronization'])
        
        breakdown['total_latency'] = (breakdown['total_communication'] + 
                                    breakdown['total_processing'] + 
                                    breakdown['total_system'])
        
        return breakdown
    
    def analyze_latency_bottlenecks(self, breakdown: Dict[str, float]) -> Dict[str, Any]:
        """Analyze latency breakdown to identify bottlenecks."""
        total = breakdown['total_latency']
        if total <= 0:
            return {}
        
        # Calculate percentages
        percentages = {
            'communication_percentage': (breakdown['total_communication'] / total) * 100,
            'processing_percentage': (breakdown['total_processing'] / total) * 100,
            'system_percentage': (breakdown['total_system'] / total) * 100
        }
        
        # Identify primary bottleneck
        bottleneck = max(percentages.items(), key=lambda x: x[1])
        
        # Detailed component analysis
        detailed_breakdown = {}
        for key, value in breakdown.items():
            if key.startswith('total_'):
                continue
            detailed_breakdown[key] = {
                'value_seconds': value,
                'percentage': (value / total) * 100 if total > 0 else 0
            }
        
        return {
            'bottleneck_type': bottleneck[0].replace('_percentage', ''),
            'bottleneck_percentage': bottleneck[1],
            'component_percentages': percentages,
            'detailed_breakdown': detailed_breakdown,
            'optimization_recommendations': self._get_optimization_recommendations(breakdown)
        }
    
    def _get_optimization_recommendations(self, breakdown: Dict[str, float]) -> List[str]:
        """Get optimization recommendations based on latency breakdown."""
        recommendations = []
        total = breakdown['total_latency']
        
        if total <= 0:
            return recommendations
        
        # Communication optimization
        comm_percentage = (breakdown['total_communication'] / total) * 100
        if comm_percentage > 50:
            recommendations.append("Consider edge computing to reduce communication latency")
            if breakdown['uplink_propagation'] > breakdown['uplink_transmission']:
                recommendations.append("Propagation delay dominates - consider closer processing nodes")
            else:
                recommendations.append("Transmission delay dominates - consider higher bandwidth links")
        
        # Processing optimization
        proc_percentage = (breakdown['total_processing'] / total) * 100
        if proc_percentage > 50:
            if breakdown['queue_waiting'] > breakdown['computation']:
                recommendations.append("Queue waiting dominates - consider load balancing")
            else:
                recommendations.append("Computation time dominates - consider more powerful processing nodes")
        
        # System optimization
        sys_percentage = (breakdown['total_system'] / total) * 100
        if sys_percentage > 20:
            recommendations.append("System overhead is high - optimize protocol stack")
        
        return recommendations


class AdvancedLatencyModel(LatencyModel):
    """Advanced latency model with detailed breakdown capabilities."""
    
    def __init__(self, system_params: SystemParameters):
        super().__init__(system_params)
        self.detailed_breakdown = DetailedLatencyBreakdown()
        
        # Advanced queuing model parameters
        self.service_disciplines = {
            'FIFO': self._fifo_queue_delay,
            'SJF': self._sjf_queue_delay,
            'EDF': self._edf_queue_delay,
            'Priority': self._priority_queue_delay
        }
        
        # Cache for performance
        self.latency_cache = {}
        self.cache_size = 1000
    
    def calculate_advanced_end_to_end_latency(self, task: Task, 
                                            network_path: List[Tuple[Position, NodeType]],
                                            processing_node_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced end-to-end latency with detailed breakdown."""
        # Check cache first
        cache_key = (task.id, len(network_path), processing_node_info.get('node_id'))
        if cache_key in self.latency_cache:
            return self.latency_cache[cache_key]
        
        # Calculate detailed breakdown
        breakdown = self.detailed_breakdown.calculate_detailed_latency(
            task, network_path, processing_node_info
        )
        
        # Analyze bottlenecks
        analysis = self.detailed_breakdown.analyze_latency_bottlenecks(breakdown)
        
        # Create comprehensive result
        result = {
            'latency_breakdown': breakdown,
            'bottleneck_analysis': analysis,
            'total_latency': breakdown['total_latency'],
            'meets_deadline': breakdown['total_latency'] <= (task.deadline - task.creation_time),
            'slack_time': (task.deadline - task.creation_time) - breakdown['total_latency']
        }
        
        # Cache result
        if len(self.latency_cache) < self.cache_size:
            self.latency_cache[cache_key] = result
        
        return result
    
    def _fifo_queue_delay(self, task: Task, queue: List[Task], cpu_capacity: float) -> float:
        """Calculate FIFO queue delay."""
        delay = 0.0
        for queued_task in queue:
            if queued_task.creation_time < task.creation_time:
                delay += queued_task.cpu_cycles / cpu_capacity
        return delay
    
    def _sjf_queue_delay(self, task: Task, queue: List[Task], cpu_capacity: float) -> float:
        """Calculate Shortest Job First queue delay."""
        # Tasks with shorter processing time go first
        shorter_tasks = [t for t in queue if t.cpu_cycles < task.cpu_cycles]
        return sum(t.cpu_cycles for t in shorter_tasks) / cpu_capacity
    
    def _edf_queue_delay(self, task: Task, queue: List[Task], cpu_capacity: float) -> float:
        """Calculate Earliest Deadline First queue delay."""
        # Tasks with earlier deadlines go first
        earlier_deadline_tasks = [t for t in queue if t.deadline < task.deadline]
        return sum(t.cpu_cycles for t in earlier_deadline_tasks) / cpu_capacity
    
    def _priority_queue_delay(self, task: Task, queue: List[Task], cpu_capacity: float) -> float:
        """Calculate priority-based queue delay."""
        # Assume priority is inversely related to data size (smaller tasks have higher priority)
        higher_priority_tasks = [t for t in queue if t.data_size_in < task.data_size_in]
        return sum(t.cpu_cycles for t in higher_priority_tasks) / cpu_capacity
    
    def calculate_total_latency_paper_formula(self, task: Task, communication_path: List[Tuple[Position, NodeType]],
                                            processing_node_info: Dict[str, Any]) -> LatencyComponents:
        """Calculate total latency using exact paper formula: T_total = T_prop + T_trans + T_queue + T_comp"""
        
        # Initialize components
        total_prop_delay = 0.0
        total_trans_delay = 0.0
        total_queue_delay = 0.0
        total_comp_delay = 0.0
        
        # 1. Propagation delay: T_prop = Σ d_ik / v_prop (for each hop in path)
        v_prop = self.system_params.propagation_speed  # speed of light
        for i in range(len(communication_path) - 1):
            source_pos, _ = communication_path[i]
            dest_pos, _ = communication_path[i + 1]
            distance = source_pos.distance_to(dest_pos)
            total_prop_delay += distance / v_prop
        
        # 2. Transmission delay: T_trans = Σ D_in / R_ik (for each hop)
        for i in range(len(communication_path) - 1):
            source_pos, source_type = communication_path[i]
            dest_pos, dest_type = communication_path[i + 1]
            
            # Use input data size for uplink, output for downlink
            data_size = task.data_size_in if i == 0 else task.data_size_out
            
            # Calculate transmission rate R_ik using Shannon formula
            data_rate = self.comm_model.calculate_data_rate(source_pos, dest_pos, source_type, dest_type)
            
            if data_rate > 0:
                data_size_mbits = data_size * 8  # Convert MB to Mbits
                total_trans_delay += data_size_mbits / data_rate
            else:
                return LatencyComponents(total_delay=float('inf'))
        
        # 3. Queuing delay: T_queue (time waiting in processing queue)
        processing_node_id = processing_node_info.get('node_id')
        current_queue = processing_node_info.get('current_queue', [])
        cpu_capacity = processing_node_info.get('cpu_capacity', 1e9)
        
        total_queue_delay = self.calculate_queuing_delay(task, processing_node_id, current_queue, cpu_capacity)
        
        # 4. Computing delay: T_comp = C_j / Θ_node (CPU cycles / processing capacity)
        total_comp_delay = task.cpu_cycles / cpu_capacity
        
        # Return components following paper formula
        components = LatencyComponents(
            propagation_delay=total_prop_delay,
            transmission_delay=total_trans_delay,
            queuing_delay=total_queue_delay,
            processing_delay=total_comp_delay
        )
        
        return components
