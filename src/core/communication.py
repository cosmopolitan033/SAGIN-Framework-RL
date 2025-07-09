"""
Communication models for the SAGIN system.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.types import Position, NodeType, SystemParameters


class LinkType(Enum):
    """Types of communication links."""
    V2U = "vehicle_to_uav"          # Vehicle to UAV
    U2V = "uav_to_vehicle"          # UAV to Vehicle
    U2U = "uav_to_uav"              # UAV to UAV
    U2S = "uav_to_satellite"        # UAV to Satellite
    S2U = "satellite_to_uav"        # Satellite to UAV
    S2S = "satellite_to_satellite"  # Satellite to Satellite


@dataclass
class ChannelParameters:
    """Channel parameters for different link types."""
    frequency: float  # GHz
    bandwidth: float  # MHz
    transmit_power: float  # W
    antenna_gain_tx: float  # dBi
    antenna_gain_rx: float  # dBi
    noise_figure: float  # dB
    implementation_loss: float  # dB


# Default channel parameters for different link types
DEFAULT_CHANNEL_PARAMS = {
    LinkType.V2U: ChannelParameters(
        frequency=2.4, bandwidth=20.0, transmit_power=1.0,
        antenna_gain_tx=0.0, antenna_gain_rx=10.0,
        noise_figure=3.0, implementation_loss=2.0
    ),
    LinkType.U2V: ChannelParameters(
        frequency=2.4, bandwidth=20.0, transmit_power=10.0,
        antenna_gain_tx=10.0, antenna_gain_rx=0.0,
        noise_figure=3.0, implementation_loss=2.0
    ),
    LinkType.U2U: ChannelParameters(
        frequency=5.8, bandwidth=40.0, transmit_power=10.0,
        antenna_gain_tx=10.0, antenna_gain_rx=10.0,
        noise_figure=2.0, implementation_loss=1.0
    ),
    LinkType.U2S: ChannelParameters(
        frequency=28.0, bandwidth=100.0, transmit_power=20.0,
        antenna_gain_tx=20.0, antenna_gain_rx=30.0,
        noise_figure=2.0, implementation_loss=1.0
    ),
    LinkType.S2U: ChannelParameters(
        frequency=28.0, bandwidth=100.0, transmit_power=100.0,
        antenna_gain_tx=30.0, antenna_gain_rx=20.0,
        noise_figure=2.0, implementation_loss=1.0
    ),
    LinkType.S2S: ChannelParameters(
        frequency=60.0, bandwidth=200.0, transmit_power=100.0,
        antenna_gain_tx=30.0, antenna_gain_rx=30.0,
        noise_figure=1.0, implementation_loss=0.5
    )
}


class PathLossModel:
    """Path loss models for different communication scenarios."""
    
    @staticmethod
    def free_space_path_loss(distance: float, frequency: float) -> float:
        """Calculate free space path loss in dB."""
        if distance <= 0 or frequency <= 0:
            return float('inf')
        
        # FSPL = 20 * log10(d) + 20 * log10(f) + 20 * log10(4π/c)
        # where d is in meters, f is in Hz, c is speed of light
        fspl = 20 * math.log10(distance) + 20 * math.log10(frequency * 1e9) + 20 * math.log10(4 * math.pi / 3e8)
        return fspl
    
    @staticmethod
    def air_to_ground_path_loss(distance: float, height: float, frequency: float,
                              environment: str = "suburban") -> float:
        """Calculate air-to-ground path loss using extended models."""
        if distance <= 0 or height <= 0 or frequency <= 0:
            return float('inf')
        
        # Elevation angle
        elevation_angle = math.degrees(math.atan(height / distance))
        
        # Line-of-sight probability
        if environment == "urban":
            a, b = 9.61, 0.16
        elif environment == "suburban":
            a, b = 4.88, 0.43
        else:  # rural
            a, b = 0.1, 0.59
        
        p_los = 1 / (1 + a * math.exp(-b * (elevation_angle - a)))
        
        # Path loss components
        d_3d = math.sqrt(distance**2 + height**2)
        fspl = PathLossModel.free_space_path_loss(d_3d, frequency)
        
        # Additional losses
        eta_los = 1.0  # dB (LoS)
        eta_nlos = 20.0  # dB (NLoS)
        
        # Combined path loss
        path_loss = fspl + p_los * eta_los + (1 - p_los) * eta_nlos
        
        return path_loss
    
    @staticmethod
    def satellite_path_loss(distance: float, frequency: float,
                          atmospheric_loss: float = 0.5) -> float:
        """Calculate satellite communication path loss."""
        if distance <= 0 or frequency <= 0:
            return float('inf')
        
        # Free space path loss
        fspl = PathLossModel.free_space_path_loss(distance, frequency)
        
        # Add atmospheric and other losses
        total_loss = fspl + atmospheric_loss
        
        return total_loss


class CommunicationModel:
    """Comprehensive communication model with advanced features for calculating data rates and delays."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.channel_params = DEFAULT_CHANNEL_PARAMS.copy()
        
        # Thermal noise at room temperature
        self.thermal_noise_density = -174  # dBm/Hz at 290K
        
        # Advanced model parameters
        self.enable_fading = True
        self.enable_interference = True
        self.enable_adaptive_modulation = True
        
        # Fading parameters
        self.rayleigh_scale = 1.0
        self.rician_k_factor = 10.0  # dB
        
        # Interference modeling
        self.interference_sources = []
        
        # Adaptive modulation thresholds (SNR in dB)
        self.modulation_thresholds = {
            4: 6.0,    # QPSK
            16: 12.0,  # 16-QAM
            64: 18.0,  # 64-QAM
            256: 24.0  # 256-QAM
        }
    
    def set_channel_parameters(self, link_type: LinkType, params: ChannelParameters):
        """Set custom channel parameters for a link type."""
        self.channel_params[link_type] = params
    
    def calculate_channel_gain(self, tx_pos: Position, rx_pos: Position,
                             tx_type: NodeType, rx_type: NodeType) -> float:
        """Calculate channel gain between two nodes."""
        # Determine link type
        link_type = self._determine_link_type(tx_type, rx_type)
        
        # Get channel parameters
        params = self.channel_params[link_type]
        
        # Calculate distance
        distance = tx_pos.distance_to(rx_pos)
        
        # Calculate path loss based on link type
        if link_type in [LinkType.V2U, LinkType.U2V]:
            # Air-to-ground path loss
            height = abs(tx_pos.z - rx_pos.z)
            horizontal_distance = math.sqrt((tx_pos.x - rx_pos.x)**2 + (tx_pos.y - rx_pos.y)**2)
            path_loss = PathLossModel.air_to_ground_path_loss(
                horizontal_distance, height, params.frequency
            )
        elif link_type == LinkType.U2U:
            # Free space path loss for UAV-to-UAV
            path_loss = PathLossModel.free_space_path_loss(distance, params.frequency)
        elif link_type in [LinkType.U2S, LinkType.S2U, LinkType.S2S]:
            # Satellite communication path loss
            path_loss = PathLossModel.satellite_path_loss(distance, params.frequency)
        else:
            # Default to free space
            path_loss = PathLossModel.free_space_path_loss(distance, params.frequency)
        
        # Calculate total channel gain
        # Gain = Tx_gain + Rx_gain - Path_loss - Implementation_loss
        channel_gain = (params.antenna_gain_tx + params.antenna_gain_rx - 
                       path_loss - params.implementation_loss)
        
        return channel_gain
    
    def calculate_data_rate(self, tx_pos: Position, rx_pos: Position,
                          tx_type: NodeType, rx_type: NodeType) -> float:
        """Calculate achievable data rate using advanced Shannon capacity with realistic constraints."""
        # Get advanced data rate calculation
        advanced_result = self.calculate_advanced_data_rate(tx_pos, rx_pos, tx_type, rx_type)
        return advanced_result['data_rate_mbps']
    
    def calculate_fading_factor(self, link_type: LinkType, distance: float) -> float:
        """Calculate fading factor for the channel."""
        if not self.enable_fading:
            return 1.0
        
        if link_type in [LinkType.U2S, LinkType.S2U, LinkType.S2S]:
            # Rician fading for satellite links (with strong LoS component)
            k_linear = 10 ** (self.rician_k_factor / 10)
            # Simplified Rician fading approximation
            fading_db = -np.random.exponential(scale=1.0) + k_linear / (k_linear + 1)
        else:
            # Rayleigh fading for terrestrial links
            rayleigh_sample = np.random.rayleigh(scale=self.rayleigh_scale)
            fading_db = 20 * math.log10(rayleigh_sample) if rayleigh_sample > 0 else -50
        
        return 10 ** (fading_db / 10)
    
    def calculate_interference_power(self, rx_pos: Position, rx_type: NodeType) -> float:
        """Calculate total interference power at receiver."""
        if not self.enable_interference:
            return 0.0
        
        total_interference = 0.0
        
        # Add interference from other active transmitters
        for interferer_pos, interferer_type, interferer_power in self.interference_sources:
            if interferer_type != rx_type:  # Only consider different node types
                distance = rx_pos.distance_to(interferer_pos)
                if distance > 0:
                    # Simple power law path loss for interference
                    path_loss_linear = (distance / 1000) ** 2  # Normalized to 1km
                    interference_power = interferer_power / path_loss_linear
                    total_interference += interference_power
        
        return total_interference
    
    def select_adaptive_modulation(self, snr_db: float) -> Tuple[int, float]:
        """Select optimal modulation order and coding rate based on SNR."""
        if not self.enable_adaptive_modulation:
            return 4, 0.5  # Default QPSK with 1/2 coding rate
        
        # Select highest order modulation that meets SNR threshold
        selected_order = 4
        for order, threshold in sorted(self.modulation_thresholds.items()):
            if snr_db >= threshold:
                selected_order = order
            else:
                break
        
        # Adaptive coding rate based on channel quality
        if snr_db >= 20:
            coding_rate = 0.9
        elif snr_db >= 15:
            coding_rate = 0.75
        elif snr_db >= 10:
            coding_rate = 0.5
        else:
            coding_rate = 0.33
        
        return selected_order, coding_rate
    
    def calculate_advanced_data_rate(self, tx_pos: Position, rx_pos: Position,
                                   tx_type: NodeType, rx_type: NodeType) -> Dict[str, float]:
        """Calculate data rate with advanced channel modeling."""
        # Get basic parameters
        link_type = self._determine_link_type(tx_type, rx_type)
        params = self.channel_params[link_type]
        
        # Calculate basic channel gain
        channel_gain_db = self.calculate_channel_gain(tx_pos, rx_pos, tx_type, rx_type)
        channel_gain_linear = 10 ** (channel_gain_db / 10)
        
        # Add fading effects
        distance = tx_pos.distance_to(rx_pos)
        fading_factor = self.calculate_fading_factor(link_type, distance)
        
        # Calculate interference
        interference_power = self.calculate_interference_power(rx_pos, rx_type)
        
        # Calculate noise power
        noise_power_dbm = (self.thermal_noise_density + 
                          10 * math.log10(params.bandwidth * 1e6) + 
                          params.noise_figure)
        noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)
        
        # Total noise + interference
        total_noise_power = noise_power_w + interference_power
        
        # Calculate received power with fading
        tx_power_w = params.transmit_power
        rx_power_w = tx_power_w * channel_gain_linear * fading_factor
        
        # Calculate SINR (Signal-to-Interference-plus-Noise Ratio)
        sinr = rx_power_w / total_noise_power if total_noise_power > 0 else 0
        sinr_db = 10 * math.log10(sinr) if sinr > 0 else -float('inf')
        
        # Adaptive modulation and coding
        mod_order, coding_rate = self.select_adaptive_modulation(sinr_db)
        
        # Calculate spectral efficiency
        bits_per_symbol = math.log2(mod_order)
        spectral_efficiency = bits_per_symbol * coding_rate
        
        # Advanced Shannon capacity with practical constraints
        theoretical_rate = params.bandwidth * math.log2(1 + sinr) if sinr > 0 else 0
        practical_rate = params.bandwidth * spectral_efficiency
        
        # Use minimum of theoretical and practical rates
        final_rate = min(theoretical_rate, practical_rate) if theoretical_rate > 0 else practical_rate
        
        # Apply practical efficiency factor
        practical_efficiency = 0.8
        final_rate = final_rate * practical_efficiency
        
        return {
            'data_rate_mbps': max(0, final_rate),
            'theoretical_rate_mbps': max(0, theoretical_rate),
            'practical_rate_mbps': max(0, practical_rate),
            'sinr_db': sinr_db,
            'fading_factor': fading_factor,
            'interference_power_w': interference_power,
            'modulation_order': mod_order,
            'coding_rate': coding_rate,
            'spectral_efficiency': spectral_efficiency
        }
    
    def is_link_available(self, tx_pos: Position, rx_pos: Position,
                         tx_type: NodeType, rx_type: NodeType) -> bool:
        """Check if communication link is available."""
        data_rate = self.calculate_data_rate(tx_pos, rx_pos, tx_type, rx_type)
        return data_rate >= self.system_params.min_rate_threshold
    
    def calculate_transmission_delay(self, data_size: float, tx_pos: Position,
                                   rx_pos: Position, tx_type: NodeType,
                                   rx_type: NodeType) -> float:
        """Calculate transmission delay for given data size."""
        data_rate = self.calculate_data_rate(tx_pos, rx_pos, tx_type, rx_type)
        
        if data_rate <= 0:
            return float('inf')
        
        # Convert data size from MB to Mbits
        data_size_mbits = data_size * 8
        
        # Transmission delay = Data size / Data rate
        transmission_delay = data_size_mbits / data_rate
        
        return transmission_delay
    
    def calculate_propagation_delay(self, tx_pos: Position, rx_pos: Position) -> float:
        """Calculate propagation delay."""
        distance = tx_pos.distance_to(rx_pos)
        return distance / self.system_params.propagation_speed
    
    def _determine_link_type(self, tx_type: NodeType, rx_type: NodeType) -> LinkType:
        """Determine link type based on node types."""
        if tx_type == NodeType.VEHICLE and rx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV]:
            return LinkType.V2U
        elif tx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV] and rx_type == NodeType.VEHICLE:
            return LinkType.U2V
        elif (tx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV] and 
              rx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV]):
            return LinkType.U2U
        elif tx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV] and rx_type == NodeType.SATELLITE:
            return LinkType.U2S
        elif tx_type == NodeType.SATELLITE and rx_type in [NodeType.STATIC_UAV, NodeType.DYNAMIC_UAV]:
            return LinkType.S2U
        elif tx_type == NodeType.SATELLITE and rx_type == NodeType.SATELLITE:
            return LinkType.S2S
        else:
            # Default to U2U for unknown combinations
            return LinkType.U2U
    
    def get_link_quality_metrics(self, tx_pos: Position, rx_pos: Position,
                               tx_type: NodeType, rx_type: NodeType) -> Dict[str, float]:
        """Get comprehensive link quality metrics."""
        link_type = self._determine_link_type(tx_type, rx_type)
        params = self.channel_params[link_type]
        
        distance = tx_pos.distance_to(rx_pos)
        channel_gain = self.calculate_channel_gain(tx_pos, rx_pos, tx_type, rx_type)
        
        # Get advanced data rate calculation
        advanced_result = self.calculate_advanced_data_rate(tx_pos, rx_pos, tx_type, rx_type)
        
        return {
            'distance': distance,
            'channel_gain_db': channel_gain,
            'data_rate_mbps': advanced_result['data_rate_mbps'],
            'sinr_db': advanced_result['sinr_db'],
            'link_type': link_type.value,
            'frequency_ghz': params.frequency,
            'bandwidth_mhz': params.bandwidth,
            'is_available': advanced_result['data_rate_mbps'] >= self.system_params.min_rate_threshold,
            'fading_factor': advanced_result['fading_factor'],
            'modulation_order': advanced_result['modulation_order'],
            'coding_rate': advanced_result['coding_rate'],
            'spectral_efficiency': advanced_result['spectral_efficiency']
        }
    
    def add_interference_source(self, position: Position, node_type: NodeType, power: float):
        """Add an interference source to the model."""
        self.interference_sources.append((position, node_type, power))
    
    def clear_interference_sources(self):
        """Clear all interference sources."""
        self.interference_sources.clear()
    
    def calculate_link_reliability(self, tx_pos: Position, rx_pos: Position,
                                 tx_type: NodeType, rx_type: NodeType,
                                 required_rate: float) -> float:
        """Calculate link reliability (probability of meeting rate requirement)."""
        # Run multiple realizations to estimate reliability
        num_realizations = 100
        successful_realizations = 0
        
        for _ in range(num_realizations):
            result = self.calculate_advanced_data_rate(tx_pos, rx_pos, tx_type, rx_type)
            if result['data_rate_mbps'] >= required_rate:
                successful_realizations += 1
        
        return successful_realizations / num_realizations
    
    def calculate_energy_consumption(self, data_size: float, tx_pos: Position,
                                   rx_pos: Position, tx_type: NodeType,
                                   rx_type: NodeType) -> float:
        """Calculate energy consumption for data transmission."""
        # Get transmission time
        transmission_time = self.calculate_transmission_delay(
            data_size, tx_pos, rx_pos, tx_type, rx_type
        )
        
        # Get channel parameters
        link_type = self._determine_link_type(tx_type, rx_type)
        params = self.channel_params[link_type]
        
        # Energy = Power × Time
        energy = params.transmit_power * transmission_time
        
        return energy
    
    def find_best_communication_path(self, source_pos: Position, dest_pos: Position,
                                   source_type: NodeType, dest_type: NodeType,
                                   intermediate_nodes: List[Tuple[Position, NodeType]]) -> List[Dict]:
        """Find best multi-hop communication path."""
        # Simple greedy approach - can be improved with graph algorithms
        path = []
        current_pos = source_pos
        current_type = source_type
        
        # Direct link quality
        direct_quality = self.get_link_quality_metrics(source_pos, dest_pos, source_type, dest_type)
        
        if direct_quality['is_available']:
            path.append({
                'from': (source_pos, source_type),
                'to': (dest_pos, dest_type),
                'metrics': direct_quality
            })
        else:
            # Find best intermediate node
            best_intermediate = None
            best_total_rate = 0
            
            for inter_pos, inter_type in intermediate_nodes:
                # Check first hop
                hop1_quality = self.get_link_quality_metrics(current_pos, inter_pos, current_type, inter_type)
                if not hop1_quality['is_available']:
                    continue
                
                # Check second hop
                hop2_quality = self.get_link_quality_metrics(inter_pos, dest_pos, inter_type, dest_type)
                if not hop2_quality['is_available']:
                    continue
                
                # Calculate total rate (bottleneck)
                total_rate = min(hop1_quality['data_rate_mbps'], hop2_quality['data_rate_mbps'])
                
                if total_rate > best_total_rate:
                    best_total_rate = total_rate
                    best_intermediate = (inter_pos, inter_type, hop1_quality, hop2_quality)
            
            if best_intermediate:
                inter_pos, inter_type, hop1_quality, hop2_quality = best_intermediate
                path.append({
                    'from': (source_pos, source_type),
                    'to': (inter_pos, inter_type),
                    'metrics': hop1_quality
                })
                path.append({
                    'from': (inter_pos, inter_type),
                    'to': (dest_pos, dest_type),
                    'metrics': hop2_quality
                })
        
        return path


@dataclass
class ChannelState:
    """Advanced channel state information."""
    fading_factor: float = 1.0  # Rayleigh/Rician fading factor
    interference_power: float = 0.0  # Interference power (W)
    doppler_shift: float = 0.0  # Doppler frequency shift (Hz)
    modulation_order: int = 4  # QAM modulation order (4, 16, 64, 256)
    coding_rate: float = 0.5  # Error correction coding rate
    
    def get_spectral_efficiency(self) -> float:
        """Calculate spectral efficiency based on modulation and coding."""
        bits_per_symbol = math.log2(self.modulation_order)
        return bits_per_symbol * self.coding_rate





class LoadBalancingMetrics:
    """Comprehensive load balancing metrics for SAGIN system."""
    
    def __init__(self):
        self.workload_history = []
        self.utilization_history = []
        
    def calculate_load_imbalance_paper_formula(self, node_workloads: List[float]) -> float:
        """Calculate load imbalance using the exact formula from paper: ΔL = sqrt(1/N * Σ(L_i - L̄)²)"""
        if not node_workloads or len(node_workloads) < 2:
            return 0.0
        
        # Calculate average workload: L̄ = (Σ L_v + Σ L_s) / (|V| + |S|)
        avg_workload = sum(node_workloads) / len(node_workloads)
        
        # Calculate load imbalance: ΔL = sqrt(1/N * Σ(L_i - L̄)²)
        variance_sum = sum((workload - avg_workload) ** 2 for workload in node_workloads)
        load_imbalance = math.sqrt(variance_sum / len(node_workloads))
        
        return load_imbalance
    
    def calculate_load_imbalance_coefficient(self, node_workloads: List[float]) -> float:
        """Calculate load imbalance coefficient (Gini coefficient variant)."""
        if not node_workloads or len(node_workloads) < 2:
            return 0.0
            
        # Sort workloads
        sorted_workloads = sorted(node_workloads)
        n = len(sorted_workloads)
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_workloads)) / (n * np.sum(sorted_workloads)) - (n + 1) / n
        
        return gini
    
    def calculate_fairness_index(self, node_workloads: List[float]) -> float:
        """Calculate Jain's fairness index."""
        if not node_workloads:
            return 1.0
        
        n = len(node_workloads)
        sum_x = sum(node_workloads)
        sum_x_squared = sum(x**2 for x in node_workloads)
        
        if sum_x_squared == 0:
            return 1.0
        
        fairness_index = (sum_x ** 2) / (n * sum_x_squared)
        return fairness_index
    
    def calculate_comprehensive_load_metrics(self, node_workloads: List[float]) -> Dict[str, float]:
        """Calculate comprehensive load balancing metrics."""
        if not node_workloads:
            return {
                'load_imbalance_coefficient': 0.0,
                'fairness_index': 1.0,
                'load_variance': 0.0,
                'peak_to_average_ratio': 1.0,
                'standard_deviation': 0.0,
                'coefficient_of_variation': 0.0
            }
        
        mean_load = np.mean(node_workloads)
        max_load = max(node_workloads)
        std_load = np.std(node_workloads)
        
        return {
            'load_imbalance_paper': self.calculate_load_imbalance_paper_formula(node_workloads),  # Main metric from paper
            'load_imbalance_coefficient': self.calculate_load_imbalance_coefficient(node_workloads),
            'fairness_index': self.calculate_fairness_index(node_workloads),
            'load_variance': np.var(node_workloads) / (mean_load ** 2) if mean_load > 0 else 0.0,
            'peak_to_average_ratio': max_load / mean_load if mean_load > 0 else 1.0,
            'standard_deviation': std_load,
            'coefficient_of_variation': std_load / mean_load if mean_load > 0 else 0.0
        }
    
    def update_history(self, workloads: List[float], utilizations: List[float]):
        """Update historical data for trend analysis."""
        self.workload_history.append(workloads.copy())
        self.utilization_history.append(utilizations.copy())
        
        # Keep only last 100 entries to prevent memory issues
        if len(self.workload_history) > 100:
            self.workload_history.pop(0)
        if len(self.utilization_history) > 100:
            self.utilization_history.pop(0)    


class ShannonCapacityModel:
    """Shannon capacity model with realistic constraints."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.thermal_noise_density = -174  # dBm/Hz at 290K
        
        # Realistic modulation and coding schemes
        self.modulation_schemes = {
            'BPSK': {'bits_per_symbol': 1, 'min_snr_db': 0},
            'QPSK': {'bits_per_symbol': 2, 'min_snr_db': 6},
            '16QAM': {'bits_per_symbol': 4, 'min_snr_db': 12},
            '64QAM': {'bits_per_symbol': 6, 'min_snr_db': 18},
            '256QAM': {'bits_per_symbol': 8, 'min_snr_db': 24}
        }
        
        # Coding rates for different channel conditions
        self.coding_rates = {
            'excellent': 0.95,  # Very high SNR
            'good': 0.8,        # High SNR
            'fair': 0.6,        # Medium SNR
            'poor': 0.4,        # Low SNR
            'very_poor': 0.2    # Very low SNR
        }
    
    def calculate_shannon_capacity(self, bandwidth_hz: float, transmit_power: float, 
                                 channel_gain: float, noise_power: float) -> float:
        """Calculate Shannon capacity according to paper formula: R_ij = W_ij * log2(1 + P_t * G_ij / σ²)"""
        if noise_power <= 0:
            return 0.0
        
        # Shannon capacity formula from paper
        snr_linear = (transmit_power * channel_gain) / noise_power
        capacity_bps = bandwidth_hz * math.log2(1 + snr_linear)
        
        return capacity_bps
    
    def calculate_practical_capacity(self, bandwidth_mhz: float, snr_db: float) -> Dict[str, float]:
        """Calculate practical capacity considering modulation and coding."""
        # Convert to Hz for Shannon formula
        bandwidth_hz = bandwidth_mhz * 1e6
        
        # Calculate theoretical Shannon capacity using paper formula
        snr_linear = 10 ** (snr_db / 10)
        theoretical_capacity_bps = bandwidth_hz * math.log2(1 + snr_linear)
        
        # Select best modulation scheme for practical implementation
        selected_scheme = 'BPSK'
        for scheme, params in self.modulation_schemes.items():
            if snr_db >= params['min_snr_db']:
                selected_scheme = scheme
        
        # Select coding rate based on SNR
        if snr_db >= 25:
            coding_rate = self.coding_rates['excellent']
        elif snr_db >= 20:
            coding_rate = self.coding_rates['good']
        elif snr_db >= 15:
            coding_rate = self.coding_rates['fair']
        elif snr_db >= 10:
            coding_rate = self.coding_rates['poor']
        else:
            coding_rate = self.coding_rates['very_poor']
        
        # Calculate practical capacity with modulation/coding limits
        bits_per_symbol = self.modulation_schemes[selected_scheme]['bits_per_symbol']
        symbol_rate = bandwidth_hz  # Assuming Nyquist rate
        practical_capacity_bps = symbol_rate * bits_per_symbol * coding_rate
        
        # Use minimum of theoretical Shannon and practical modulation limits
        final_capacity = min(theoretical_capacity_bps, practical_capacity_bps)
        
        return {
            'theoretical_capacity_mbps': theoretical_capacity_bps / 1e6,
            'practical_capacity_mbps': final_capacity / 1e6,
            'efficiency': final_capacity / theoretical_capacity_bps if theoretical_capacity_bps > 0 else 0,
            'modulation_scheme': selected_scheme,
            'coding_rate': coding_rate,
            'snr_db': snr_db
        }
    
    def calculate_adaptive_capacity(self, channel_conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive capacity based on dynamic channel conditions."""
        snr_db = channel_conditions.get('snr_db', 0)
        bandwidth_mhz = channel_conditions.get('bandwidth_hz', 20e6) / 1e6
        fading_margin = channel_conditions.get('fading_margin_db', 3)
        interference_margin = channel_conditions.get('interference_margin_db', 2)
        
        # Apply margins
        effective_snr_db = snr_db - fading_margin - interference_margin
        
        # Calculate capacity with margins
        capacity_result = self.calculate_practical_capacity(bandwidth_mhz, effective_snr_db)
        
        # Add reliability factor
        reliability_factor = min(1.0, max(0.1, effective_snr_db / 20))
        capacity_result['reliable_capacity_bps'] = capacity_result['practical_capacity_mbps'] * 1e6 * reliability_factor
        
        return capacity_result

class EnergyModel:
    """Energy consumption model following paper formula: E_v(t+1) = E_v(t) - (E_flight + E_comm + E_comp)"""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        
        # Energy model parameters from paper
        self.P_hover = 500.0  # Hovering power (W)
        self.xi = 10.0  # Energy cost per meter flown (J/m)
        self.eta_comm = 0.1  # Communication energy coefficient (J/MB)
        self.kappa = 1e-9  # Energy per CPU cycle (J/cycle)
        self.E_min = 1000.0  # Minimum energy threshold (J)
    
    def calculate_flight_energy(self, velocity_vector: Tuple[float, float, float], epoch_duration: float) -> float:
        """Calculate flight energy: E_flight = P_hover * δ + ξ * ||v|| * δ"""
        velocity_magnitude = math.sqrt(sum(v**2 for v in velocity_vector))
        flight_energy = self.P_hover * epoch_duration + self.xi * velocity_magnitude * epoch_duration
        return flight_energy
    
    def calculate_communication_energy(self, data_transmitted: float) -> float:
        """Calculate communication energy: E_comm = η_comm * D_in"""
        comm_energy = self.eta_comm * data_transmitted
        return comm_energy
    
    def calculate_computation_energy(self, cpu_cycles: float) -> float:
        """Calculate computation energy: E_comp = κ * Σ C_j"""
        comp_energy = self.kappa * cpu_cycles
        return comp_energy
    
    def update_uav_energy(self, current_energy: float, velocity_vector: Tuple[float, float, float],
                         data_transmitted: float, cpu_cycles: float, epoch_duration: float) -> float:
        """Update UAV energy using paper formula: E_v(t+1) = E_v(t) - (E_flight + E_comm + E_comp)"""
        
        # Calculate individual energy components
        flight_energy = self.calculate_flight_energy(velocity_vector, epoch_duration)
        comm_energy = self.calculate_communication_energy(data_transmitted)
        comp_energy = self.calculate_computation_energy(cpu_cycles)
        
        # Total energy consumption
        total_consumption = flight_energy + comm_energy + comp_energy
        
        # Update energy level
        new_energy = current_energy - total_consumption
        
        return max(0.0, new_energy)  # Ensure non-negative energy
    
    def check_energy_constraint(self, energy_level: float) -> bool:
        """Check if energy constraint is satisfied: E_v(t) ≥ E_min"""
        return energy_level >= self.E_min
    
    def get_energy_breakdown(self, velocity_vector: Tuple[float, float, float],
                            data_transmitted: float, cpu_cycles: float, epoch_duration: float) -> Dict[str, float]:
        """Get detailed energy breakdown for analysis"""
        flight_energy = self.calculate_flight_energy(velocity_vector, epoch_duration)
        comm_energy = self.calculate_communication_energy(data_transmitted)
        comp_energy = self.calculate_computation_energy(cpu_cycles)
        total_energy = flight_energy + comm_energy + comp_energy
        
        return {
            'flight_energy': flight_energy,
            'communication_energy': comm_energy,
            'computation_energy': comp_energy,
            'total_energy': total_energy,
            'flight_percentage': (flight_energy / total_energy * 100) if total_energy > 0 else 0,
            'communication_percentage': (comm_energy / total_energy * 100) if total_energy > 0 else 0,
            'computation_percentage': (comp_energy / total_energy * 100) if total_energy > 0 else 0
        }

# Alias for backward compatibility
EnhancedShannonCapacityModel = ShannonCapacityModel
