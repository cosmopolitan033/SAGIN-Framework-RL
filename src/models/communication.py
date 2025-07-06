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
    """Communication model for calculating data rates and delays."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.channel_params = DEFAULT_CHANNEL_PARAMS.copy()
        
        # Thermal noise at room temperature
        self.thermal_noise_density = -174  # dBm/Hz at 290K
    
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
        """Calculate achievable data rate using Shannon capacity."""
        # Get link type and parameters
        link_type = self._determine_link_type(tx_type, rx_type)
        params = self.channel_params[link_type]
        
        # Calculate channel gain
        channel_gain_db = self.calculate_channel_gain(tx_pos, rx_pos, tx_type, rx_type)
        
        # Convert to linear scale
        channel_gain_linear = 10 ** (channel_gain_db / 10)
        
        # Calculate noise power
        noise_power_dbm = (self.thermal_noise_density + 
                          10 * math.log10(params.bandwidth * 1e6) + 
                          params.noise_figure)
        noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)  # Convert to Watts
        
        # Calculate received power
        tx_power_w = params.transmit_power
        rx_power_w = tx_power_w * channel_gain_linear
        
        # Calculate SNR
        snr = rx_power_w / noise_power_w
        
        # Shannon capacity
        data_rate = params.bandwidth * math.log2(1 + snr)  # Mbps
        
        return max(0, data_rate)
    
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
        data_rate = self.calculate_data_rate(tx_pos, rx_pos, tx_type, rx_type)
        
        # Calculate SNR
        channel_gain_linear = 10 ** (channel_gain / 10)
        noise_power_dbm = (self.thermal_noise_density + 
                          10 * math.log10(params.bandwidth * 1e6) + 
                          params.noise_figure)
        noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)
        rx_power_w = params.transmit_power * channel_gain_linear
        snr_db = 10 * math.log10(rx_power_w / noise_power_w)
        
        return {
            'distance': distance,
            'channel_gain_db': channel_gain,
            'data_rate_mbps': data_rate,
            'snr_db': snr_db,
            'link_type': link_type.value,
            'frequency_ghz': params.frequency,
            'bandwidth_mhz': params.bandwidth,
            'is_available': data_rate >= self.system_params.min_rate_threshold
        }
    
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
