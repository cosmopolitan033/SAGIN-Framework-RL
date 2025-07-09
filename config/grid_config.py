"""
Comprehensive configuration for SAGIN network simulation.
"""

from typing import Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for network grid topology."""
    
    # Grid dimensions
    grid_rows: int = 5
    grid_cols: int = 10
    
    # Area bounds (min_x, max_x, min_y, max_y)
    area_bounds: Tuple[float, float, float, float] = (0.0, 10000.0, 0.0, 5000.0)
    
    # Region properties
    region_base_intensity: float = 1.0
    region_intensity_variance: float = 0.5  # Random variance in intensity
    
    # UAV properties per region
    static_uav_cpu_capacity: float = 1e9  # CPU cycles per second
    
    @property
    def total_regions(self) -> int:
        """Total number of regions in the grid."""
        return self.grid_rows * self.grid_cols
    
    @property
    def region_width(self) -> float:
        """Width of each region."""
        return (self.area_bounds[1] - self.area_bounds[0]) / self.grid_cols
    
    @property
    def region_height(self) -> float:
        """Height of each region."""
        return (self.area_bounds[3] - self.area_bounds[2]) / self.grid_rows
    
    def get_region_center(self, row: int, col: int) -> Tuple[float, float]:
        """Get the center coordinates of a region at (row, col)."""
        min_x, max_x, min_y, max_y = self.area_bounds
        
        center_x = min_x + (col + 0.5) * self.region_width
        center_y = min_y + (row + 0.5) * self.region_height
        
        return center_x, center_y
    
    def get_region_id(self, row: int, col: int) -> int:
        """Get the region ID for a grid position."""
        return row * self.grid_cols + col + 1
    
    def get_grid_position(self, region_id: int) -> Tuple[int, int]:
        """Get the grid position (row, col) for a region ID."""
        region_index = region_id - 1
        row = region_index // self.grid_cols
        col = region_index % self.grid_cols
        return row, col


@dataclass
class VehicleConfig:
    """Configuration for vehicles in the network."""
    
    # Vehicle counts
    random_vehicles: int = 50
    bus_vehicles: int = 15
    
    # Vehicle properties
    vehicle_speed_range: Tuple[float, float] = (5.0, 20.0)  # m/s
    bus_speed: float = 10.0  # m/s
    
    # Vehicle distribution
    vehicle_distribution: str = "uniform"  # uniform, clustered, random


@dataclass
class UAVConfig:
    """Configuration for UAVs in the network."""
    
    # UAV counts
    dynamic_uavs: int = 10
    
    # UAV properties
    uav_max_speed: float = 20.0  # m/s
    uav_altitude: float = 100.0  # m
    uav_cpu_capacity: float = 1e9  # cycles per second
    
    # Energy system
    battery_capacity: float = 10000.0  # J
    min_energy_threshold: float = 1000.0  # J
    
    # Communication
    transmit_power: float = 10.0  # W
    antenna_gain: float = 10.0  # dB
    communication_range: float = 1000.0  # m


@dataclass
class SatelliteConfig:
    """Configuration for satellites in the network."""
    
    # Satellite constellation
    num_satellites: int = 12
    num_planes: int = 3
    
    # Satellite properties
    altitude: float = 600000.0  # m (600km)
    orbital_period: float = 5760.0  # seconds (96 minutes)
    
    # Communication
    transmit_power: float = 100.0  # W
    antenna_gain: float = 30.0  # dB


@dataclass
class TaskConfig:
    """Configuration for task generation."""
    
    # Task generation parameters
    base_task_rate: float = 1.0  # tasks per second per region
    task_rate_variance: float = 0.5
    
    # Task characteristics
    cpu_cycles_mean: float = 1e9
    cpu_cycles_std: float = 5e8
    data_size_in_mean: float = 1.0  # MB
    data_size_in_std: float = 0.5
    data_size_out_mean: float = 0.5  # MB
    data_size_out_std: float = 0.2
    deadline_mean: float = 8.0  # seconds
    deadline_std: float = 2.0
    
    # Burst events
    burst_events: List[Tuple[int, float, float, float]] = None  # (region_id, start_time, duration, amplitude)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    
    # Simulation timing
    epoch_duration: float = 1.0  # seconds
    total_epochs: int = 100
    
    # Performance thresholds
    min_rate_threshold: float = 1.0  # Mbps
    max_load_imbalance: float = 0.3
    
    # Logging
    logging_level: str = "medium"  # low, medium, high
    log_decisions: bool = True
    log_resource_usage: bool = True
    detailed_interval: int = 5
    progress_interval: int = 25
    
    # Output
    export_results: bool = True
    results_filename: str = "sagin_results.json"


@dataclass
class SAGINConfig:
    """Complete SAGIN network configuration."""
    
    name: str = "default"
    description: str = "Default SAGIN configuration"
    
    # Component configurations
    grid: GridConfig = None
    vehicles: VehicleConfig = None
    uavs: UAVConfig = None
    satellites: SatelliteConfig = None
    tasks: TaskConfig = None
    simulation: SimulationConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.grid is None:
            self.grid = GridConfig()
        if self.vehicles is None:
            self.vehicles = VehicleConfig()
        if self.uavs is None:
            self.uavs = UAVConfig()
        if self.satellites is None:
            self.satellites = SatelliteConfig()
        if self.tasks is None:
            self.tasks = TaskConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()
    
    def get_system_parameters(self):
        """Convert to SystemParameters object."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from src.core.types import SystemParameters
        return SystemParameters(
            epoch_duration=self.simulation.epoch_duration,
            total_epochs=self.simulation.total_epochs,
            min_rate_threshold=self.simulation.min_rate_threshold,
            uav_max_speed=self.uavs.uav_max_speed,
            uav_altitude=self.uavs.uav_altitude,
            max_load_imbalance=self.simulation.max_load_imbalance
        )


# Predefined comprehensive configurations
SAGIN_CONFIGS = {
    "small_test": SAGINConfig(
        name="small_test",
        description="Small test configuration for quick validation",
        grid=GridConfig(
            grid_rows=2,
            grid_cols=3,
            area_bounds=(0.0, 3000.0, 0.0, 2000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=10,
            bus_vehicles=3
        ),
        uavs=UAVConfig(
            dynamic_uavs=2
        ),
        satellites=SatelliteConfig(
            num_satellites=4,
            num_planes=2
        ),
        tasks=TaskConfig(
            base_task_rate=0.5,
            burst_events=[(1, 10.0, 5.0, 2.0)]
        ),
        simulation=SimulationConfig(
            total_epochs=50,
            logging_level="high"
        )
    ),
    
    "medium_demo": SAGINConfig(
        name="medium_demo",
        description="Medium configuration for standard demonstrations",
        grid=GridConfig(
            grid_rows=4,
            grid_cols=4,
            area_bounds=(0.0, 8000.0, 0.0, 8000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=30,
            bus_vehicles=10
        ),
        uavs=UAVConfig(
            dynamic_uavs=5
        ),
        satellites=SatelliteConfig(
            num_satellites=8,
            num_planes=3
        ),
        tasks=TaskConfig(
            base_task_rate=1.0,
            burst_events=[(2, 30.0, 20.0, 2.0), (3, 60.0, 15.0, 1.8)]
        ),
        simulation=SimulationConfig(
            total_epochs=100,
            logging_level="medium"
        )
    ),
    
    "large_simulation": SAGINConfig(
        name="large_simulation",
        description="Large configuration for comprehensive simulations",
        grid=GridConfig(
            grid_rows=5,
            grid_cols=10,
            area_bounds=(0.0, 20000.0, 0.0, 10000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=80,
            bus_vehicles=20
        ),
        uavs=UAVConfig(
            dynamic_uavs=15
        ),
        satellites=SatelliteConfig(
            num_satellites=12,
            num_planes=3
        ),
        tasks=TaskConfig(
            base_task_rate=1.5,
            burst_events=[(5, 100.0, 50.0, 3.0), (10, 200.0, 30.0, 2.5), (15, 350.0, 40.0, 2.0)]
        ),
        simulation=SimulationConfig(
            total_epochs=500,
            logging_level="low"
        )
    ),
    
    "highway_scenario": SAGINConfig(
        name="highway_scenario",
        description="Highway scenario with linear topology",
        grid=GridConfig(
            grid_rows=1,
            grid_cols=20,
            area_bounds=(0.0, 40000.0, 0.0, 2000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=60,
            bus_vehicles=15,
            vehicle_speed_range=(15.0, 30.0)
        ),
        uavs=UAVConfig(
            dynamic_uavs=8
        ),
        satellites=SatelliteConfig(
            num_satellites=6,
            num_planes=2
        ),
        tasks=TaskConfig(
            base_task_rate=2.0,
            burst_events=[(5, 50.0, 20.0, 3.0), (15, 150.0, 25.0, 2.5)]
        ),
        simulation=SimulationConfig(
            total_epochs=300,
            logging_level="medium"
        )
    ),
    
    "city_scenario": SAGINConfig(
        name="city_scenario",
        description="Dense city scenario with high task load",
        grid=GridConfig(
            grid_rows=8,
            grid_cols=12,
            area_bounds=(0.0, 24000.0, 0.0, 16000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=150,
            bus_vehicles=30,
            vehicle_speed_range=(3.0, 15.0)
        ),
        uavs=UAVConfig(
            dynamic_uavs=20
        ),
        satellites=SatelliteConfig(
            num_satellites=16,
            num_planes=4
        ),
        tasks=TaskConfig(
            base_task_rate=2.5,
            cpu_cycles_mean=2e9,
            data_size_in_mean=2.0,
            burst_events=[(10, 100.0, 60.0, 4.0), (20, 200.0, 40.0, 3.0), (30, 300.0, 50.0, 3.5)]
        ),
        simulation=SimulationConfig(
            total_epochs=400,
            logging_level="low"
        )
    ),
    
    "sparse_rural": SAGINConfig(
        name="sparse_rural",
        description="Sparse rural scenario with low vehicle density",
        grid=GridConfig(
            grid_rows=6,
            grid_cols=8,
            area_bounds=(0.0, 32000.0, 0.0, 24000.0)
        ),
        vehicles=VehicleConfig(
            random_vehicles=25,
            bus_vehicles=5,
            vehicle_speed_range=(10.0, 25.0)
        ),
        uavs=UAVConfig(
            dynamic_uavs=6
        ),
        satellites=SatelliteConfig(
            num_satellites=10,
            num_planes=3
        ),
        tasks=TaskConfig(
            base_task_rate=0.3,
            burst_events=[(5, 120.0, 30.0, 2.0)]
        ),
        simulation=SimulationConfig(
            total_epochs=200,
            logging_level="medium"
        )
    )
}


def get_sagin_config(config_name: str = "medium_demo") -> SAGINConfig:
    """Get a predefined SAGIN configuration."""
    if config_name in SAGIN_CONFIGS:
        return SAGIN_CONFIGS[config_name]
    else:
        print(f"Warning: Unknown config '{config_name}', using default 'medium_demo'")
        return SAGIN_CONFIGS["medium_demo"]


def get_grid_config(config_name: str = "medium_demo") -> GridConfig:
    """Get a grid configuration from a SAGIN config."""
    sagin_config = get_sagin_config(config_name)
    return sagin_config.grid


def list_available_configs() -> List[str]:
    """List all available configuration names."""
    return list(SAGIN_CONFIGS.keys())


def print_config_summary(config_name: str = None):
    """Print a summary of available configurations or a specific config."""
    if config_name:
        if config_name in SAGIN_CONFIGS:
            config = SAGIN_CONFIGS[config_name]
            print(f"\n{config.name.upper()} Configuration:")
            print(f"Description: {config.description}")
            print(f"Grid: {config.grid.grid_rows}x{config.grid.grid_cols} ({config.grid.total_regions} regions)")
            print(f"Area: {config.grid.area_bounds[1]/1000:.1f}km x {config.grid.area_bounds[3]/1000:.1f}km")
            print(f"Vehicles: {config.vehicles.random_vehicles} random + {config.vehicles.bus_vehicles} bus")
            print(f"UAVs: {config.grid.total_regions} static + {config.uavs.dynamic_uavs} dynamic")
            print(f"Satellites: {config.satellites.num_satellites} satellites in {config.satellites.num_planes} planes")
            print(f"Simulation: {config.simulation.total_epochs} epochs, {config.simulation.logging_level} logging")
            if config.tasks.burst_events:
                print(f"Burst events: {len(config.tasks.burst_events)} configured")
        else:
            print(f"Configuration '{config_name}' not found")
    else:
        print("\nAvailable SAGIN Configurations:")
        print("=" * 50)
        for name, config in SAGIN_CONFIGS.items():
            print(f"{name:20} - {config.description}")
            print(f"{'':20}   Grid: {config.grid.grid_rows}x{config.grid.grid_cols}, "
                  f"Vehicles: {config.vehicles.random_vehicles + config.vehicles.bus_vehicles}, "
                  f"UAVs: {config.uavs.dynamic_uavs}, "
                  f"Epochs: {config.simulation.total_epochs}")


def create_custom_sagin_config(name: str, **kwargs) -> SAGINConfig:
    """Create a custom SAGIN configuration."""
    config = SAGINConfig(name=name, description=f"Custom configuration: {name}")
    
    # Override default values with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested config updates
            for component in ['grid', 'vehicles', 'uavs', 'satellites', 'tasks', 'simulation']:
                component_obj = getattr(config, component)
                if hasattr(component_obj, key):
                    setattr(component_obj, key, value)
                    break
    
    return config
