"""
SAGIN - Space-Air-Ground Integrated Network
"""

from .core.network import SAGINNetwork
from .core.types import SystemParameters, Position, Task, Region
from .core.vehicles import VehicleManager
from .core.uavs import UAVManager
from .core.satellites import SatelliteConstellation
from .core.tasks import TaskManager

__version__ = "0.1.0"
__author__ = "SAGIN Development Team"

__all__ = [
    "SAGINNetwork",
    "SystemParameters", 
    "Position",
    "Task",
    "Region",
    "VehicleManager",
    "UAVManager", 
    "SatelliteConstellation",
    "TaskManager"
]
