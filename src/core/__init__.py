"""
Core SAGIN components
"""

from .network import SAGINNetwork
from .types import SystemParameters, Position, Task, Region
from .vehicles import VehicleManager
from .uavs import UAVManager, UAVStatus
from .satellites import SatelliteConstellation
from .tasks import TaskManager

# Import RL extension if available
try:
    from .network_rl_extension import *
    HAS_RL_EXTENSION = True
except ImportError:
    HAS_RL_EXTENSION = False

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
