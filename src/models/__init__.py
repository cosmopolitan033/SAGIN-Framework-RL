"""
Communication and computation models
"""

from .communication import CommunicationModel
from .latency import LatencyModel

__all__ = [
    "CommunicationModel",
    "LatencyModel"
]
