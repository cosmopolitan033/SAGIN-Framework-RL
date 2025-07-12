"""
Visualization module for SAGIN network simulation.
"""

try:
    from .network_visualizer import NetworkVisualizer
    from .real_time_visualizer import RealTimeNetworkVisualizer
    
    __all__ = ['NetworkVisualizer', 'RealTimeNetworkVisualizer']
except ImportError as e:
    print(f"Warning: Could not import visualization modules. Make sure matplotlib is installed: {e}")
    
    # Create dummy classes to prevent import errors
    class NetworkVisualizer:
        def __init__(self, network):
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib")
    
    class RealTimeNetworkVisualizer:
        def __init__(self, network):
            raise ImportError("Matplotlib not available. Install with: pip install matplotlib")
    
    __all__ = ['NetworkVisualizer', 'RealTimeNetworkVisualizer']
