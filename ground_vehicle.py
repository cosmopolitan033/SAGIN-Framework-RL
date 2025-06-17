# ground_vehicle.py
import numpy as np
from communication_model import compute_rate_general

class GroundVehicle:
    def __init__(self, vehicle_id, initial_position, mobility_model, cache_size, compute_power, energy, max_queue, duration):
        self.vehicle_id = vehicle_id
        self.position = initial_position  # (x, y, 0) - z=0 for ground level
        self.mobility_model = mobility_model  # 'random_waypoint', 'route_following', etc.
        self.route = []  # List of waypoints if following a route
        self.speed = np.random.uniform(5, 30)  # m/s (18-108 km/h)
        self.direction = np.random.uniform(0, 2*np.pi)  # Random initial direction in radians
        
        # Resource constraints - energy not considered for vehicles
        self.energy = float('inf')  # Infinite energy - vehicles don't consider energy
        self.energy_used_this_slot = 0.0
        self.compute_power = 0  # No local computation ability
        
        # Storage
        self.cache_capacity_mb = cache_size
        self.cache_used_mb = 0.0
        self.cache_storage = {}  # Content cache: cid → content metadata
        self.aggregated_content = {}  # Temporary content: cid → content metadata
        
        # Task management
        self.max_queue = max_queue
        self.task_queue = []
        self.duration = duration
        self.total_tasks = 0
        self.cache_hits = 0
        self.tasks_completed_within_bound = 0
        self.neighbor_vehicles = []  # List of neighbor vehicle IDs
        
        # V2V communication
        self.max_v2v_range = 300  # meters
        self.v2v_links = {}  # {vehicle_id: link_quality}
        
        # V2U (Vehicle-to-UAV) communication
        self.connected_uav = None  # UAV currently serving this vehicle
        
        # Content popularity tracking
        self.content_popularity = {}  # {content_id: popularity_score}
        self.current_time = 0

    def move(self, timestep):
        """Update vehicle position based on mobility model"""
        if self.mobility_model == 'random_waypoint':
            # Simple random movement
            self.direction += np.random.uniform(-0.5, 0.5)  # Random direction change
            dx = self.speed * np.cos(self.direction) * self.duration
            dy = self.speed * np.sin(self.direction) * self.duration
            
            x, y, _ = self.position
            new_x = x + dx
            new_y = y + dy
            self.position = (new_x, new_y, 0)
            
        elif self.mobility_model == 'route_following' and self.route:
            # Follow predetermined route
            if not self.route:
                return
                
            target = self.route[0]
            x, y, _ = self.position
            tx, ty, _ = target
            
            # Direction to target
            angle = np.arctan2(ty - y, tx - x)
            
            # Distance to move this timestep
            distance = self.speed * self.duration
            
            # Check if we reach or pass the waypoint
            dist_to_waypoint = np.sqrt((tx - x)**2 + (ty - y)**2)
            
            if distance >= dist_to_waypoint:
                # Reached waypoint, remove it and continue to next
                self.route.pop(0)
                remaining_dist = distance - dist_to_waypoint
                
                # If we have another waypoint, continue moving
                if self.route:
                    next_target = self.route[0]
                    ntx, nty, _ = next_target
                    next_angle = np.arctan2(nty - y, ntx - x)
                    
                    # Move remaining distance toward next waypoint
                    self.position = (
                        tx + np.cos(next_angle) * remaining_dist,
                        ty + np.sin(next_angle) * remaining_dist,
                        0
                    )
                else:
                    # End of route
                    self.position = (tx, ty, 0)
            else:
                # Move toward waypoint but don't reach it yet
                self.position = (
                    x + np.cos(angle) * distance,
                    y + np.sin(angle) * distance,
                    0
                )

    def update_v2v_links(self, all_vehicles):
        """Update vehicle-to-vehicle communication links based on distance"""
        self.v2v_links = {}
        self.neighbor_vehicles = []
        
        for v_id, vehicle in all_vehicles.items():
            if v_id == self.vehicle_id:
                continue
                
            # Calculate distance
            d = np.linalg.norm(np.array(self.position) - np.array(vehicle.position))
            
            # If within range, establish link
            if d <= self.max_v2v_range:
                # Link quality based on distance (1.0 is best, 0.0 is worst)
                quality = max(0.0, 1.0 - (d / self.max_v2v_range))
                self.v2v_links[v_id] = quality
                self.neighbor_vehicles.append(v_id)

    def receive_task(self, task, from_vehicle_id=None):
        """Receive a task from another entity (vehicle, UAV, etc.)"""
        if len(self.task_queue) >= self.max_queue:
            print(f"[GV {self.vehicle_id}] Task queue full, dropping task {task['task_id']}")
            return False
            
        # Add to queue
        self.task_queue.append(task)
        
        # Sort by priority and delay bound
        self.task_queue.sort(key=lambda t: (
            -t.get('priority', 0),  # Higher priority first
            t.get('delay_bound', float('inf'))  # Earlier deadline first
        ))
        
        print(f"[GV {self.vehicle_id}] Received task {task['task_id']}")
        return True

    def execute_tasks(self, timestep):
        """Vehicles cannot execute tasks locally - just forward them"""
        # All tasks remain in the queue for potential forwarding
        # Vehicle doesn't have computing capability
        return []

    def generate_tasks(self, timestep):
        """Generate computational tasks based on content available"""
        tasks = []
        # Random task generation based on available content
        for cid, content in list(self.aggregated_content.items()) + list(self.cache_storage.items()):
            # 30% chance of generating a task for each piece of content
            if np.random.random() < 0.3:
                task_type = np.random.choice(['RTP', 'CD', 'MRP', 'INF'])  # Different task types
                
                # Task requirements based on type
                if task_type == 'RTP':  # Real-Time Perception
                    cpu_req = np.random.uniform(10, 30)
                    delay_bound = 0.05  # 50ms
                    priority = 3  # Highest priority
                elif task_type == 'CD':  # Cooperative Driving
                    cpu_req = np.random.uniform(5, 15)
                    delay_bound = 0.1  # 100ms
                    priority = 2
                elif task_type == 'MRP':  # Map/Route Planning
                    cpu_req = np.random.uniform(20, 50)
                    delay_bound = 0.5  # 500ms
                    priority = 1
                else:  # INF - Infotainment
                    cpu_req = np.random.uniform(5, 100)
                    delay_bound = 2.0  # 2s
                    priority = 0  # Lowest priority
                
                task = {
                    'task_id': f"t_{self.vehicle_id}_{timestep}_{len(tasks)}",
                    'content_id': cid,
                    'generation_time': timestep * self.duration,
                    'source_vehicle': self.vehicle_id,
                    'required_cpu': cpu_req,
                    'size': np.random.uniform(0.5, 5.0),  # MB
                    'delay_bound': delay_bound,  # seconds
                    'task_type': task_type,
                    'priority': priority,
                    'remaining_cpu': cpu_req
                }
                tasks.append(task)
        
        return tasks

    def update_cache(self, timestep, global_satellite_content_pool=None, is_connected_to_satellite=False):
        """Update vehicle's cache based on popularity and TTL"""
        # Evict expired content
        self.evict_expired_content(timestep)
        
        # Combine all available content for caching decisions
        candidate_pool = {}
        
        # 1. Add from local cache_storage (with all meta info)
        for cid, meta in self.cache_storage.items():
            candidate_pool[cid] = meta
            
        # 2. Add from aggregated_content
        for cid, content in self.aggregated_content.items():
            candidate_pool[cid] = content
            
        # 3. Sort by popularity (current implementation: random)
        # In a real system, this would use more sophisticated popularity metrics
        sorted_candidates = sorted(
            candidate_pool.items(),
            key=lambda x: np.random.random(),  # Random for now
            reverse=True
        )
        
        # 4. Fill cache based on available space
        self.cache_storage = {}
        self.cache_used_mb = 0
        
        for cid, content in sorted_candidates:
            content_size = content.get('size', 1.0)  # Size in MB
            if self.cache_used_mb + content_size <= self.cache_capacity_mb:
                self.cache_storage[cid] = content
                self.cache_used_mb += content_size
            else:
                break

    def evict_expired_content(self, timestep):
        """Remove expired content based on TTL"""
        current_time = timestep * self.duration
        
        # Find expired content
        expired_cache = []
        for cid, meta in self.cache_storage.items():
            # Expired if past TTL
            if current_time - meta.get('generation_time', 0) > meta.get('ttl', float('inf')):
                expired_cache.append(cid)
                
        # Delete expired content
        for cid in expired_cache:
            content_size = self.cache_storage[cid].get('size', 1.0)
            self.cache_used_mb -= content_size
            del self.cache_storage[cid]
            
        # Also check aggregated content
        expired_agg = []
        for cid, meta in self.aggregated_content.items():
            if current_time - meta.get('generation_time', 0) > meta.get('ttl', float('inf')):
                expired_agg.append(cid)
                
        for cid in expired_agg:
            del self.aggregated_content[cid]

    def clear_aggregated_content(self):
        """Clear temporary aggregated content after processing"""
        self.aggregated_content = {}

    def can_offload_to_vehicle(self, task, vehicle):
        """Check if a task can be offloaded to another vehicle for hopping purposes"""
        # Check if target vehicle is in range
        if vehicle.vehicle_id not in self.v2v_links:
            return False
            
        # For hopping, we don't need the content - we're just relaying to a UAV
        # But we do need queue space
        if len(vehicle.task_queue) >= vehicle.max_queue:
            return False
            
        return True

    def get_position_as_grid(self, grid_size_x, grid_size_y):
        """Convert continuous position to grid coordinates"""
        x, y, _ = self.position
        grid_x = int(x / 100)  # Assuming 100m grid cells
        grid_y = int(y / 100)
        
        # Ensure within bounds
        grid_x = max(0, min(grid_x, grid_size_x - 1))
        grid_y = max(0, min(grid_y, grid_size_y - 1))
        
        return (grid_x, grid_y)

    def observe(self, neighbor_loads, satellite_in_range, activation_mask):
        """Create observation vector for RL agent"""
        # Example observation structure (to be adjusted based on your RL model)
        obs = np.zeros(28)  # Placeholder - update size as needed
        
        # 1. Vehicle state
        obs[0] = self.energy / 1000.0  # Normalized energy
        obs[1] = len(self.task_queue) / self.max_queue  # Queue utilization
        obs[2] = self.cache_used_mb / self.cache_capacity_mb  # Cache utilization
        
        # 2. Content information (simplified)
        # This would need to be expanded based on your specific needs
        
        # 3. Neighbor information
        # This would include V2V link quality and neighbor load info
        
        return obs