"""
Task generation and management for the SAGIN system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

from .types import (
    Task, TaskStatus, TaskDecision, Position, 
    SystemParameters, BurstEvent, Region
)


class TaskType(Enum):
    """Types of computational tasks."""
    COMPUTATION_INTENSIVE = "computation_intensive"
    DATA_INTENSIVE = "data_intensive"
    LATENCY_SENSITIVE = "latency_sensitive"
    NORMAL = "normal"


@dataclass
class TaskCharacteristics:
    """Characteristics for different types of tasks."""
    cpu_cycles_mean: float
    cpu_cycles_std: float
    data_size_in_mean: float
    data_size_in_std: float
    data_size_out_mean: float
    data_size_out_std: float
    deadline_mean: float
    deadline_std: float
    priority: float = 1.0


# Default task characteristics for different types
DEFAULT_TASK_CHARACTERISTICS = {
    TaskType.COMPUTATION_INTENSIVE: TaskCharacteristics(
        cpu_cycles_mean=5e9, cpu_cycles_std=2e9,
        data_size_in_mean=0.5, data_size_in_std=0.2,
        data_size_out_mean=0.1, data_size_out_std=0.05,
        deadline_mean=10.0, deadline_std=3.0,
        priority=1.2
    ),
    TaskType.DATA_INTENSIVE: TaskCharacteristics(
        cpu_cycles_mean=1e9, cpu_cycles_std=5e8,
        data_size_in_mean=10.0, data_size_in_std=5.0,
        data_size_out_mean=5.0, data_size_out_std=2.0,
        deadline_mean=15.0, deadline_std=5.0,
        priority=1.0
    ),
    TaskType.LATENCY_SENSITIVE: TaskCharacteristics(
        cpu_cycles_mean=1e8, cpu_cycles_std=5e7,
        data_size_in_mean=0.1, data_size_in_std=0.05,
        data_size_out_mean=0.05, data_size_out_std=0.02,
        deadline_mean=2.0, deadline_std=0.5,
        priority=1.5
    ),
    TaskType.NORMAL: TaskCharacteristics(
        cpu_cycles_mean=1e9, cpu_cycles_std=5e8,
        data_size_in_mean=1.0, data_size_in_std=0.5,
        data_size_out_mean=0.5, data_size_out_std=0.2,
        deadline_mean=8.0, deadline_std=2.0,
        priority=1.0
    )
}


class TaskGenerator:
    """Generates tasks based on spatio-temporal patterns."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.task_characteristics = DEFAULT_TASK_CHARACTERISTICS.copy()
        self.next_task_id = 1
        
        # Task generation history
        self.generation_history: List[Tuple[float, int, int]] = []  # (time, region_id, count)
        self.burst_events: List[BurstEvent] = []
        
        # Statistics
        self.stats = {
            'total_generated': 0,
            'by_region': defaultdict(int),
            'by_type': defaultdict(int),
            'by_vehicle': defaultdict(int)
        }
    
    def set_task_characteristics(self, task_type: TaskType, characteristics: TaskCharacteristics):
        """Set custom task characteristics for a task type."""
        self.task_characteristics[task_type] = characteristics
    
    def add_burst_event(self, region_id: int, start_time: float, duration: float, 
                       amplitude: float):
        """Add a burst event that increases task generation."""
        end_time = start_time + duration
        burst_event = BurstEvent(region_id, start_time, end_time, amplitude)
        self.burst_events.append(burst_event)
    
    def calculate_region_intensity(self, region_id: int, current_time: float,
                                 base_intensity: float) -> float:
        """Calculate current task intensity for a region."""
        intensity = base_intensity
        
        # Apply burst events
        for event in self.burst_events:
            if event.region_id == region_id and event.is_active(current_time):
                intensity *= event.amplitude
        
        # Add time-of-day variation (optional)
        hour_of_day = (current_time / 3600) % 24
        if 8 <= hour_of_day <= 18:  # Business hours
            intensity *= 1.5
        elif 22 <= hour_of_day or hour_of_day <= 6:  # Night hours
            intensity *= 0.3
        
        return intensity
    
    def generate_task(self, vehicle_id: int, region_id: int, current_time: float,
                     task_type: TaskType = TaskType.NORMAL) -> Task:
        """Generate a single task."""
        task_id = self.next_task_id
        self.next_task_id += 1
        
        characteristics = self.task_characteristics[task_type]
        
        # Generate task parameters with some randomness
        cpu_cycles = max(1e6, np.random.normal(
            characteristics.cpu_cycles_mean, 
            characteristics.cpu_cycles_std
        ))
        
        data_size_in = max(0.01, np.random.normal(
            characteristics.data_size_in_mean,
            characteristics.data_size_in_std
        ))
        
        data_size_out = max(0.01, np.random.normal(
            characteristics.data_size_out_mean,
            characteristics.data_size_out_std
        ))
        
        deadline_offset = max(0.5, np.random.normal(
            characteristics.deadline_mean,
            characteristics.deadline_std
        ))
        
        deadline = current_time + deadline_offset
        
        task = Task(
            id=task_id,
            source_vehicle_id=vehicle_id,
            region_id=region_id,
            data_size_in=data_size_in,
            data_size_out=data_size_out,
            cpu_cycles=cpu_cycles,
            deadline=deadline,
            creation_time=current_time,
            arrival_time=current_time
        )
        
        # Update statistics
        self.stats['total_generated'] += 1
        self.stats['by_region'][region_id] += 1
        self.stats['by_type'][task_type] += 1
        self.stats['by_vehicle'][vehicle_id] += 1
        
        return task
    
    def generate_tasks_for_region(self, region: Region, vehicles_in_region: List[int],
                                current_time: float, dt: float) -> List[Task]:
        """Generate tasks for all vehicles in a region."""
        tasks = []
        
        if not vehicles_in_region:
            return tasks
        
        # Calculate current intensity
        current_intensity = self.calculate_region_intensity(
            region.id, current_time, region.base_intensity
        )
        
        # Generate tasks for each vehicle
        for vehicle_id in vehicles_in_region:
            # Poisson process for task generation
            lambda_rate = current_intensity / len(vehicles_in_region)  # Distribute among vehicles
            
            # Number of tasks to generate in this time interval
            num_tasks = np.random.poisson(lambda_rate * dt)
            
            for _ in range(num_tasks):
                # Select task type based on probabilities
                task_type = self._select_task_type()
                task = self.generate_task(vehicle_id, region.id, current_time, task_type)
                tasks.append(task)
        
        # Record generation history
        if tasks:
            self.generation_history.append((current_time, region.id, len(tasks)))
        
        return tasks
    
    def _select_task_type(self) -> TaskType:
        """Select task type based on predefined probabilities."""
        # Task type probabilities
        probabilities = {
            TaskType.NORMAL: 0.6,
            TaskType.COMPUTATION_INTENSIVE: 0.2,
            TaskType.DATA_INTENSIVE: 0.15,
            TaskType.LATENCY_SENSITIVE: 0.05
        }
        
        rand = np.random.random()
        cumulative = 0.0
        
        for task_type, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return task_type
        
        return TaskType.NORMAL
    
    def get_generation_statistics(self) -> Dict:
        """Get task generation statistics."""
        return {
            'total_generated': self.stats['total_generated'],
            'by_region': dict(self.stats['by_region']),
            'by_type': dict(self.stats['by_type']),
            'by_vehicle': dict(self.stats['by_vehicle']),
            'active_burst_events': len([e for e in self.burst_events if e.is_active(0)]),
            'total_burst_events': len(self.burst_events)
        }


class TaskQueue:
    """Priority queue for task management."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.heap: List[Tuple[float, int, Task]] = []  # (priority, counter, task)
        self.max_size = max_size
        self.dropped_tasks = 0
        self._counter = 0  # Unique counter to break ties in heap
    
    def add_task(self, task: Task, priority: Optional[float] = None) -> bool:
        """Add a task to the queue."""
        if self.max_size and len(self.heap) >= self.max_size:
            # Drop task if queue is full
            self.dropped_tasks += 1
            return False
        
        if priority is None:
            # Default priority based on deadline urgency
            priority = -task.deadline  # Earlier deadlines have higher priority
        
        # Use counter to ensure heap entries are always comparable
        heapq.heappush(self.heap, (priority, self._counter, task))
        self._counter += 1
        return True
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task with highest priority."""
        if not self.heap:
            return None
        
        _, _, task = heapq.heappop(self.heap)
        return task
    
    def peek_next_task(self) -> Optional[Task]:
        """Peek at the next task without removing it."""
        if not self.heap:
            return None
        
        _, _, task = self.heap[0]
        return task
    
    def remove_expired_tasks(self, current_time: float) -> List[Task]:
        """Remove tasks that have exceeded their deadlines."""
        expired = []
        remaining = []
        
        while self.heap:
            priority, counter, task = heapq.heappop(self.heap)
            if task.deadline <= current_time:
                task.status = TaskStatus.DEADLINE_MISSED
                expired.append(task)
            else:
                remaining.append((priority, counter, task))
        
        # Rebuild heap with remaining tasks
        self.heap = remaining
        heapq.heapify(self.heap)
        
        return expired
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self.heap)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.heap) == 0
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the queue (for inspection)."""
        return [task for _, _, task in self.heap]


class TaskManager:
    """Central task management system."""
    
    def __init__(self, system_params: SystemParameters):
        self.system_params = system_params
        self.task_generator = TaskGenerator(system_params)
        
        # Task tracking
        self.all_tasks: Dict[int, Task] = {}
        self.active_tasks: Dict[int, Task] = {}
        self.completed_tasks: Dict[int, Task] = {}
        self.failed_tasks: Dict[int, Task] = {}
        
        # Regional task queues (for static UAVs)
        self.region_queues: Dict[int, TaskQueue] = {}
        
        # Performance metrics
        self.metrics = {
            'total_generated': 0,
            'total_completed': 0,
            'total_failed': 0,
            'deadline_violations': 0,
            'average_completion_time': 0.0,
            'average_latency': 0.0,
            'success_rate': 0.0
        }
    
    def initialize_region_queues(self, region_ids: List[int], queue_size: int = 1000):
        """Initialize task queues for regions."""
        for region_id in region_ids:
            self.region_queues[region_id] = TaskQueue(max_size=queue_size)
    
    def generate_tasks(self, regions: Dict[int, Region], 
                      vehicles_by_region: Dict[int, List[int]], 
                      current_time: float, dt: float) -> List[Task]:
        """Generate tasks for all regions."""
        all_new_tasks = []
        
        for region_id, region in regions.items():
            vehicles = vehicles_by_region.get(region_id, [])
            new_tasks = self.task_generator.generate_tasks_for_region(
                region, vehicles, current_time, dt
            )
            
            for task in new_tasks:
                # Add to tracking
                self.all_tasks[task.id] = task
                self.active_tasks[task.id] = task
                
                # Add to regional queue if it exists
                if region_id in self.region_queues:
                    self.region_queues[region_id].add_task(task)
            
            all_new_tasks.extend(new_tasks)
        
        self.metrics['total_generated'] += len(all_new_tasks)
        return all_new_tasks
    
    def get_tasks_for_region(self, region_id: int, max_tasks: int = 10) -> List[Task]:
        """Get pending tasks for a region."""
        if region_id not in self.region_queues:
            return []
        
        queue = self.region_queues[region_id]
        tasks = []
        
        for _ in range(min(max_tasks, queue.size())):
            task = queue.get_next_task()
            if task:
                tasks.append(task)
            else:
                break
        
        return tasks
    
    def mark_task_completed(self, task: Task):
        """Mark a task as completed."""
        task.status = TaskStatus.COMPLETED
        
        # Move from active to completed
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        self.completed_tasks[task.id] = task
        
        # Update metrics
        self.metrics['total_completed'] += 1
        
        if task.completion_time > 0:
            completion_time = task.completion_time - task.creation_time
            current_avg = self.metrics['average_completion_time']
            count = self.metrics['total_completed']
            self.metrics['average_completion_time'] = (
                (current_avg * (count - 1) + completion_time) / count
            )
    
    def mark_task_failed(self, task: Task, reason: str = "unknown"):
        """Mark a task as failed."""
        if task.deadline < task.completion_time:
            task.status = TaskStatus.DEADLINE_MISSED
            self.metrics['deadline_violations'] += 1
        else:
            task.status = TaskStatus.FAILED
        
        # Move from active to failed
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        self.failed_tasks[task.id] = task
        self.metrics['total_failed'] += 1
    
    def cleanup_expired_tasks(self, current_time: float):
        """Remove expired tasks from all queues."""
        total_expired = 0
        
        for region_id, queue in self.region_queues.items():
            expired_tasks = queue.remove_expired_tasks(current_time)
            for task in expired_tasks:
                self.mark_task_failed(task, "deadline_expired")
            total_expired += len(expired_tasks)
        
        return total_expired
    
    def get_regional_task_stats(self) -> Dict[int, Dict]:
        """Get task statistics by region."""
        stats = {}
        
        for region_id, queue in self.region_queues.items():
            stats[region_id] = {
                'queued_tasks': queue.size(),
                'dropped_tasks': queue.dropped_tasks,
                'queue_utilization': queue.size() / queue.max_size if queue.max_size else 0
            }
        
        return stats
    
    def get_system_metrics(self) -> Dict:
        """Get overall system metrics."""
        # Calculate success rate
        total_processed = self.metrics['total_completed'] + self.metrics['total_failed']
        success_rate = self.metrics['total_completed'] / total_processed if total_processed > 0 else 0
        
        self.metrics['success_rate'] = success_rate
        
        return self.metrics.copy()
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return self.all_tasks.get(task_id)
    
    def get_active_tasks_count(self) -> int:
        """Get number of active tasks."""
        return len(self.active_tasks)
    
    def get_load_by_region(self) -> Dict[int, float]:
        """Get current load (task count) by region."""
        return {
            region_id: queue.size() 
            for region_id, queue in self.region_queues.items()
        }
    
    def add_burst_event(self, region_id: int, start_time: float, 
                       duration: float, amplitude: float):
        """Add a burst event for increased task generation."""
        self.task_generator.add_burst_event(region_id, start_time, duration, amplitude)
    
    def get_recent_tasks(self, max_tasks: int = 10) -> List[Task]:
        """Get recent completed tasks for analysis."""
        # Get most recent completed tasks
        recent_completed = list(self.completed_tasks.values())
        
        # Sort by completion time (most recent first)
        recent_completed.sort(key=lambda t: t.completion_time if t.completion_time > 0 else 0, reverse=True)
        
        # Return up to max_tasks recent tasks
        return recent_completed[:max_tasks]
