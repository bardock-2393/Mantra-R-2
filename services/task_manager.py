"""
Task Management Service for Round 2
Handles video analysis tasks with GPU processing queue and progress tracking
"""

import os
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
from config import Config
from services.ai_service_fixed import minicpm_service
from services.video_processing_service import video_processor
from services.performance_service import PerformanceMonitor

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class AnalysisTask:
    """Video analysis task data structure"""
    task_id: str
    video_path: str
    analysis_type: str
    user_focus: str
    priority: TaskPriority
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    performance_metrics: Optional[Dict] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class TaskManager:
    """Manages video analysis tasks with GPU processing queue"""
    
    def __init__(self):
        self.tasks: Dict[str, AnalysisTask] = {}
        self.gpu_queue: asyncio.Queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self.performance_monitor = PerformanceMonitor()
        self.max_concurrent_tasks = Config.PERFORMANCE_TARGETS['concurrent_sessions']
        self.is_running = False
        
    async def start(self):
        """Start the task manager and worker threads"""
        if self.is_running:
            return
        
        print("🚀 Starting Task Manager...")
        
        # Start worker threads
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        self.is_running = True
        print(f"✅ Task Manager started with {self.max_concurrent_tasks} workers")
    
    async def stop(self):
        """Stop the task manager and cleanup"""
        if not self.is_running:
            return
        
        print("🛑 Stopping Task Manager...")
        
        # Cancel all worker tasks
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Clear tasks
        self.tasks.clear()
        
        self.is_running = False
        print("✅ Task Manager stopped")
    
    async def queue_analysis(self, video_path: str, analysis_type: str, user_focus: str, 
                           priority: TaskPriority = TaskPriority.NORMAL, 
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> str:
        """Queue a new video analysis task"""
        try:
            # Validate video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Create task
            task_id = str(uuid.uuid4())
            task = AnalysisTask(
                task_id=task_id,
                video_path=video_path,
                analysis_type=analysis_type,
                user_focus=user_focus,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                user_id=user_id,
                session_id=session_id
            )
            
            # Add to task registry
            self.tasks[task_id] = task
            
            # Add to GPU processing queue
            await self.gpu_queue.put((priority.value, task_id))
            
            print(f"✅ Task {task_id} queued for analysis (priority: {priority.value})")
            return task_id
            
        except Exception as e:
            print(f"❌ Failed to queue analysis task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get current status of a task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return asdict(task)
    
    async def get_user_tasks(self, user_id: str) -> List[Dict]:
        """Get all tasks for a specific user"""
        user_tasks = []
        for task in self.tasks.values():
            if task.user_id == user_id:
                user_tasks.append(asdict(task))
        
        # Sort by creation time (newest first)
        user_tasks.sort(key=lambda x: x['created_at'], reverse=True)
        return user_tasks
    
    async def get_session_tasks(self, session_id: str) -> List[Dict]:
        """Get all tasks for a specific session"""
        session_tasks = []
        for task in self.tasks.values():
            if task.session_id == session_id:
                session_tasks.append(asdict(task))
        
        # Sort by creation time (newest first)
        session_tasks.sort(key=lambda x: x['created_at'], reverse=True)
        return session_tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            print(f"✅ Task {task_id} cancelled")
            return True
        
        return False
    
    async def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            'queue_size': self.gpu_queue.qsize(),
            'active_workers': len([t for t in self.worker_tasks if not t.done()]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'processing_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PROCESSING]),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        }
    
    async def _worker_loop(self, worker_name: str):
        """Worker loop for processing tasks from the GPU queue"""
        print(f"👷 {worker_name} started")
        
        try:
            while self.is_running:
                try:
                    # Get next task from queue
                    priority, task_id = await asyncio.wait_for(self.gpu_queue.get(), timeout=1.0)
                    
                    # Process the task
                    await self._process_task(task_id, worker_name)
                    
                    # Mark task as done
                    self.gpu_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No tasks in queue, continue
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"❌ {worker_name} encountered error: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ {worker_name} worker loop failed: {e}")
        finally:
            print(f"👷 {worker_name} stopped")
    
    async def _process_task(self, task_id: str, worker_name: str):
        """Process a single analysis task"""
        if task_id not in self.tasks:
            print(f"⚠️ {worker_name}: Task {task_id} not found")
            return
        
        task = self.tasks[task_id]
        
        # Check if task was cancelled
        if task.status == TaskStatus.CANCELLED:
            return
        
        try:
            print(f"🎬 {worker_name}: Processing task {task_id} - {task.analysis_type}")
            
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            task.progress = 0.0
            
            # Step 1: Video processing (30% of progress)
            print(f"📹 {worker_name}: Processing video frames...")
            video_result = await video_processor.process_video(
                task.video_path, 
                task.analysis_type
            )
            
            if 'error' in video_result:
                raise Exception(f"Video processing failed: {video_result['error']}")
            
            task.progress = 30.0
            
            # Step 2: AI analysis (70% of progress)
            print(f"🤖 {worker_name}: Running AI analysis...")
            analysis_result = await minicpm_service.analyze_video(
                task.video_path,
                task.analysis_type,
                task.user_focus
            )
            
            task.progress = 100.0
            
            # Record performance metrics
            if task.started_at:
                total_time = (time.time() - task.started_at) * 1000
                self.performance_monitor.record_analysis_latency(total_time)
            
            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = {
                'video_processing': video_result,
                'ai_analysis': analysis_result,
                'total_processing_time_ms': (task.completed_at - task.started_at) * 1000 if task.started_at else 0
            }
            
            print(f"✅ {worker_name}: Task {task_id} completed successfully")
            
        except Exception as e:
            print(f"❌ {worker_name}: Task {task_id} failed: {e}")
            
            # Update task with error
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = str(e)
            task.progress = 0.0
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed/failed tasks"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if current_time - task.completed_at > max_age_seconds:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        if tasks_to_remove:
            print(f"🧹 Cleaned up {len(tasks_to_remove)} old tasks")
    
    async def get_performance_summary(self) -> Dict:
        """Get performance summary for all tasks"""
        if not self.tasks:
            return {"message": "No tasks processed yet"}
        
        # Calculate task statistics
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate average processing times
        processing_times = []
        for task in self.tasks.values():
            if task.started_at and task.completed_at:
                processing_time = task.completed_at - task.started_at
                processing_times.append(processing_time)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate_percent': round(success_rate, 1),
            'average_processing_time_seconds': round(avg_processing_time, 2),
            'queue_status': await self.get_queue_status(),
            'performance_metrics': self.performance_monitor.get_performance_summary()
        }
    
    async def export_task_report(self, task_id: str) -> Optional[str]:
        """Export detailed report for a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        report = f"""
# AI Video Detective - Task Report

## Task Information
- **Task ID**: {task.task_id}
- **Status**: {task.status.value.upper()}
- **Created**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.created_at))}
- **Analysis Type**: {task.analysis_type}
- **User Focus**: {task.user_focus}
- **Priority**: {task.priority.value.upper()}

## Processing Timeline
- **Created**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.created_at))}
"""
        
        if task.started_at:
            report += f"- **Started**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.started_at))}\n"
        
        if task.completed_at:
            report += f"- **Completed**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.completed_at))}\n"
            
            if task.started_at:
                processing_time = task.completed_at - task.started_at
                report += f"- **Processing Time**: {processing_time:.2f} seconds\n"
        
        report += f"- **Progress**: {task.progress:.1f}%\n"
        
        if task.error:
            report += f"\n## Error\n{task.error}\n"
        
        if task.result:
            report += f"\n## Results\n"
            
            if 'video_processing' in task.result:
                video_info = task.result['video_processing'].get('video_info', {})
                report += f"- **Video**: {video_info.get('file_name', 'Unknown')}\n"
                report += f"- **Duration**: {video_info.get('duration_minutes', 0):.1f} minutes\n"
                report += f"- **Resolution**: {video_info.get('resolution', 'Unknown')}\n"
                report += f"- **FPS**: {video_info.get('fps', 0):.1f}\n"
            
            if 'ai_analysis' in task.result:
                report += f"- **AI Analysis**: {len(str(task.result['ai_analysis']))} characters\n"
            
            if 'total_processing_time_ms' in task.result:
                report += f"- **Total Processing Time**: {task.result['total_processing_time_ms']:.2f}ms\n"
        
        return report

# Global instance
task_manager = TaskManager() 