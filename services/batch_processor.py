"""
Batch Processor Service for Round 2 - Efficient Video Processing
Handles batch video processing, memory management, and parallel processing
"""

import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from queue import Queue, Empty
import psutil
import gc
from config import Config
from services.performance_service import PerformanceMonitor
from services.gpu_service import GPUService

@dataclass
class ProcessingTask:
    """Represents a video processing task"""
    task_id: str
    video_path: str
    analysis_type: str
    priority: int = 1
    created_at: float = None
    status: str = "pending"
    progress: float = 0.0
    result: Dict = None
    error: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class ProcessingResult:
    """Represents the result of a processing task"""
    task_id: str
    success: bool
    data: Dict = None
    error: str = None
    processing_time: float = 0.0
    memory_used_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0

class BatchProcessor:
    """Batch video processor with memory management and parallel processing"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.gpu_service = GPUService()
        self.is_initialized = False
        
        # Task management
        self.task_queue = Queue()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.failed_tasks: Dict[str, ProcessingResult] = {}
        
        # Processing configuration
        self.max_concurrent_tasks = Config.BATCH_PROCESSING_CONFIG['max_concurrent_tasks']
        self.max_memory_usage_mb = Config.BATCH_PROCESSING_CONFIG['max_memory_usage_mb']
        self.max_gpu_memory_usage_mb = Config.BATCH_PROCESSING_CONFIG['max_gpu_memory_usage_mb']
        self.batch_size = Config.BATCH_PROCESSING_CONFIG['batch_size']
        self.chunk_duration_seconds = Config.BATCH_PROCESSING_CONFIG['chunk_duration_seconds']
        
        # Processing workers
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self.process_pool = ProcessPoolExecutor(max_workers=2)  # For CPU-intensive tasks
        
        # Memory monitoring
        self.memory_monitor_thread = None
        self.memory_monitor_running = False
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        
    async def initialize(self):
        """Initialize the batch processor"""
        try:
            print("🚀 Initializing Batch Processor...")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Start memory monitoring
            await self._start_memory_monitoring()
            
            # Start task processing loop
            asyncio.create_task(self._process_task_queue())
            
            self.is_initialized = True
            print("✅ Batch Processor initialized successfully")
            
        except Exception as e:
            print(f"❌ Batch Processor initialization failed: {e}")
            raise
    
    async def _start_memory_monitoring(self):
        """Start memory monitoring thread"""
        try:
            self.memory_monitor_running = True
            self.memory_monitor_thread = threading.Thread(
                target=self._memory_monitor_loop,
                daemon=True
            )
            self.memory_monitor_thread.start()
            print("📊 Memory monitoring started")
            
        except Exception as e:
            print(f"⚠️ Warning: Memory monitoring failed to start: {e}")
    
    def _memory_monitor_loop(self):
        """Memory monitoring loop running in separate thread"""
        while self.memory_monitor_running:
            try:
                # Monitor system memory
                memory_info = psutil.virtual_memory()
                memory_usage_mb = memory_info.used / (1024 * 1024)
                memory_percent = memory_info.percent
                
                # Monitor GPU memory if available
                gpu_memory_usage_mb = 0
                try:
                    gpu_memory_info = self.gpu_service.get_memory_info()
                    gpu_memory_usage_mb = gpu_memory_info.get('used_mb', 0)
                except:
                    pass
                
                # Record memory usage
                self.performance_monitor.record_memory_usage(memory_usage_mb, gpu_memory_usage_mb)
                
                # Check memory limits
                if memory_usage_mb > self.max_memory_usage_mb:
                    print(f"⚠️ Warning: High memory usage: {memory_usage_mb:.1f}MB")
                    self._handle_high_memory_usage()
                
                if gpu_memory_usage_mb > self.max_gpu_memory_usage_mb:
                    print(f"⚠️ Warning: High GPU memory usage: {gpu_memory_usage_mb:.1f}MB")
                    self._handle_high_gpu_memory_usage()
                
                # Sleep for monitoring interval
                time.sleep(Config.BATCH_PROCESSING_CONFIG['memory_monitor_interval'])
                
            except Exception as e:
                print(f"⚠️ Memory monitoring error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _handle_high_memory_usage(self):
        """Handle high memory usage by triggering garbage collection"""
        try:
            print("🧹 Triggering garbage collection due to high memory usage...")
            collected = gc.collect()
            print(f"✅ Garbage collection completed: {collected} objects collected")
            
            # Force memory cleanup
            if hasattr(self, '_force_memory_cleanup'):
                self._force_memory_cleanup()
                
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")
    
    def _handle_high_gpu_memory_usage(self):
        """Handle high GPU memory usage"""
        try:
            print("🎮 Triggering GPU memory cleanup...")
            
            # Clear GPU cache if available
            if hasattr(self.gpu_service, 'clear_cache'):
                self.gpu_service.clear_cache()
            
            # Force GPU memory cleanup
            if hasattr(self, '_force_gpu_memory_cleanup'):
                self._force_gpu_memory_cleanup()
                
        except Exception as e:
            print(f"⚠️ GPU memory cleanup failed: {e}")
    
    async def add_task(self, video_path: str, analysis_type: str = "general", priority: int = 1) -> str:
        """Add a video processing task to the queue"""
        try:
            # Validate video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{os.path.basename(video_path)}"
            
            # Create processing task
            task = ProcessingTask(
                task_id=task_id,
                video_path=video_path,
                analysis_type=analysis_type,
                priority=priority
            )
            
            # Add to queue
            self.task_queue.put((priority, task))
            
            print(f"📋 Task added to queue: {task_id} (Priority: {priority})")
            
            return task_id
            
        except Exception as e:
            print(f"❌ Failed to add task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Get the status of a specific task"""
        try:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': task.status,
                    'progress': task.progress,
                    'created_at': task.created_at,
                    'video_path': task.video_path,
                    'analysis_type': task.analysis_type
                }
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                result = self.completed_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'progress': 100.0,
                    'created_at': result.data.get('created_at', 0),
                    'processing_time': result.processing_time,
                    'memory_used': result.memory_used_mb,
                    'gpu_memory_used': result.gpu_memory_used_mb
                }
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                result = self.failed_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': result.error,
                    'created_at': result.data.get('created_at', 0) if result.data else 0
                }
            
            return {'error': 'Task not found'}
            
        except Exception as e:
            print(f"❌ Failed to get task status: {e}")
            return {'error': str(e)}
    
    async def get_queue_status(self) -> Dict:
        """Get the current status of the task queue"""
        try:
            return {
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_tasks_processed': self.total_tasks_processed,
                'avg_processing_time': self.avg_processing_time,
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                'gpu_memory_usage_mb': self.gpu_service.get_memory_info().get('used_mb', 0) if hasattr(self.gpu_service, 'get_memory_info') else 0
            }
            
        except Exception as e:
            print(f"❌ Failed to get queue status: {e}")
            return {'error': str(e)}
    
    async def _process_task_queue(self):
        """Main task processing loop"""
        print("🔄 Starting task processing loop...")
        
        while True:
            try:
                # Check if we can process more tasks
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    try:
                        # Get next task from queue (non-blocking)
                        priority, task = self.task_queue.get_nowait()
                        
                        # Add to active tasks
                        self.active_tasks[task.task_id] = task
                        task.status = "processing"
                        
                        # Start processing task
                        asyncio.create_task(self._process_single_task(task))
                        
                        print(f"🚀 Started processing task: {task.task_id}")
                        
                    except Empty:
                        # No tasks in queue
                        pass
                
                # Wait before next iteration
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Task processing loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_task(self, task: ProcessingTask):
        """Process a single video processing task"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        try:
            print(f"🎬 Processing video: {task.video_path}")
            
            # Update task status
            task.status = "processing"
            task.progress = 10.0
            
            # Process video in chunks
            result = await self._process_video_in_chunks(task)
            
            # Calculate processing metrics
            processing_time = time.time() - start_time
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_used = end_memory - start_memory
            
            # Get GPU memory usage
            gpu_memory_used = 0
            try:
                gpu_info = self.gpu_service.get_memory_info()
                gpu_memory_used = gpu_info.get('used_mb', 0)
            except:
                pass
            
            # Create processing result
            processing_result = ProcessingResult(
                task_id=task.task_id,
                success=True,
                data=result,
                processing_time=processing_time,
                memory_used_mb=memory_used,
                gpu_memory_used_mb=gpu_memory_used
            )
            
            # Store result
            self.completed_tasks[task.task_id] = processing_result
            
            # Update performance metrics
            self.total_tasks_processed += 1
            self.total_processing_time += processing_time
            self.avg_processing_time = self.total_processing_time / self.total_tasks_processed
            
            # Record performance
            self.performance_monitor.record_batch_processing_time(processing_time)
            self.performance_monitor.record_memory_usage(end_memory, gpu_memory_used)
            
            print(f"✅ Task completed successfully: {task.task_id}")
            
        except Exception as e:
            print(f"❌ Task failed: {task.task_id} - {e}")
            
            # Create failure result
            processing_time = time.time() - start_time
            processing_result = ProcessingResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                data={'created_at': task.created_at}
            )
            
            # Store failure result
            self.failed_tasks[task.task_id] = processing_result
            
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update task status
            task.status = "completed" if task.task_id in self.completed_tasks else "failed"
            task.progress = 100.0
    
    async def _process_video_in_chunks(self, task: ProcessingTask) -> Dict:
        """Process video in chunks for memory efficiency"""
        try:
            # Get video information
            video_info = await self._get_video_info(task.video_path)
            
            # Calculate chunk parameters
            total_frames = video_info['frame_count']
            fps = video_info['fps']
            chunk_frames = int(self.chunk_duration_seconds * fps)
            
            # Process video in chunks
            all_frames_data = []
            total_chunks = (total_frames + chunk_frames - 1) // chunk_frames
            
            for chunk_idx in range(total_chunks):
                # Update progress
                progress = (chunk_idx / total_chunks) * 100
                task.progress = 10.0 + (progress * 0.8)  # 10-90% range
                
                # Calculate chunk frame range
                start_frame = chunk_idx * chunk_frames
                end_frame = min(start_frame + chunk_frames, total_frames)
                
                print(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} (frames {start_frame}-{end_frame})")
                
                # Process chunk
                chunk_data = await self._process_video_chunk(
                    task.video_path, start_frame, end_frame, fps
                )
                
                all_frames_data.extend(chunk_data)
                
                # Memory cleanup after each chunk
                await self._cleanup_chunk_memory()
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Final progress update
            task.progress = 100.0
            
            return {
                'video_info': video_info,
                'frames_data': all_frames_data,
                'total_chunks': total_chunks,
                'total_frames_processed': len(all_frames_data),
                'processing_type': 'chunked_batch'
            }
            
        except Exception as e:
            print(f"❌ Chunked processing failed: {e}")
            raise
    
    async def _process_video_chunk(self, video_path: str, start_frame: int, end_frame: int, fps: float) -> List[Dict]:
        """Process a chunk of video frames"""
        try:
            import cv2
            
            frames_data = []
            cap = cv2.VideoCapture(video_path)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_idx = start_frame
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_data = await self._analyze_single_frame(frame, frame_idx, fps)
                frames_data.append(frame_data)
                
                frame_idx += 1
                
                # Progress update every 100 frames
                if (frame_idx - start_frame) % 100 == 0:
                    chunk_progress = ((frame_idx - start_frame) / (end_frame - start_frame)) * 100
                    print(f"  📊 Chunk progress: {chunk_progress:.1f}%")
            
            cap.release()
            
            return frames_data
            
        except Exception as e:
            print(f"❌ Chunk processing failed: {e}")
            return []
    
    async def _analyze_single_frame(self, frame, frame_idx: int, fps: float) -> Dict:
        """Analyze a single video frame"""
        try:
            frame_start = time.time()
            
            # Basic frame analysis (simplified for batch processing)
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'timestamp_formatted': self._format_timestamp(frame_idx / fps),
                'frame_size': frame.shape,
                'brightness': self._calculate_brightness(frame),
                'processing_time_ms': 0
            }
            
            # Calculate processing time
            frame_data['processing_time_ms'] = (time.time() - frame_start) * 1000
            
            return frame_data
            
        except Exception as e:
            print(f"⚠️ Frame analysis failed: {e}")
            return {
                'frame_idx': frame_idx,
                'error': str(e),
                'processing_time_ms': 0
            }
    
    def _calculate_brightness(self, frame) -> str:
        """Calculate frame brightness level"""
        try:
            import numpy as np
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 64:
                return 'dark'
            elif mean_brightness < 128:
                return 'medium'
            else:
                return 'bright'
                
        except Exception:
            return 'unknown'
    
    async def _get_video_info(self, video_path: str) -> Dict:
        """Get video information"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration_seconds,
                'duration_minutes': duration_seconds / 60,
                'resolution': f"{width}x{height}",
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return {}
    
    async def _cleanup_chunk_memory(self):
        """Clean up memory after processing a chunk"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear any cached data
            if hasattr(self, '_clear_cached_data'):
                self._clear_cached_data()
            
            # Small delay to allow memory cleanup
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or active task"""
        try:
            # Check if task is active
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = "cancelled"
                
                # Remove from active tasks
                del self.active_tasks[task_id]
                
                print(f"❌ Task cancelled: {task_id}")
                return True
            
            # Check if task is in queue (this is more complex with priority queue)
            # For now, we'll just return False for queued tasks
            print(f"⚠️ Cannot cancel queued task: {task_id}")
            return False
            
        except Exception as e:
            print(f"❌ Failed to cancel task: {e}")
            return False
    
    async def clear_completed_tasks(self, max_age_hours: int = 24):
        """Clear old completed tasks to free memory"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Clear old completed tasks
            old_completed = [
                task_id for task_id, result in self.completed_tasks.items()
                if current_time - result.data.get('created_at', 0) > max_age_seconds
            ]
            
            for task_id in old_completed:
                del self.completed_tasks[task_id]
            
            # Clear old failed tasks
            old_failed = [
                task_id for task_id, result in self.failed_tasks.items()
                if current_time - result.data.get('created_at', 0) > max_age_seconds
            ]
            
            for task_id in old_failed:
                del self.failed_tasks[task_id]
            
            print(f"🧹 Cleared {len(old_completed)} completed and {len(old_failed)} failed old tasks")
            
        except Exception as e:
            print(f"⚠️ Failed to clear old tasks: {e}")
    
    async def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            return {
                'total_tasks_processed': self.total_tasks_processed,
                'avg_processing_time': self.avg_processing_time,
                'total_processing_time': self.total_processing_time,
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'success_rate': (len(self.completed_tasks) / max(1, self.total_tasks_processed)) * 100,
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                'cpu_percent': psutil.cpu_percent()
            }
            
        except Exception as e:
            print(f"❌ Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up batch processor resources"""
        try:
            print("🧹 Cleaning up Batch Processor...")
            
            # Stop memory monitoring
            self.memory_monitor_running = False
            if self.memory_monitor_thread:
                self.memory_monitor_thread.join(timeout=5)
            
            # Shutdown thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # Clean up GPU service
            await self.gpu_service.cleanup()
            
            # Clear all tasks
            self.active_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            
            # Clear queue
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except Empty:
                    break
            
            print("✅ Batch Processor cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Batch Processor cleanup failed: {e}")

# Global instance for easy access
batch_processor = BatchProcessor() 