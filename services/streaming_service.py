"""
Streaming Service for Round 2 - Real-time Video Analysis with 7B Model
Handles real-time video streaming, live event detection, and continuous processing using Qwen2.5-VL-7B
"""

import os
import time
import asyncio
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor
from services.ai_service import ai_service
import psutil

@dataclass
class StreamEvent:
    """Represents a detected event in the video stream"""
    event_id: str
    event_type: str
    timestamp: float
    confidence: float
    bbox: List[int] = None
    metadata: Dict = None
    frame_data: Dict = None
    ai_analysis: str = None  # NEW: 7B model analysis

@dataclass
class StreamMetrics:
    """Represents streaming performance metrics"""
    fps_current: float
    fps_target: float
    latency_ms: float
    frame_processing_time_ms: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    events_detected: int
    stream_duration_seconds: float
    ai_analysis_count: int  # NEW: Count of 7B model analyses

class StreamingService:
    """Real-time video streaming service with live 7B model analysis"""
    
    def __init__(self):
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.ai_service = ai_service  # NEW: Direct integration with 7B model
        self.is_initialized = False
        
        # Streaming configuration
        self.fps_target = Config.STREAMING_CONFIG['fps_target']
        self.max_latency_ms = Config.STREAMING_CONFIG['max_latency_ms']
        self.event_detection_enabled = Config.STREAMING_CONFIG['event_detection_enabled']
        self.continuous_processing = Config.STREAMING_CONFIG['continuous_processing']
        self.real_time_7b_analysis = Config.STREAMING_CONFIG['real_time_7b_analysis']
        self.analysis_interval = Config.STREAMING_CONFIG['analysis_interval']
        
        # Stream management
        self.active_streams: Dict[str, Dict] = {}
        self.stream_events: Dict[str, List[StreamEvent]] = {}
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        
        # Event detection
        self.event_detectors: Dict[str, Callable] = {}
        self.event_thresholds = Config.STREAMING_CONFIG['event_thresholds']
        
        # Performance monitoring
        self.frame_times: List[float] = []
        self.latency_history: List[float] = []
        self.event_history: List[StreamEvent] = []
        
        # Processing queues
        self.frame_queue = Queue(maxsize=100)
        self.event_queue = Queue(maxsize=50)
        self.analysis_queue = Queue(maxsize=50)  # NEW: Queue for 7B model analysis
        
        # Background tasks
        self.processing_task = None
        self.monitoring_task = None
        self.analysis_task = None  # NEW: Background task for 7B model analysis
        self.is_running = False
        
    async def initialize(self):
        """Initialize the streaming service"""
        try:
            print("🚀 Initializing Streaming Service with 7B Model...")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Initialize AI service (7B model)
            if self.real_time_7b_analysis:
                await self.ai_service.initialize()
                print("✅ 7B Model initialized for streaming analysis")
            
            # Initialize event detectors
            await self._initialize_event_detectors()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            print("✅ Streaming Service initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Streaming Service: {e}")
            raise
    
    async def _initialize_event_detectors(self):
        """Initialize event detection algorithms"""
        try:
            print("🔍 Initializing event detectors...")
            
            # Basic motion detection
            self.event_detectors['motion'] = self._detect_motion_event
            
            # Object detection (basic)
            self.event_detectors['object'] = self._detect_object_event
            
            # Scene change detection
            self.event_detectors['scene_change'] = self._detect_scene_change_event
            
            # Anomaly detection
            self.event_detectors['anomaly'] = self._detect_anomaly_event
            
            print(f"✅ Initialized {len(self.event_detectors)} event detectors")
            
        except Exception as e:
            print(f"⚠️ Event detector initialization failed: {e}")
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        try:
            print("🔄 Starting background tasks...")
            
            # Start frame processing loop
            self.processing_task = asyncio.create_task(self._frame_processing_loop())
            
            # Start monitoring loop
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start AI analysis loop (NEW)
            if self.real_time_7b_analysis:
                self.analysis_task = asyncio.create_task(self._ai_analysis_loop())
                print("✅ AI analysis task started")
            
            self.is_running = True
            print("✅ Background tasks started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start background tasks: {e}")
            raise
    
    async def start_stream(self, stream_id: str, video_source: str, stream_config: Dict = None) -> bool:
        """Start a new video stream"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            print(f"🎬 Starting stream: {stream_id}")
            
            # Validate video source
            if not self._validate_video_source(video_source):
                raise ValueError(f"Invalid video source: {video_source}")
            
            # Create stream configuration
            config = stream_config or {}
            config.update({
                'fps_target': self.fps_target,
                'max_latency_ms': self.max_latency_ms,
                'event_detection': self.event_detection_enabled,
                'continuous_processing': self.continuous_processing,
                'ai_analysis_enabled': self.real_time_7b_analysis,
                'analysis_interval': self.analysis_interval
            })
            
            # Initialize stream
            stream_info = {
                'stream_id': stream_id,
                'video_source': video_source,
                'config': config,
                'start_time': time.time(),
                'status': 'active',
                'frame_count': 0,
                'event_count': 0,
                'ai_analysis_count': 0,  # NEW: Track 7B model analyses
                'last_frame_time': 0,
                'last_analysis_time': 0
            }
            
            # Initialize stream metrics
            self.stream_metrics[stream_id] = StreamMetrics(
                fps_current=0.0,
                fps_target=config['fps_target'],
                latency_ms=0.0,
                frame_processing_time_ms=0.0,
                memory_usage_mb=0.0,
                gpu_memory_usage_mb=0.0,
                events_detected=0,
                stream_duration_seconds=0.0,
                ai_analysis_count=0
            )
            
            # Initialize event storage
            self.stream_events[stream_id] = []
            
            # Store stream info
            self.active_streams[stream_id] = stream_info
            
            # Start stream processing
            asyncio.create_task(self._process_stream(stream_id, video_source, config))
            
            print(f"✅ Stream started: {stream_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start stream: {e}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop an active video stream"""
        try:
            if stream_id not in self.active_streams:
                print(f"⚠️ Stream not found: {stream_id}")
                return False
            
            print(f"🛑 Stopping stream: {stream_id}")
            
            # Update stream status
            self.active_streams[stream_id]['status'] = 'stopped'
            
            # Calculate final metrics
            stream_info = self.active_streams[stream_id]
            duration = time.time() - stream_info['start_time']
            
            # Final performance recording
            if stream_id in self.stream_metrics:
                metrics = self.stream_metrics[stream_id]
                metrics.stream_duration_seconds = duration
                
                # Record final performance
                self.performance_monitor.record_stream_duration(duration)
                self.performance_monitor.record_stream_events(stream_info['event_count'])
                self.performance_monitor.record_ai_analyses(stream_info['ai_analysis_count'])
            
            print(f"✅ Stream stopped: {stream_id} (Duration: {duration:.1f}s, Events: {stream_info['event_count']}, AI Analyses: {stream_info['ai_analysis_count']})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop stream: {e}")
            return False
    
    async def _process_stream(self, stream_id: str, video_source: str, config: Dict):
        """Process a video stream in real-time"""
        try:
            print(f"🔄 Processing stream: {stream_id}")
            
            # Open video source
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {video_source}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Stream properties: {frame_width}x{frame_height}, {fps:.1f} FPS")
            
            frame_count = 0
            last_time = time.time()
            
            while self.active_streams[stream_id]['status'] == 'active':
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                frame_count += 1
                
                # Calculate current FPS
                if current_time - last_time > 0:
                    current_fps = 1.0 / (current_time - last_time)
                    last_time = current_time
                    
                    # Update metrics
                    if stream_id in self.stream_metrics:
                        self.stream_metrics[stream_id].fps_current = current_fps
                
                # Add frame to processing queue
                frame_data = {
                    'stream_id': stream_id,
                    'frame': frame,
                    'timestamp': current_time,
                    'frame_number': frame_count,
                    'config': config
                }
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                except Queue.Full:
                    # Skip frame if queue is full
                    continue
                
                # Update stream info
                self.active_streams[stream_id]['frame_count'] = frame_count
                self.active_streams[stream_id]['last_frame_time'] = current_time
                
                # Control frame rate
                target_frame_time = 1.0 / config['fps_target']
                elapsed = time.time() - current_time
                if elapsed < target_frame_time:
                    await asyncio.sleep(target_frame_time - elapsed)
            
            cap.release()
            print(f"🔄 Stream processing completed: {stream_id}")
            
        except Exception as e:
            print(f"❌ Stream processing failed: {stream_id}: {e}")
            self.active_streams[stream_id]['status'] = 'error'
    
    async def _frame_processing_loop(self):
        """Process frames from the queue"""
        while self.is_running:
            try:
                # Get frame from queue
                try:
                    frame_data = self.frame_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process frame
                await self._process_single_frame(frame_data)
                
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_single_frame(self, frame_data: Dict):
        """Process a single frame"""
        try:
            stream_id = frame_data['stream_id']
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            config = frame_data['config']
            
            start_time = time.time()
            
            # Basic frame processing
            frame_result = {
                'timestamp': timestamp,
                'frame_number': frame_data['frame_number'],
                'size': frame.shape,
                'processing_time': 0
            }
            
            # Event detection
            if config.get('event_detection', False):
                events = await self._detect_events_in_frame(frame, frame_result, stream_id, timestamp)
                
                # Add events to queue for AI analysis
                for event in events:
                    try:
                        self.event_queue.put_nowait({
                            'event': event,
                            'stream_id': stream_id,
                            'frame': frame,
                            'timestamp': timestamp
                        })
                    except Queue.Full:
                        print(f"⚠️ Event queue full, skipping event: {event.event_type}")
                
                # Update metrics
                if stream_id in self.stream_metrics:
                    self.stream_metrics[stream_id].events_detected += len(events)
            
            # Add to AI analysis queue if enabled
            if config.get('ai_analysis_enabled', False) and self.real_time_7b_analysis:
                # Only analyze every Nth frame based on interval
                if frame_data['frame_number'] % config.get('analysis_interval', 1) == 0:
                    try:
                        self.analysis_queue.put_nowait({
                            'stream_id': stream_id,
                            'frame': frame,
                            'timestamp': timestamp,
                            'frame_number': frame_data['frame_number']
                        })
                    except Queue.Full:
                        print(f"⚠️ Analysis queue full, skipping frame: {frame_data['frame_number']}")
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            frame_result['processing_time'] = processing_time
            
            # Update metrics
            if stream_id in self.stream_metrics:
                self.stream_metrics[stream_id].frame_processing_time_ms = processing_time
                self.stream_metrics[stream_id].latency_ms = processing_time
            
        except Exception as e:
            print(f"❌ Single frame processing failed: {e}")
    
    async def _ai_analysis_loop(self):
        """Background loop for 7B model analysis - NEW"""
        while self.is_running:
            try:
                # Get analysis request from queue
                try:
                    analysis_data = self.analysis_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)
                    continue
                
                # Perform AI analysis
                await self._analyze_frame_with_7b_model(analysis_data)
                
            except Exception as e:
                print(f"❌ AI analysis loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _analyze_frame_with_7b_model(self, analysis_data: Dict):
        """Analyze frame using 7B model - NEW"""
        try:
            stream_id = analysis_data['stream_id']
            frame = analysis_data['frame']
            timestamp = analysis_data['timestamp']
            frame_number = analysis_data['frame_number']
            
            start_time = time.time()
            
            # Perform 7B model analysis
            analysis_result = await self.ai_service.analyze_stream_frame(
                frame, 
                analysis_type="realtime",
                user_focus="detect any notable events, objects, or activities"
            )
            
            # Create AI analysis event
            ai_event = StreamEvent(
                event_id=f"ai_{stream_id}_{frame_number}_{int(timestamp)}",
                event_type="ai_analysis",
                timestamp=timestamp,
                confidence=0.9,  # High confidence for AI analysis
                metadata={
                    'frame_number': frame_number,
                    'analysis_type': 'realtime',
                    'processing_time_ms': (time.time() - start_time) * 1000
                },
                ai_analysis=analysis_result
            )
            
            # Add to stream events
            if stream_id in self.stream_events:
                self.stream_events[stream_id].append(ai_event)
            
            # Update metrics
            if stream_id in self.stream_metrics:
                self.stream_metrics[stream_id].ai_analysis_count += 1
            
            # Update stream info
            if stream_id in self.active_streams:
                self.active_streams[stream_id]['ai_analysis_count'] += 1
                self.active_streams[stream_id]['last_analysis_time'] = timestamp
            
            print(f"🧠 AI Analysis completed for stream {stream_id}, frame {frame_number}: {analysis_result[:100]}...")
            
        except Exception as e:
            print(f"❌ AI analysis failed: {e}")
    
    async def _detect_events_in_frame(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> List[StreamEvent]:
        """Detect events in a frame"""
        events = []
        
        try:
            # Run all event detectors
            for event_type, detector in self.event_detectors.items():
                try:
                    event = await detector(frame, frame_result, stream_id, timestamp)
                    if event:
                        events.append(event)
                except Exception as e:
                    print(f"⚠️ Event detector {event_type} failed: {e}")
            
        except Exception as e:
            print(f"❌ Event detection failed: {e}")
        
        return events
    
    async def _detect_motion_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect motion in frame"""
        try:
            # Simple motion detection using frame difference
            # In a real implementation, you'd use more sophisticated methods
            
            # For now, return None (placeholder)
            return None
            
        except Exception as e:
            print(f"❌ Motion detection failed: {e}")
            return None
    
    async def _detect_object_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect objects in frame"""
        try:
            # Basic object detection placeholder
            # In a real implementation, you'd use YOLO or similar
            
            return None
            
        except Exception as e:
            print(f"❌ Object detection failed: {e}")
            return None
    
    async def _detect_scene_change_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect scene changes"""
        try:
            # Scene change detection placeholder
            return None
            
        except Exception as e:
            print(f"❌ Scene change detection failed: {e}")
            return None
    
    async def _detect_anomaly_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect anomalies in frame"""
        try:
            # Anomaly detection placeholder
            return None
            
        except Exception as e:
            print(f"❌ Anomaly detection failed: {e}")
            return None
    
    async def _monitoring_loop(self):
        """Monitor system performance and stream health"""
        while self.is_running:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check stream health
                await self._check_stream_health()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Get system memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # Update metrics for all active streams
            for stream_id in self.active_streams:
                if stream_id in self.stream_metrics:
                    self.stream_metrics[stream_id].memory_usage_mb = memory_usage_mb
                    
                    # Get GPU memory if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                            self.stream_metrics[stream_id].gpu_memory_usage_mb = gpu_memory
                    except:
                        pass
            
        except Exception as e:
            print(f"⚠️ System metrics update failed: {e}")
    
    async def _check_stream_health(self):
        """Check health of all active streams"""
        for stream_id, stream_info in list(self.active_streams.items()):
            try:
                if stream_info['status'] == 'active':
                    # Check if stream is responsive
                    last_frame_time = stream_info.get('last_frame_time', 0)
                    current_time = time.time()
                    
                    # If no frames for 10 seconds, mark as stalled
                    if current_time - last_frame_time > 10:
                        print(f"⚠️ Stream {stream_id} appears stalled, restarting...")
                        await self._restart_stream(stream_id)
                        
            except Exception as e:
                print(f"⚠️ Stream health check failed for {stream_id}: {e}")
    
    async def _restart_stream(self, stream_id: str):
        """Restart a stalled stream"""
        try:
            print(f"🔄 Restarting stream: {stream_id}")
            
            # Stop current stream
            await self.stop_stream(stream_id)
            
            # Get stream info
            stream_info = self.active_streams[stream_id]
            
            # Start new stream
            await self.start_stream(stream_id, stream_info['video_source'], stream_info['config'])
            
        except Exception as e:
            print(f"❌ Stream restart failed: {stream_id}: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old events and metrics"""
        try:
            current_time = time.time()
            cleanup_threshold = 3600  # 1 hour
            
            # Clean up old events
            for stream_id in self.stream_events:
                self.stream_events[stream_id] = [
                    event for event in self.stream_events[stream_id]
                    if current_time - event.timestamp < cleanup_threshold
                ]
            
            # Clean up old metrics
            if len(self.frame_times) > 1000:
                self.frame_times = self.frame_times[-500:]
            
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-500:]
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def _validate_video_source(self, video_source: str) -> bool:
        """Validate video source"""
        try:
            # Check if it's a file path
            if os.path.exists(video_source):
                return True
            
            # Check if it's a camera index
            if video_source.isdigit():
                cap = cv2.VideoCapture(int(video_source))
                if cap.isOpened():
                    cap.release()
                    return True
            
            # Check if it's a URL
            if video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Video source validation failed: {e}")
            return False
    
    async def get_stream_status(self, stream_id: str) -> Dict:
        """Get status of a specific stream"""
        try:
            if stream_id not in self.active_streams:
                return {'error': 'Stream not found'}
            
            stream_info = self.active_streams[stream_id]
            metrics = self.stream_metrics.get(stream_id, {})
            events = self.stream_events.get(stream_id, [])
            
            return {
                'stream_id': stream_id,
                'status': stream_info['status'],
                'video_source': stream_info['video_source'],
                'start_time': stream_info['start_time'],
                'frame_count': stream_info['frame_count'],
                'event_count': stream_info['event_count'],
                'ai_analysis_count': stream_info.get('ai_analysis_count', 0),
                'metrics': metrics,
                'recent_events': events[-10:],  # Last 10 events
                'config': stream_info['config']
            }
            
        except Exception as e:
            return {'error': f'Failed to get stream status: {str(e)}'}
    
    async def get_all_streams_status(self) -> Dict:
        """Get status of all streams"""
        try:
            streams_status = {}
            
            for stream_id in self.active_streams:
                streams_status[stream_id] = await self.get_stream_status(stream_id)
            
            return {
                'total_streams': len(self.active_streams),
                'active_streams': len([s for s in self.active_streams.values() if s['status'] == 'active']),
                'streams': streams_status
            }
            
        except Exception as e:
            return {'error': f'Failed to get streams status: {str(e)}'}
    
    async def get_performance_stats(self) -> Dict:
        """Get overall performance statistics"""
        try:
            total_events = sum(len(events) for events in self.stream_events.values())
            total_ai_analyses = sum(stream.get('ai_analysis_count', 0) for stream in self.active_streams.values())
            
            return {
                'total_streams': len(self.active_streams),
                'total_events': total_events,
                'total_ai_analyses': total_ai_analyses,
                'average_latency': np.mean(self.latency_history) if self.latency_history else 0,
                'average_fps': np.mean(self.frame_times) if self.frame_times else 0,
                'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
                'gpu_memory_mb': self._get_gpu_memory_usage()
            }
            
        except Exception as e:
            return {'error': f'Failed to get performance stats: {str(e)}'}
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            pass
        return 0.0
    
    async def cleanup(self):
        """Clean up streaming service"""
        try:
            print("🧹 Cleaning up Streaming Service...")
            
            # Stop all streams
            for stream_id in list(self.active_streams.keys()):
                await self.stop_stream(stream_id)
            
            # Stop background tasks
            self.is_running = False
            
            if self.processing_task:
                self.processing_task.cancel()
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            if self.analysis_task:
                self.analysis_task.cancel()
            
            # Clear queues
            while not self.frame_queue.empty():
                self.frame_queue.get()
            
            while not self.event_queue.empty():
                self.event_queue.get()
            
            while not self.analysis_queue.empty():
                self.analysis_queue.get()
            
            print("✅ Streaming Service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

# Create global streaming service instance
streaming_service = StreamingService() 