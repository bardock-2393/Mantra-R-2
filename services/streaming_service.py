"""
Streaming Service for Round 2 - Real-time Video Analysis
Handles real-time video streaming, live event detection, and continuous processing
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
from models.deepstream_pipeline import DeepStreamPipeline
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

class StreamingService:
    """Real-time video streaming service with live event detection"""
    
    def __init__(self):
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.deepstream_pipeline = DeepStreamPipeline()
        self.is_initialized = False
        
        # Streaming configuration
        self.fps_target = Config.STREAMING_CONFIG['fps_target']
        self.max_latency_ms = Config.STREAMING_CONFIG['max_latency_ms']
        self.event_detection_enabled = Config.STREAMING_CONFIG['event_detection_enabled']
        self.continuous_processing = Config.STREAMING_CONFIG['continuous_processing']
        
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
        
        # Background tasks
        self.processing_task = None
        self.monitoring_task = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize the streaming service"""
        try:
            print("🚀 Initializing Streaming Service...")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Initialize DeepStream pipeline
            await self.deepstream_pipeline.initialize()
            
            # Initialize event detectors
            await self._initialize_event_detectors()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            print("✅ Streaming Service initialized successfully")
            
        except Exception as e:
            print(f"❌ Streaming Service initialization failed: {e}")
            raise
    
    async def _initialize_event_detectors(self):
        """Initialize event detection models"""
        try:
            print("🔍 Initializing event detectors...")
            
            # Motion detection
            self.event_detectors['motion'] = self._detect_motion_event
            
            # Object detection
            self.event_detectors['object'] = self._detect_object_event
            
            # Scene change detection
            self.event_detectors['scene_change'] = self._detect_scene_change_event
            
            # Anomaly detection
            self.event_detectors['anomaly'] = self._detect_anomaly_event
            
            print(f"✅ Initialized {len(self.event_detectors)} event detectors")
            
        except Exception as e:
            print(f"⚠️ Warning: Event detector initialization failed: {e}")
    
    async def _start_background_tasks(self):
        """Start background processing and monitoring tasks"""
        try:
            self.is_running = True
            
            # Start frame processing task
            self.processing_task = asyncio.create_task(self._frame_processing_loop())
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            print("🔄 Background tasks started")
            
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
                'continuous_processing': self.continuous_processing
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
                'last_frame_time': 0
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
                stream_duration_seconds=0.0
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
            
            print(f"✅ Stream stopped: {stream_id} (Duration: {duration:.1f}s, Events: {stream_info['event_count']})")
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
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video properties - FPS: {fps}, Frames: {frame_count}")
            
            frame_idx = 0
            last_frame_time = time.time()
            target_frame_interval = 1.0 / config['fps_target']
            
            while self.active_streams[stream_id]['status'] == 'active':
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print(f"📺 End of stream: {stream_id}")
                    break
                
                # Calculate timing
                current_time = time.time()
                frame_interval = current_time - last_frame_time
                
                # Process frame at target FPS
                if frame_interval >= target_frame_interval:
                    # Add frame to processing queue
                    if not self.frame_queue.full():
                        self.frame_queue.put({
                            'stream_id': stream_id,
                            'frame': frame,
                            'frame_idx': frame_idx,
                            'timestamp': current_time,
                            'config': config
                        })
                    
                    # Update stream info
                    self.active_streams[stream_id]['frame_count'] += 1
                    self.active_streams[stream_id]['last_frame_time'] = current_time
                    
                    # Update metrics
                    if stream_id in self.stream_metrics:
                        metrics = self.stream_metrics[stream_id]
                        metrics.fps_current = 1.0 / frame_interval
                        metrics.stream_duration_seconds = current_time - self.active_streams[stream_id]['start_time']
                    
                    last_frame_time = current_time
                    frame_idx += 1
                    
                    # Progress update
                    if frame_idx % 100 == 0:
                        progress = (frame_idx / frame_count) * 100 if frame_count > 0 else 0
                        print(f"📊 Stream {stream_id} progress: {progress:.1f}%")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
            
            # Cleanup
            cap.release()
            print(f"🧹 Stream cleanup completed: {stream_id}")
            
        except Exception as e:
            print(f"❌ Stream processing failed: {stream_id} - {e}")
            self.active_streams[stream_id]['status'] = 'error'
    
    async def _frame_processing_loop(self):
        """Main frame processing loop"""
        print("🔄 Starting frame processing loop...")
        
        while self.is_running:
            try:
                # Get frame from queue
                try:
                    frame_data = self.frame_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.001)
                    continue
                
                # Process frame
                await self._process_single_frame(frame_data)
                
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_single_frame(self, frame_data: Dict):
        """Process a single frame from the stream"""
        try:
            stream_id = frame_data['stream_id']
            frame = frame_data['frame']
            frame_idx = frame_data['frame_idx']
            timestamp = frame_data['timestamp']
            config = frame_data['config']
            
            # Start timing
            processing_start = time.time()
            
            # Process frame with DeepStream pipeline
            if self.deepstream_pipeline.is_initialized:
                frame_result = await self.deepstream_pipeline._analyze_frame_deepstream(
                    frame, frame_idx, {'fps': config['fps_target']}
                )
            else:
                frame_result = await self.deepstream_pipeline._analyze_frame_opencv(
                    frame, frame_idx, {'fps': config['fps_target']}
                )
            
            # Calculate processing time
            processing_time = (time.time() - processing_start) * 1000
            
            # Update metrics
            if stream_id in self.stream_metrics:
                metrics = self.stream_metrics[stream_id]
                metrics.frame_processing_time_ms = processing_time
                
                # Calculate latency
                current_time = time.time()
                latency = (current_time - timestamp) * 1000
                metrics.latency_ms = latency
                
                # Record performance
                self.performance_monitor.record_frame_processing_latency(processing_time)
                self.performance_monitor.record_streaming_latency(latency)
            
            # Event detection
            if config.get('event_detection', False):
                events = await self._detect_events_in_frame(frame, frame_result, stream_id, timestamp)
                
                # Add events to queue
                for event in events:
                    if not self.event_queue.full():
                        self.event_queue.put(event)
                    
                    # Update stream metrics
                    if stream_id in self.stream_metrics:
                        self.stream_metrics[stream_id].events_detected += 1
                    
                    # Store event
                    if stream_id in self.stream_events:
                        self.stream_events[stream_id].append(event)
            
            # Memory management
            if len(self.frame_times) > 1000:
                self.frame_times = self.frame_times[-500:]
            
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-500:]
            
            # Record frame time
            self.frame_times.append(processing_time)
            
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
    
    async def _detect_events_in_frame(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> List[StreamEvent]:
        """Detect events in a single frame"""
        events = []
        
        try:
            # Motion detection
            if 'motion' in self.event_detectors:
                motion_event = await self.event_detectors['motion'](frame, frame_result, stream_id, timestamp)
                if motion_event:
                    events.append(motion_event)
            
            # Object detection
            if 'object' in self.event_detectors:
                object_event = await self.event_detectors['object'](frame, frame_result, stream_id, timestamp)
                if object_event:
                    events.append(object_event)
            
            # Scene change detection
            if 'scene_change' in self.event_detectors:
                scene_event = await self.event_detectors['scene_change'](frame, frame_result, stream_id, timestamp)
                if scene_event:
                    events.append(scene_event)
            
            # Anomaly detection
            if 'anomaly' in self.event_detectors:
                anomaly_event = await self.event_detectors['anomaly'](frame, frame_result, stream_id, timestamp)
                if anomaly_event:
                    events.append(anomaly_event)
            
        except Exception as e:
            print(f"⚠️ Event detection failed: {e}")
        
        return events
    
    async def _detect_motion_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect motion events in frame"""
        try:
            motion_data = frame_result.get('motion_detection', {})
            
            if motion_data.get('motion_detected', False):
                motion_intensity = motion_data.get('motion_intensity', 0.0)
                
                # Check if motion intensity exceeds threshold
                if motion_intensity > self.event_thresholds.get('motion', 0.1):
                    event = StreamEvent(
                        event_id=f"motion_{stream_id}_{int(timestamp * 1000)}",
                        event_type='motion',
                        timestamp=timestamp,
                        confidence=motion_intensity,
                        bbox=motion_data.get('motion_regions', []),
                        metadata={
                            'motion_intensity': motion_intensity,
                            'motion_type': motion_data.get('motion_type', 'unknown')
                        },
                        frame_data=frame_result
                    )
                    
                    return event
            
            return None
            
        except Exception as e:
            print(f"⚠️ Motion detection failed: {e}")
            return None
    
    async def _detect_object_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect object events in frame"""
        try:
            objects = frame_result.get('objects', [])
            
            for obj in objects:
                confidence = obj.get('confidence', 0.0)
                
                # Check if object confidence exceeds threshold
                if confidence > self.event_thresholds.get('object', 0.7):
                    event = StreamEvent(
                        event_id=f"object_{stream_id}_{int(timestamp * 1000)}",
                        event_type='object_detection',
                        timestamp=timestamp,
                        confidence=confidence,
                        bbox=obj.get('bbox', []),
                        metadata={
                            'object_class': obj.get('class', 'unknown'),
                            'track_id': obj.get('track_id', 0)
                        },
                        frame_data=frame_result
                    )
                    
                    return event
            
            return None
            
        except Exception as e:
            print(f"⚠️ Object detection failed: {e}")
            return None
    
    async def _detect_scene_change_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect scene change events in frame"""
        try:
            # This is a simplified scene change detection
            # In production, you'd use more sophisticated methods
            
            # Check if we have previous frame data
            if not hasattr(self, '_prev_scene_data'):
                self._prev_scene_data = {}
            
            if stream_id not in self._prev_scene_data:
                self._prev_scene_data[stream_id] = frame_result
                return None
            
            prev_frame = self._prev_scene_data[stream_id]
            
            # Calculate scene similarity (simplified)
            scene_similarity = self._calculate_scene_similarity(frame_result, prev_frame)
            
            # Check if scene change threshold is exceeded
            if scene_similarity < self.event_thresholds.get('scene_change', 0.8):
                event = StreamEvent(
                    event_id=f"scene_change_{stream_id}_{int(timestamp * 1000)}",
                    event_type='scene_change',
                    timestamp=timestamp,
                    confidence=1.0 - scene_similarity,
                    metadata={
                        'scene_similarity': scene_similarity,
                        'change_type': 'scene_transition'
                    },
                    frame_data=frame_result
                )
                
                # Update previous scene data
                self._prev_scene_data[stream_id] = frame_result
                
                return event
            
            # Update previous scene data
            self._prev_scene_data[stream_id] = frame_result
            return None
            
        except Exception as e:
            print(f"⚠️ Scene change detection failed: {e}")
            return None
    
    async def _detect_anomaly_event(self, frame: np.ndarray, frame_result: Dict, stream_id: str, timestamp: float) -> Optional[StreamEvent]:
        """Detect anomaly events in frame"""
        try:
            # This is a simplified anomaly detection
            # In production, you'd use more sophisticated methods
            
            # Check for unusual processing times
            processing_time = frame_result.get('processing_time_ms', 0)
            avg_processing_time = np.mean(self.frame_times) if self.frame_times else 0
            
            if avg_processing_time > 0 and processing_time > avg_processing_time * 2:
                event = StreamEvent(
                    event_id=f"anomaly_{stream_id}_{int(timestamp * 1000)}",
                    event_type='processing_anomaly',
                    timestamp=timestamp,
                    confidence=0.8,
                    metadata={
                        'processing_time_ms': processing_time,
                        'avg_processing_time_ms': avg_processing_time,
                        'anomaly_type': 'high_processing_time'
                    },
                    frame_data=frame_result
                )
                
                return event
            
            return None
            
        except Exception as e:
            print(f"⚠️ Anomaly detection failed: {e}")
            return None
    
    def _calculate_scene_similarity(self, current_frame: Dict, previous_frame: Dict) -> float:
        """Calculate similarity between two frames"""
        try:
            # Simplified similarity calculation
            # In production, you'd use more sophisticated methods like feature matching
            
            # Compare basic properties
            current_objects = len(current_frame.get('objects', []))
            previous_objects = len(previous_frame.get('objects', []))
            
            # Compare scene analysis
            current_scene = current_frame.get('scene_analysis', {})
            previous_scene = previous_frame.get('scene_analysis', {})
            
            # Calculate similarity score
            object_similarity = 1.0 - abs(current_objects - previous_objects) / max(1, max(current_objects, previous_objects))
            
            # Scene analysis similarity (simplified)
            scene_similarity = 0.8  # Placeholder
            
            # Combined similarity
            total_similarity = (object_similarity + scene_similarity) / 2
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            print(f"⚠️ Scene similarity calculation failed: {e}")
            return 0.5  # Default similarity
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        print("📊 Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check stream health
                await self._check_stream_health()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(Config.STREAMING_CONFIG['monitoring_interval'])
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_mb = memory_info.used / (1024 * 1024)
            
            # Get GPU memory usage
            gpu_memory_usage_mb = 0
            try:
                gpu_info = self.gpu_service.get_memory_info()
                gpu_memory_usage_mb = gpu_info.get('used_mb', 0)
            except:
                pass
            
            # Update stream metrics
            for stream_id in self.active_streams:
                if stream_id in self.stream_metrics:
                    metrics = self.stream_metrics[stream_id]
                    metrics.memory_usage_mb = memory_usage_mb
                    metrics.gpu_memory_usage_mb = gpu_memory_usage_mb
            
            # Record performance
            self.performance_monitor.record_memory_usage(memory_usage_mb, gpu_memory_usage_mb)
            
        except Exception as e:
            print(f"⚠️ System metrics update failed: {e}")
    
    async def _check_stream_health(self):
        """Check health of active streams"""
        try:
            current_time = time.time()
            
            for stream_id, stream_info in list(self.active_streams.items()):
                if stream_info['status'] != 'active':
                    continue
                
                # Check for stalled streams
                last_frame_time = stream_info.get('last_frame_time', 0)
                if last_frame_time > 0 and (current_time - last_frame_time) > 10:
                    print(f"⚠️ Stream appears stalled: {stream_id}")
                    
                    # Check if we should restart the stream
                    if (current_time - last_frame_time) > 30:
                        print(f"🔄 Restarting stalled stream: {stream_id}")
                        await self._restart_stream(stream_id)
                
                # Check performance metrics
                if stream_id in self.stream_metrics:
                    metrics = self.stream_metrics[stream_id]
                    
                    # Check latency
                    if metrics.latency_ms > self.max_latency_ms:
                        print(f"⚠️ High latency detected: {stream_id} - {metrics.latency_ms:.1f}ms")
                    
                    # Check FPS
                    if metrics.fps_current < metrics.fps_target * 0.8:
                        print(f"⚠️ Low FPS detected: {stream_id} - {metrics.fps_current:.1f} < {metrics.fps_target}")
            
        except Exception as e:
            print(f"⚠️ Stream health check failed: {e}")
    
    async def _restart_stream(self, stream_id: str):
        """Restart a stalled stream"""
        try:
            print(f"🔄 Restarting stream: {stream_id}")
            
            # Stop current stream
            await self.stop_stream(stream_id)
            
            # Get stream configuration
            stream_info = self.active_streams[stream_id]
            video_source = stream_info['video_source']
            config = stream_info['config']
            
            # Remove old stream info
            del self.active_streams[stream_id]
            if stream_id in self.stream_events:
                del self.stream_events[stream_id]
            if stream_id in self.stream_metrics:
                del self.stream_metrics[stream_id]
            
            # Wait a moment before restarting
            await asyncio.sleep(1)
            
            # Restart stream
            await self.start_stream(stream_id, video_source, config)
            
        except Exception as e:
            print(f"❌ Stream restart failed: {stream_id} - {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        try:
            current_time = time.time()
            max_age_seconds = Config.STREAMING_CONFIG['data_retention_seconds']
            
            # Clean up old events
            for stream_id in self.stream_events:
                events = self.stream_events[stream_id]
                old_events = [
                    event for event in events
                    if (current_time - event.timestamp) > max_age_seconds
                ]
                
                for event in old_events:
                    events.remove(event)
                
                if old_events:
                    print(f"🧹 Cleaned up {len(old_events)} old events from stream: {stream_id}")
            
            # Clean up old frame times and latency history
            if len(self.frame_times) > 1000:
                self.frame_times = self.frame_times[-500:]
            
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-500:]
            
        except Exception as e:
            print(f"⚠️ Data cleanup failed: {e}")
    
    def _validate_video_source(self, video_source: str) -> bool:
        """Validate video source"""
        try:
            # Check if it's a file path
            if os.path.exists(video_source):
                return True
            
            # Check if it's a URL
            if video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                return True
            
            # Check if it's a device index
            if video_source.isdigit():
                return True
            
            return False
            
        except Exception:
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
                'metrics': metrics,
                'recent_events': events[-10:] if events else [],  # Last 10 events
                'uptime_seconds': time.time() - stream_info['start_time']
            }
            
        except Exception as e:
            print(f"❌ Failed to get stream status: {e}")
            return {'error': str(e)}
    
    async def get_all_streams_status(self) -> Dict:
        """Get status of all active streams"""
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
            print(f"❌ Failed to get all streams status: {e}")
            return {'error': str(e)}
    
    async def get_performance_stats(self) -> Dict:
        """Get overall performance statistics"""
        try:
            return {
                'total_streams': len(self.active_streams),
                'active_streams': len([s for s in self.active_streams.values() if s['status'] == 'active']),
                'total_events': sum(len(events) for events in self.stream_events.values()),
                'avg_frame_processing_time': np.mean(self.frame_times) if self.frame_times else 0,
                'avg_latency': np.mean(self.latency_history) if self.latency_history else 0,
                'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                'cpu_percent': psutil.cpu_percent()
            }
            
        except Exception as e:
            print(f"❌ Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up streaming service resources"""
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
            
            # Clean up GPU service
            await self.gpu_service.cleanup()
            
            # Clean up DeepStream pipeline
            await self.deepstream_pipeline.cleanup()
            
            # Clear all data
            self.active_streams.clear()
            self.stream_events.clear()
            self.stream_metrics.clear()
            self.frame_times.clear()
            self.latency_history.clear()
            self.event_history.clear()
            
            print("✅ Streaming Service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Streaming Service cleanup failed: {e}")

# Global instance for easy access
streaming_service = StreamingService() 