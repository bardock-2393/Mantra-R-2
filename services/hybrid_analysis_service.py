"""
Hybrid Analysis Service - DeepStream + 7B Model + Vector Search
Combines the best of all three systems for maximum accuracy and performance
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config import Config
from models.deepstream_pipeline import DeepStreamPipeline
from services.ai_service import ai_service
from services.vector_search_service import VectorSearchService
from services.gpu_service import GPUService

class HybridAnalysisService:
    """Hybrid video analysis service combining DeepStream + 7B model + Vector search"""
    
    def __init__(self):
        self.deepstream_pipeline = DeepStreamPipeline()
        self.ai_service = ai_service
        self.vector_service = VectorSearchService()
        self.gpu_service = GPUService()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all components of the hybrid system"""
        try:
            print("🚀 Initializing Hybrid Analysis Service...")
            
            # Initialize GPU service first
            await self.gpu_service.initialize()
            
            # Initialize DeepStream pipeline
            print("🔍 Initializing DeepStream pipeline...")
            await self.deepstream_pipeline.initialize()
            
            # Initialize AI service (7B model)
            print("🧠 Initializing AI service (7B model)...")
            await self.ai_service.initialize()
            
            # Initialize vector search service
            print("💾 Initializing vector search service...")
            await self.vector_service.initialize()
            
            self.is_initialized = True
            print("✅ Hybrid Analysis Service initialized successfully")
            print("   - DeepStream Pipeline: Ready")
            print("   - 7B Model Service: Ready") 
            print("   - Vector Search Service: Ready")
            
        except Exception as e:
            print(f"❌ Hybrid Analysis Service initialization failed: {e}")
            raise
    
    async def analyze_video_hybrid(self, video_path: str, analysis_type: str = "hybrid") -> Dict:
        """Analyze video using the hybrid approach: DeepStream + 7B + Vector Search"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            print(f"🎬 Starting hybrid analysis of: {video_path}")
            
            # Phase 1: DeepStream Real-Time Analysis
            print("🔍 Phase 1: DeepStream Analysis...")
            try:
                deepstream_results = await self.deepstream_pipeline.process_video(video_path)
                
                if 'error' in deepstream_results:
                    print(f"⚠️ DeepStream analysis failed: {deepstream_results['error']}")
                    print("🔄 Falling back to basic video processing...")
                    # Fallback to basic video processing
                    deepstream_results = await self._fallback_video_processing(video_path)
            except Exception as e:
                print(f"⚠️ DeepStream analysis error: {e}")
                print("🔄 Falling back to basic video processing...")
                deepstream_results = await self._fallback_video_processing(video_path)
            
            # Phase 2: 7B Model Content Understanding
            print("🧠 Phase 2: 7B Model Analysis...")
            qwen_results = await self._analyze_with_7b_model(deepstream_results, video_path)
            
            # Phase 3: Combine Results
            print("🔗 Phase 3: Combining Results...")
            combined_results = self._combine_analysis_results(deepstream_results, qwen_results)
            
            # Phase 4: Vector Database Storage
            print("💾 Phase 4: Vector Database Storage...")
            session_id = self._generate_session_id(video_path)
            await self._store_in_vector_database(session_id, combined_results)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            video_duration = deepstream_results.get('video_info', {}).get('duration_seconds', 0)
            
            performance_metrics = {
                'total_processing_time': processing_time,
                'video_duration_seconds': video_duration,
                'processing_speed_ratio': video_duration / processing_time if processing_time > 0 else 0,
                'deepstream_fps': deepstream_results.get('processing_metrics', {}).get('actual_fps', 0),
                'frames_processed': deepstream_results.get('processing_metrics', {}).get('frames_processed', 0)
            }
            
            final_results = {
                'session_id': session_id,
                'video_path': video_path,
                'analysis_type': analysis_type,
                'deepstream_results': deepstream_results,
                'qwen_results': qwen_results,
                'combined_results': combined_results,
                'performance_metrics': performance_metrics,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            print(f"✅ Hybrid analysis completed in {processing_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            print(f"❌ Hybrid analysis failed: {e}")
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_with_7b_model(self, deepstream_results: Dict, video_path: str) -> Dict:
        """Analyze video content using the 7B model based on DeepStream insights"""
        try:
            # Extract key frames and insights from DeepStream results
            key_frames = self._extract_key_frames(deepstream_results)
            object_summary = self._summarize_objects(deepstream_results)
            motion_summary = self._summarize_motion(deepstream_results)
            
            # Prepare context for 7B model
            analysis_context = {
                'video_path': video_path,
                'key_frames': key_frames,
                'object_summary': object_summary,
                'motion_summary': motion_summary,
                'video_info': deepstream_results.get('video_info', {})
            }
            
            # Generate comprehensive analysis using 7B model
            qwen_analysis = await self.ai_service.analyze_video(
                video_path=video_path,
                analysis_type="hybrid_comprehensive",
                user_focus="comprehensive_analysis"
            )
            
            return {
                'content_analysis': qwen_analysis,
                'key_frames_analyzed': len(key_frames),
                'object_summary': object_summary,
                'motion_summary': motion_summary,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ 7B model analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_key_frames(self, deepstream_results: Dict) -> List[Dict]:
        """Extract key frames for 7B model analysis"""
        try:
            frames_data = deepstream_results.get('frames_data', [])
            key_frames = []
            
            # Select frames with significant events
            for frame_data in frames_data:
                if self._is_key_frame(frame_data):
                    key_frames.append({
                        'frame_idx': frame_data.get('frame_idx'),
                        'timestamp': frame_data.get('timestamp'),
                        'objects': frame_data.get('objects', []),
                        'motion': frame_data.get('motion_detection', {}),
                        'scene': frame_data.get('scene_analysis', {})
                    })
            
            # Limit to reasonable number for 7B model
            return key_frames[:20]  # Max 20 key frames
            
        except Exception as e:
            print(f"⚠️ Key frame extraction failed: {e}")
            return []
    
    def _is_key_frame(self, frame_data: Dict) -> bool:
        """Determine if a frame is a key frame"""
        try:
            # Check for object detection
            objects = frame_data.get('objects', [])
            if len(objects) > 0:
                return True
            
            # Check for motion
            motion = frame_data.get('motion_detection', {})
            if motion.get('motion_detected', False) and motion.get('motion_intensity', 0) > 0.3:
                return True
            
            # Check for scene changes
            scene = frame_data.get('scene_analysis', {})
            if scene.get('scene_type') in ['outdoor', 'indoor', 'transition']:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _summarize_objects(self, deepstream_results: Dict) -> Dict:
        """Summarize object detection results"""
        try:
            frames_data = deepstream_results.get('frames_data', [])
            object_counts = {}
            object_timestamps = {}
            
            for frame_data in frames_data:
                objects = frame_data.get('objects', [])
                timestamp = frame_data.get('timestamp', 0)
                
                for obj in objects:
                    obj_class = obj.get('class', 'unknown')
                    
                    # Count objects
                    if obj_class not in object_counts:
                        object_counts[obj_class] = 0
                    object_counts[obj_class] += 1
                    
                    # Track timestamps
                    if obj_class not in object_timestamps:
                        object_timestamps[obj_class] = []
                    object_timestamps[obj_class].append(timestamp)
            
            return {
                'total_objects_detected': sum(object_counts.values()),
                'object_types': object_counts,
                'object_timestamps': object_timestamps,
                'most_common_objects': sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            print(f"⚠️ Object summarization failed: {e}")
            return {}
    
    def _summarize_motion(self, deepstream_results: Dict) -> Dict:
        """Summarize motion detection results"""
        try:
            frames_data = deepstream_results.get('frames_data', [])
            motion_events = []
            total_motion_frames = 0
            
            for frame_data in frames_data:
                motion = frame_data.get('motion_detection', {})
                if motion.get('motion_detected', False):
                    total_motion_frames += 1
                    motion_events.append({
                        'timestamp': frame_data.get('timestamp', 0),
                        'intensity': motion.get('motion_intensity', 0),
                        'type': motion.get('motion_type', 'unknown')
                    })
            
            return {
                'total_motion_frames': total_motion_frames,
                'motion_events': motion_events,
                'average_motion_intensity': sum([e['intensity'] for e in motion_events]) / len(motion_events) if motion_events else 0,
                'motion_timeline': sorted(motion_events, key=lambda x: x['timestamp'])
            }
            
        except Exception as e:
            print(f"⚠️ Motion summarization failed: {e}")
            return {}
    
    def _combine_analysis_results(self, deepstream_results: Dict, qwen_results: Dict) -> Dict:
        """Combine DeepStream and 7B model results into unified analysis"""
        try:
            combined = {
                'video_metadata': deepstream_results.get('video_info', {}),
                'real_time_analysis': {
                    'object_detection': deepstream_results.get('frames_data', []),
                    'motion_analysis': deepstream_results.get('frames_data', []),
                    'performance_metrics': deepstream_results.get('processing_metrics', {})
                },
                'intelligent_analysis': {
                    'content_understanding': qwen_results.get('content_analysis', {}),
                    'object_summary': qwen_results.get('object_summary', {}),
                    'motion_summary': qwen_results.get('motion_summary', {}),
                    'key_frames': qwen_results.get('key_frames_analyzed', 0)
                },
                'analysis_summary': {
                    'total_frames_analyzed': len(deepstream_results.get('frames_data', [])),
                    'objects_detected': qwen_results.get('object_summary', {}).get('total_objects_detected', 0),
                    'motion_events': len(qwen_results.get('motion_summary', {}).get('motion_events', [])),
                    'analysis_quality': 'hybrid_high_accuracy'
                }
            }
            
            return combined
            
        except Exception as e:
            print(f"⚠️ Result combination failed: {e}")
            return {'error': str(e)}
    
    async def _store_in_vector_database(self, session_id: str, combined_results: Dict):
        """Store analysis results in vector database for search"""
        try:
            # Store in vector database
            success = self.vector_service.create_embeddings(session_id, combined_results)
            
            if success:
                print(f"✅ Results stored in vector database: {session_id}")
            else:
                print(f"⚠️ Warning: Vector database storage failed for session: {session_id}")
                
        except Exception as e:
            print(f"⚠️ Vector database storage failed: {e}")
    
    def _generate_session_id(self, video_path: str) -> str:
        """Generate unique session ID for video analysis"""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        return f"hybrid_{timestamp}_{video_hash}"
    
    async def search_analysis_results(self, session_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """Search analysis results using vector search"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Search in vector database
            search_results = self.vector_service.search_similar_content(session_id, query, top_k)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    async def get_analysis_summary(self, session_id: str) -> Dict:
        """Get summary of analysis results"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get summary from vector service
            summary = self.vector_service.get_session_summary(session_id)
            
            return summary
            
        except Exception as e:
            print(f"❌ Summary retrieval failed: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up hybrid analysis service resources"""
        try:
            print("🧹 Cleaning up Hybrid Analysis Service...")
            
            # Clean up DeepStream pipeline
            await self.deepstream_pipeline.cleanup()
            
            # Clean up GPU service
            await self.gpu_service.cleanup()
            
            print("✅ Hybrid Analysis Service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Hybrid Analysis Service cleanup failed: {e}")

    async def _fallback_video_processing(self, video_path: str) -> Dict:
        """Fallback video processing when DeepStream is not available"""
        try:
            print("🔄 Using fallback video processing...")
            
            # Basic video info extraction
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get basic video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Create basic frame data structure
            frames_data = []
            sample_interval = max(1, frame_count // 100)  # Sample 100 frames
            
            for i in range(0, frame_count, sample_interval):
                frames_data.append({
                    'frame_idx': i,
                    'timestamp': i / fps if fps > 0 else 0,
                    'objects': [],  # No object detection in fallback
                    'motion_detection': {'motion_detected': False, 'motion_intensity': 0.0},
                    'scene_analysis': {'scene_type': 'unknown'}
                })
            
            return {
                'video_info': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration_seconds': duration_seconds,
                    'duration_minutes': duration_seconds / 60,
                    'resolution': f"{width}x{height}"
                },
                'frames_data': frames_data,
                'processing_metrics': {
                    'processing_time_seconds': 0.1,
                    'actual_fps': len(frames_data),
                    'target_fps': 30,
                    'frames_processed': len(frames_data)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Fallback video processing failed: {e}")
            return {
                'video_info': {},
                'frames_data': [],
                'processing_metrics': {},
                'error': f"Fallback processing failed: {e}"
            }

# Global instance for easy access
hybrid_analysis_service = HybridAnalysisService() 