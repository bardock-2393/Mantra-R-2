"""
Video Chunking Service for Long Video Processing
Handles splitting 120+ minute videos into 5-10 minute chunks for efficient GPU processing
"""

import os
import cv2
import torch
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import decord
from decord import VideoReader
import gc

class VideoChunkingService:
    """Service for processing long videos by splitting into chunks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_duration = config.get('chunk_duration', 300)  # 5 minutes default
        self.max_workers = config.get('max_workers', 2)  # GPU memory constraints
        self.frame_rate = config.get('frame_rate', 1)  # 1 fps for analysis
        self.resolution = config.get('resolution', (720, 480))  # 720p default
        self.use_decord = config.get('use_decord', True)
        self.memory_cleanup = config.get('memory_cleanup', True)
        
        # Initialize decord
        if self.use_decord:
            decord.bridge.set_bridge('torch')
    
    def split_video_into_chunks(self, video_path: str) -> List[Dict]:
        """Split video into chunks with metadata"""
        try:
            if self.use_decord:
                return self._split_with_decord(video_path)
            else:
                return self._split_with_opencv(video_path)
        except Exception as e:
            print(f"❌ Error splitting video: {e}")
            return []
    
    def _split_with_decord(self, video_path: str) -> List[Dict]:
        """Split video using decord for better performance"""
        try:
            # Load video with decord
            vr = VideoReader(video_path)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = total_frames / fps
            
            print(f"📹 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
            
            # Calculate chunk parameters
            frames_per_chunk = int(self.chunk_duration * fps)
            total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
            
            chunks = []
            for i in range(total_chunks):
                start_frame = i * frames_per_chunk
                end_frame = min((i + 1) * frames_per_chunk, total_frames)
                
                chunk_info = {
                    'chunk_id': i + 1,
                    'total_chunks': total_chunks,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'duration': (end_frame - start_frame) / fps,
                    'frame_count': end_frame - start_frame,
                    'video_path': video_path
                }
                chunks.append(chunk_info)
            
            print(f"✂️ Split video into {len(chunks)} chunks of ~{self.chunk_duration}s each")
            return chunks
            
        except Exception as e:
            print(f"❌ Decord splitting failed: {e}")
            return []
    
    def _split_with_opencv(self, video_path: str) -> List[Dict]:
        """Fallback splitting using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            print(f"📹 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
            
            # Calculate chunk parameters
            frames_per_chunk = int(self.chunk_duration * fps)
            total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
            
            chunks = []
            for i in range(total_chunks):
                start_frame = i * frames_per_chunk
                end_frame = min((i + 1) * frames_per_chunk, total_frames)
                
                chunk_info = {
                    'chunk_id': i + 1,
                    'total_chunks': total_chunks,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'duration': (end_frame - start_frame) / fps,
                    'frame_count': end_frame - start_frame,
                    'video_path': video_path
                }
                chunks.append(chunk_info)
            
            cap.release()
            print(f"✂️ Split video into {len(chunks)} chunks of ~{self.chunk_duration}s each")
            return chunks
            
        except Exception as e:
            print(f"❌ OpenCV splitting failed: {e}")
            return []
    
    def extract_frames_from_chunk(self, chunk_info: Dict, max_frames: int = 4) -> List[torch.Tensor]:
        """Extract key frames from a specific chunk"""
        try:
            if self.use_decord:
                return self._extract_frames_decord(chunk_info, max_frames)
            else:
                return self._extract_frames_opencv(chunk_info, max_frames)
        except Exception as e:
            print(f"❌ Frame extraction failed for chunk {chunk_info['chunk_id']}: {e}")
            return []
    
    def _extract_frames_decord(self, chunk_info: Dict, max_frames: int) -> List[torch.Tensor]:
        """Extract frames using decord for better performance"""
        try:
            vr = VideoReader(chunk_info['video_path'])
            
            # Calculate frame indices for this chunk
            start_frame = chunk_info['start_frame']
            end_frame = chunk_info['end_frame']
            frame_count = end_frame - start_frame
            
            # Select frames evenly distributed across the chunk
            if frame_count <= max_frames:
                frame_indices = list(range(start_frame, end_frame))
            else:
                step = frame_count / max_frames
                frame_indices = [start_frame + int(i * step) for i in range(max_frames)]
            
            # Extract frames
            frames = vr.get_batch(frame_indices)
            
            # Resize and normalize frames
            processed_frames = []
            for frame in frames:
                # Resize to target resolution
                if frame.shape[:2] != self.resolution:
                    frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0).permute(0, 3, 1, 2),
                        size=self.resolution,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                
                # Normalize to [0, 1]
                if frame.dtype == torch.uint8:
                    frame = frame.float() / 255.0
                
                processed_frames.append(frame)
            
            print(f"🎬 Extracted {len(processed_frames)} frames from chunk {chunk_info['chunk_id']}")
            return processed_frames
            
        except Exception as e:
            print(f"❌ Decord frame extraction failed: {e}")
            return []
    
    def _extract_frames_opencv(self, chunk_info: Dict, max_frames: int) -> List[torch.Tensor]:
        """Fallback frame extraction using OpenCV"""
        try:
            cap = cv2.VideoCapture(chunk_info['video_path'])
            
            # Calculate frame indices for this chunk
            start_frame = chunk_info['start_frame']
            end_frame = chunk_info['end_frame']
            frame_count = end_frame - start_frame
            
            # Select frames evenly distributed across the chunk
            if frame_count <= max_frames:
                frame_indices = list(range(start_frame, end_frame))
            else:
                step = frame_count / max_frames
                frame_indices = [start_frame + int(i * step) for i in range(max_frames)]
            
            # Extract frames
            processed_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target resolution
                    if frame_rgb.shape[:2] != self.resolution:
                        frame_rgb = cv2.resize(frame_rgb, self.resolution)
                    
                    # Convert to tensor and normalize
                    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                    processed_frames.append(frame_tensor)
            
            cap.release()
            print(f"🎬 Extracted {len(processed_frames)} frames from chunk {chunk_info['chunk_id']}")
            return processed_frames
            
        except Exception as e:
            print(f"❌ OpenCV frame extraction failed: {e}")
            return []
    
    def process_chunks_parallel(self, chunks: List[Dict], 
                              process_func: Callable,
                              progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Process chunks in parallel with progress tracking"""
        results = []
        total_chunks = len(chunks)
        
        def process_single_chunk(chunk):
            try:
                # Process the chunk
                result = process_func(chunk)
                
                # Clean up GPU memory if enabled
                if self.memory_cleanup:
                    self._cleanup_gpu_memory()
                
                return {
                    'chunk_id': chunk['chunk_id'],
                    'success': True,
                    'result': result,
                    'error': None
                }
                
            except Exception as e:
                print(f"❌ Chunk {chunk['chunk_id']} processing failed: {e}")
                return {
                    'chunk_id': chunk['chunk_id'],
                    'success': False,
                    'result': None,
                    'error': str(e)
                }
        
        # Process chunks with limited parallelism
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                result = future.result()
                results.append(result)
                
                completed += 1
                if progress_callback:
                    progress = (completed / total_chunks) * 100
                    progress_callback(progress, completed, total_chunks, chunk['chunk_id'])
                
                print(f"✅ Chunk {chunk['chunk_id']}/{total_chunks} completed ({completed}/{total_chunks})")
        
        # Sort results by chunk ID
        results.sort(key=lambda x: x['chunk_id'])
        return results
    
    def merge_chunk_results(self, chunk_results: List[Dict]) -> Dict:
        """Merge results from all chunks into a unified analysis"""
        try:
            successful_results = [r for r in chunk_results if r['success']]
            failed_chunks = [r for r in chunk_results if not r['success']]
            
            if not successful_results:
                return {
                    'success': False,
                    'error': 'All chunks failed to process',
                    'failed_chunks': failed_chunks
                }
            
            # Merge successful results
            merged_result = {
                'success': True,
                'total_chunks': len(chunk_results),
                'successful_chunks': len(successful_results),
                'failed_chunks': len(failed_chunks),
                'chunk_results': chunk_results,
                'summary': self._generate_summary(chunk_results)
            }
            
            print(f"🔗 Merged {len(successful_results)} successful chunks")
            return merged_result
            
        except Exception as e:
            print(f"❌ Error merging chunk results: {e}")
            return {
                'success': False,
                'error': f'Merge failed: {str(e)}',
                'chunk_results': chunk_results
            }
    
    def _generate_summary(self, chunk_results: List[Dict]) -> str:
        """Generate a summary of all chunk results"""
        try:
            successful_results = [r for r in chunk_results if r['success']]
            
            if not successful_results:
                return "No successful chunk results to summarize"
            
            # Extract key information from each chunk
            summaries = []
            for result in successful_results:
                chunk_id = result['chunk_id']
                chunk_result = result['result']
                
                if isinstance(chunk_result, str):
                    summaries.append(f"Chunk {chunk_id}: {chunk_result[:100]}...")
                elif isinstance(chunk_result, dict):
                    # Extract key fields from dict results
                    key_info = []
                    for key in ['content', 'analysis', 'summary']:
                        if key in chunk_result:
                            value = chunk_result[key]
                            if isinstance(value, str):
                                key_info.append(f"{key}: {value[:50]}...")
                    summaries.append(f"Chunk {chunk_id}: {' | '.join(key_info)}")
                else:
                    summaries.append(f"Chunk {chunk_id}: {type(chunk_result).__name__}")
            
            return "\n".join(summaries)
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory between chunks"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                print("🧹 GPU memory cleaned up")
        except Exception as e:
            print(f"⚠️ GPU memory cleanup failed: {e}")
    
    def get_memory_usage(self) -> Dict:
        """Get current GPU memory usage"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                return {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'available_gb': round(80 - allocated, 2)  # Assuming 80GB GPU
                }
            else:
                return {'error': 'CUDA not available'}
        except Exception as e:
            return {'error': f'Memory check failed: {str(e)}'}
