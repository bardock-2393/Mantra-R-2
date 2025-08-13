"""
Enhanced Video Processor Service
Improves video analysis accuracy through advanced processing techniques
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
from PIL import Image
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quality_threshold = 0.8
        self.max_frames = 150
        self.min_frames = 30
        
        # Enhanced processing parameters
        self.processing_config = {
            "frame_extraction": {
                "method": "adaptive",
                "quality_threshold": 0.8,
                "motion_detection": True,
                "content_analysis": True,
                "edge_detection": True
            },
            "image_enhancement": {
                "super_resolution": False,  # Disable for accuracy
                "noise_reduction": True,
                "contrast_enhancement": True,
                "sharpening": True
            },
            "multi_scale": {
                "enabled": True,
                "scales": [0.5, 1.0, 1.5],
                "weight_method": "confidence_weighted"
            }
        }
        
        # Initialize OpenCV optimizations
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

    def process_video_enhanced(self, video_path: str) -> Dict[str, Any]:
        """Enhanced video processing with multiple quality improvements"""
        try:
            logger.info(f"🎬 Starting enhanced video processing: {video_path}")
            
            # Validate video file
            if not self._validate_video_file(video_path):
                return {"success": False, "error": "Invalid video file"}
            
            # Extract enhanced frames
            frames = self._extract_enhanced_frames(video_path)
            if not frames:
                return {"success": False, "error": "No frames extracted"}
            
            # Process frames for quality
            processed_frames = self._process_frames_quality(frames)
            
            # Multi-scale analysis
            multi_scale_results = self._multi_scale_analysis(processed_frames)
            
            # Generate comprehensive analysis data
            analysis_data = self._generate_analysis_data(processed_frames, multi_scale_results)
            
            logger.info(f"✅ Enhanced video processing completed: {len(processed_frames)} frames")
            return {
                "success": True,
                "frames": processed_frames,
                "multi_scale_results": multi_scale_results,
                "analysis_data": analysis_data,
                "frame_count": len(processed_frames),
                "processing_method": "enhanced"
            }
            
        except Exception as e:
            logger.error(f"Enhanced video processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _validate_video_file(self, video_path: str) -> bool:
        """Validate video file integrity and format"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                logger.error(f"Video file is empty: {video_path}")
                return False
            
            # Check if it's a valid video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            # Check basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                logger.error(f"Invalid video properties: fps={fps}, frames={frame_count}, size={width}x{height}")
                return False
            
            logger.info(f"✅ Video validation passed: {width}x{height}, {fps}fps, {frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}")
            return False

    def _extract_enhanced_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract frames using enhanced algorithms for better quality"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Adaptive frame extraction
            extraction_rate = max(1, total_frames // self.max_frames)
            
            logger.info(f"📊 Video properties: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
            logger.info(f"🎯 Extraction rate: 1 frame every {extraction_rate} frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frames based on rate and quality
                if frame_count % extraction_rate == 0:
                    # Analyze frame quality
                    quality_score = self._assess_frame_quality(frame)
                    
                    if quality_score >= self.quality_threshold:
                        # Enhanced frame processing
                        enhanced_frame = self._enhance_frame(frame)
                        
                        frame_data = {
                            "frame_number": frame_count,
                            "timestamp": frame_count / fps if fps > 0 else 0,
                            "quality_score": quality_score,
                            "frame": enhanced_frame,
                            "original_frame": frame.copy(),
                            "motion_score": self._calculate_motion_score(frame, frames),
                            "content_score": self._calculate_content_score(frame),
                            "edge_density": self._calculate_edge_density(frame)
                        }
                        
                        frames.append(frame_data)
                        
                        if len(frames) >= self.max_frames:
                            break
                
                frame_count += 1
            
            cap.release()
            
            # Ensure minimum frame count
            if len(frames) < self.min_frames:
                logger.warning(f"⚠️ Only {len(frames)} frames extracted, below minimum {self.min_frames}")
                # Extract additional frames if needed
                frames = self._extract_additional_frames(video_path, frames)
            
            logger.info(f"✅ Extracted {len(frames)} high-quality frames")
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            return []

    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality using multiple metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalize
            
            # 2. Contrast (standard deviation)
            contrast_score = np.std(gray) / 128.0  # Normalize
            
            # 3. Brightness (mean intensity)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
            
            # 4. Noise level (using high-pass filter)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise_response = cv2.filter2D(gray, -1, kernel)
            noise_score = 1.0 - min(np.std(noise_response) / 50.0, 1.0)
            
            # Weighted quality score
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.3 +
                brightness_score * 0.2 +
                noise_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Frame quality assessment failed: {str(e)}")
            return 0.5  # Default score

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality for better analysis"""
        try:
            enhanced = frame.copy()
            
            # 1. Noise reduction
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # 2. Contrast enhancement using CLAHE
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. Sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. Color balance
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Frame enhancement failed: {str(e)}")
            return frame

    def _calculate_motion_score(self, frame: np.ndarray, previous_frames: List[Dict]) -> float:
        """Calculate motion score between current and previous frames"""
        try:
            if not previous_frames:
                return 0.0
            
            # Get the last frame
            last_frame = previous_frames[-1]["original_frame"]
            
            # Convert to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_last, gray_current, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = np.mean(magnitude) / 10.0  # Normalize
            
            return min(motion_score, 1.0)
            
        except Exception as e:
            logger.error(f"Motion score calculation failed: {str(e)}")
            return 0.0

    def _calculate_content_score(self, frame: np.ndarray) -> float:
        """Calculate content richness score"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture complexity
            texture_score = np.std(gray) / 128.0
            
            # Color diversity
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_diversity = np.std(hsv[:, :, 1]) / 128.0  # Saturation diversity
            
            # Combined content score
            content_score = (
                edge_density * 0.4 +
                texture_score * 0.3 +
                color_diversity * 0.3
            )
            
            return min(content_score, 1.0)
            
        except Exception as e:
            logger.error(f"Content score calculation failed: {str(e)}")
            return 0.5

    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density for frame complexity"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            return edge_density
        except Exception as e:
            logger.error(f"Edge density calculation failed: {str(e)}")
            return 0.0

    def _extract_additional_frames(self, video_path: str, existing_frames: List[Dict]) -> List[Dict]:
        """Extract additional frames if minimum count not met"""
        try:
            if len(existing_frames) >= self.min_frames:
                return existing_frames
            
            cap = cv2.VideoCapture(video_path)
            additional_frames = []
            frame_count = 0
            
            # Extract frames with lower quality threshold
            lower_threshold = self.quality_threshold * 0.7
            
            while len(existing_frames) + len(additional_frames) < self.min_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames already extracted
                if any(f["frame_number"] == frame_count for f in existing_frames):
                    frame_count += 1
                    continue
                
                quality_score = self._assess_frame_quality(frame)
                if quality_score >= lower_threshold:
                    enhanced_frame = self._enhance_frame(frame)
                    
                    frame_data = {
                        "frame_number": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                        "quality_score": quality_score,
                        "frame": enhanced_frame,
                        "original_frame": frame.copy(),
                        "motion_score": 0.0,
                        "content_score": self._calculate_content_score(frame),
                        "edge_density": self._calculate_edge_density(frame)
                    }
                    
                    additional_frames.append(frame_data)
                
                frame_count += 1
            
            cap.release()
            
            # Combine frames and sort by frame number
            all_frames = existing_frames + additional_frames
            all_frames.sort(key=lambda x: x["frame_number"])
            
            logger.info(f"✅ Added {len(additional_frames)} additional frames, total: {len(all_frames)}")
            return all_frames
            
        except Exception as e:
            logger.error(f"Additional frame extraction failed: {str(e)}")
            return existing_frames

    def _process_frames_quality(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process frames for quality enhancement"""
        try:
            processed_frames = []
            
            for frame_data in frames:
                # Apply additional quality enhancements
                enhanced_frame = self._apply_quality_enhancements(frame_data["frame"])
                
                # Update frame data
                processed_frame = frame_data.copy()
                processed_frame["frame"] = enhanced_frame
                processed_frame["final_quality_score"] = self._assess_frame_quality(enhanced_frame)
                
                processed_frames.append(processed_frame)
            
            # Sort by final quality score
            processed_frames.sort(key=lambda x: x["final_quality_score"], reverse=True)
            
            logger.info(f"✅ Quality processing completed for {len(processed_frames)} frames")
            return processed_frames
            
        except Exception as e:
            logger.error(f"Frame quality processing failed: {str(e)}")
            return frames

    def _apply_quality_enhancements(self, frame: np.ndarray) -> np.ndarray:
        """Apply additional quality enhancements"""
        try:
            enhanced = frame.copy()
            
            # Adaptive histogram equalization for better contrast
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Bilateral filtering for edge preservation
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Unsharp masking for sharpening
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Quality enhancement failed: {str(e)}")
            return frame

    def _multi_scale_analysis(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform multi-scale analysis for better object detection"""
        try:
            if not self.processing_config["multi_scale"]["enabled"]:
                return {"enabled": False}
            
            multi_scale_results = {
                "enabled": True,
                "scales": self.processing_config["multi_scale"]["scales"],
                "results": {}
            }
            
            for scale in self.processing_config["multi_scale"]["scales"]:
                scale_results = []
                
                for frame_data in frames:
                    # Resize frame to scale
                    frame = frame_data["frame"]
                    height, width = frame.shape[:2]
                    new_height, new_width = int(height * scale), int(width * scale)
                    
                    scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Analyze scaled frame
                    scale_analysis = self._analyze_frame_at_scale(scaled_frame, scale)
                    scale_analysis["frame_number"] = frame_data["frame_number"]
                    scale_analysis["scale"] = scale
                    
                    scale_results.append(scale_analysis)
                
                multi_scale_results["results"][scale] = scale_results
            
            logger.info(f"✅ Multi-scale analysis completed for {len(frames)} frames")
            return multi_scale_results
            
        except Exception as e:
            logger.error(f"Multi-scale analysis failed: {str(e)}")
            return {"enabled": False, "error": str(e)}

    def _analyze_frame_at_scale(self, frame: np.ndarray, scale: float) -> Dict[str, Any]:
        """Analyze frame at specific scale"""
        try:
            analysis = {
                "scale": scale,
                "size": frame.shape[:2],
                "edge_density": self._calculate_edge_density(frame),
                "content_score": self._calculate_content_score(frame),
                "quality_score": self._assess_frame_quality(frame)
            }
            
            # Add scale-specific analysis
            if scale < 1.0:
                analysis["detail_level"] = "low"
                analysis["focus"] = "overview"
            elif scale > 1.0:
                analysis["detail_level"] = "high"
                analysis["focus"] = "details"
            else:
                analysis["detail_level"] = "medium"
                analysis["focus"] = "balanced"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Scale analysis failed: {str(e)}")
            return {"scale": scale, "error": str(e)}

    def _generate_analysis_data(self, frames: List[Dict[str, Any]], multi_scale_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis data"""
        try:
            analysis_data = {
                "frame_summary": {
                    "total_frames": len(frames),
                    "high_quality_frames": len([f for f in frames if f["final_quality_score"] >= 0.9]),
                    "medium_quality_frames": len([f for f in frames if 0.7 <= f["final_quality_score"] < 0.9]),
                    "low_quality_frames": len([f for f in frames if f["final_quality_score"] < 0.7])
                },
                "quality_metrics": {
                    "average_quality": np.mean([f["final_quality_score"] for f in frames]),
                    "quality_std": np.std([f["final_quality_score"] for f in frames]),
                    "best_frame": max(frames, key=lambda x: x["final_quality_score"]),
                    "worst_frame": min(frames, key=lambda x: x["final_quality_score"])
                },
                "motion_analysis": {
                    "high_motion_frames": len([f for f in frames if f["motion_score"] >= 0.7]),
                    "average_motion": np.mean([f["motion_score"] for f in frames]),
                    "motion_pattern": self._analyze_motion_pattern(frames)
                },
                "content_analysis": {
                    "rich_content_frames": len([f for f in frames if f["content_score"] >= 0.7]),
                    "average_content": np.mean([f["content_score"] for f in frames]),
                    "content_distribution": self._analyze_content_distribution(frames)
                },
                "multi_scale_summary": multi_scale_results,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_method": "enhanced"
            }
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Analysis data generation failed: {str(e)}")
            return {"error": str(e)}

    def _analyze_motion_pattern(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze motion patterns across frames"""
        try:
            motion_scores = [f["motion_score"] for f in frames]
            
            return {
                "pattern": "continuous" if np.std(motion_scores) < 0.2 else "variable",
                "peak_motion": max(motion_scores),
                "motion_variance": np.var(motion_scores),
                "motion_trend": "increasing" if motion_scores[-1] > motion_scores[0] else "decreasing"
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_content_distribution(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content distribution across frames"""
        try:
            content_scores = [f["content_score"] for f in frames]
            
            return {
                "distribution": "uniform" if np.std(content_scores) < 0.2 else "varied",
                "content_peaks": len([i for i in range(1, len(content_scores)-1) if content_scores[i] > content_scores[i-1] and content_scores[i] > content_scores[i+1]]),
                "content_valleys": len([i for i in range(1, len(content_scores)-1) if content_scores[i] < content_scores[i-1] and content_scores[i] < content_scores[i+1]])
            }
        except Exception as e:
            return {"error": str(e)}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics"""
        return {
            "service": "Enhanced Video Processor",
            "version": "1.0.0",
            "capabilities": [
                "Adaptive frame extraction",
                "Quality assessment",
                "Frame enhancement",
                "Multi-scale analysis",
                "Motion detection",
                "Content analysis"
            ],
            "config": self.processing_config,
            "performance": {
                "max_frames": self.max_frames,
                "min_frames": self.min_frames,
                "quality_threshold": self.quality_threshold
            }
        }
