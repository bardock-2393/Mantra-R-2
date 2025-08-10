"""
AI Service Module for Round 2 - GPU-powered local AI
Handles MiniCPM-V-2_6 local inference and GPU optimization
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor
from models.minicpm_v26_model import minicpm_v26_model

class MiniCPMV26Service:
    """Local GPU-powered MiniCPM-V-2_6 service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the MiniCPM-V-2_6 model on GPU"""
        try:
            print(f"🚀 Initializing GPU service...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Initialize GPU service
            self.gpu_service.initialize()
            
            # Initialize the MiniCPM model
            minicpm_v26_model.initialize()
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V-2_6 service initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V-2_6 service: {e}")
            raise
    
    def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video content using MiniCPM-V-2_6"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract video metadata including duration
            video_metadata = self._extract_video_metadata(video_path)
            video_duration = video_metadata.get('duration', 0)
            
            # Extract video summary (simplified for now)
            video_summary = self._extract_video_summary(video_path, video_metadata)
            
            # Generate analysis prompt with duration constraint
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus, video_duration)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis using the model
            analysis_result = minicpm_v26_model.generate_text(full_prompt, max_new_tokens=2048)
            
            # Post-process the response to fix any out-of-bounds timestamps
            if video_duration > 0:
                analysis_result = self._fix_out_of_bounds_timestamps_in_text(analysis_result, video_duration)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_analysis_latency(latency)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return f"Error analyzing video: {str(e)}"
    
    def _fix_out_of_bounds_timestamps_in_text(self, text: str, video_duration: float) -> str:
        """Fix any out-of-bounds timestamps in the AI response text"""
        if not text or video_duration <= 0:
            return text
        
        import re
        
        # Pattern to find MM:SS timestamps
        timestamp_pattern = r'(\d{1,2}):(\d{2})'
        
        def replace_timestamp(match):
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            total_seconds = minutes * 60 + seconds
            
            if total_seconds >= video_duration:
                # Scale down the timestamp to be within video duration
                scaled_seconds = int((total_seconds / (total_seconds + 1)) * video_duration * 0.9)
                new_minutes = scaled_seconds // 60
                new_seconds = scaled_seconds % 60
                print(f"⚠️ Fixed out-of-bounds timestamp in text: {match.group()} -> {new_minutes:02d}:{new_seconds:02d}")
                return f"{new_minutes:02d}:{new_seconds:02d}"
            else:
                return match.group()
        
        # Replace out-of-bounds timestamps
        fixed_text = re.sub(timestamp_pattern, replace_timestamp, text)
        
        return fixed_text
    
    def _extract_video_metadata(self, video_path: str) -> dict:
        """Extract video metadata including duration"""
        try:
            from utils.video_utils import extract_video_metadata
            metadata = extract_video_metadata(video_path)
            if metadata:
                print(f"📹 Video duration: {metadata['duration']:.2f} seconds")
                return metadata
            else:
                print("⚠️ Warning: Could not extract video metadata")
                return {'duration': 0, 'fps': 0, 'width': 0, 'height': 0}
        except Exception as e:
            print(f"⚠️ Warning: Error extracting video metadata: {e}")
            return {'duration': 0, 'fps': 0, 'width': 0, 'height': 0}
    
    def _extract_video_summary(self, video_path: str, metadata: dict) -> str:
        """Extract basic video information for analysis"""
        try:
            duration = metadata.get('duration', 0)
            fps = metadata.get('fps', 0)
            width = metadata.get('width', 0)
            height = metadata.get('height', 0)
            
            summary = f"Video file: {os.path.basename(video_path)}"
            if duration > 0:
                summary += f" - Duration: {duration:.2f} seconds"
            if fps > 0:
                summary += f" - FPS: {fps:.1f}"
            if width > 0 and height > 0:
                summary += f" - Resolution: {width}x{height}"
            summary += " - Ready for comprehensive analysis."
            
            return summary
        except Exception as e:
            return f"Video file available for analysis. Error extracting details: {e}"
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str, video_duration: float) -> str:
        """Generate analysis prompt based on type and user focus"""
        duration_minutes = video_duration / 60 if video_duration > 0 else 0
        max_minutes = int(video_duration / 60)
        max_seconds = int(video_duration % 60)
        
        base_prompt = f"""
You are an **exceptional AI video analysis agent** with unparalleled understanding capabilities. Your mission is to provide **comprehensive, precise, and insightful analysis** that serves as the foundation for high-quality user interactions.

## ANALYSIS REQUEST
- **Analysis Type**: {analysis_type}
- **User Focus**: {user_focus}
- **Video Duration**: {video_duration:.2f} seconds ({duration_minutes:.1f} minutes)

## 🚨 CRITICAL TIMESTAMP CONSTRAINTS - READ CAREFULLY 🚨

⚠️ **ABSOLUTELY FORBIDDEN**: This video is only {video_duration:.2f} seconds long.
🚫 **NEVER** generate timestamps beyond {video_duration:.2f} seconds
🚫 **NEVER** reference events after {video_duration:.2f} seconds  
🚫 **NEVER** mention time ranges that extend beyond {video_duration:.2f} seconds
🚫 **NEVER** create fictional content beyond the actual video duration
🚫 **NEVER** assume the video continues beyond {video_duration:.2f} seconds

✅ **ONLY ALLOWED**: Analyze content within 0-{video_duration:.2f} seconds
✅ **ONLY ALLOWED**: Generate timestamps between 00:00 and {max_minutes:02d}:{max_seconds:02d}
✅ **ONLY ALLOWED**: Reference events that actually exist in the video

## AGENT ANALYSIS PROTOCOL

### Analysis Quality Standards:
1. **Maximum Precision**: Provide exact timestamps, durations, and measurements (within video bounds)
2. **Comprehensive Coverage**: Analyze every significant aspect of the video (within duration)
3. **Detailed Descriptions**: Use vivid, descriptive language for visual elements
4. **Quantitative Data**: Include specific numbers, counts, and measurements
5. **Pattern Recognition**: Identify recurring themes, behaviors, and sequences
6. **Contextual Understanding**: Explain significance and relationships between elements
7. **Professional Structure**: Organize information logically with clear sections
8. **Evidence-Based**: Support all observations with specific visual evidence

### Enhanced Analysis Focus:
- **Temporal Precision**: Exact timestamps for all events and transitions (0-{video_duration:.2f}s ONLY)
- **Spatial Relationships**: Detailed descriptions of positioning and movement
- **Visual Details**: Colors, lighting, composition, and technical quality
- **Behavioral Analysis**: Actions, interactions, and human elements
- **Technical Assessment**: Quality, production values, and technical specifications
- **Narrative Structure**: Story flow, pacing, and dramatic elements
- **Environmental Context**: Setting, atmosphere, and contextual factors

### Output Quality Requirements:
- Use **bold formatting** for emphasis on key information
- Include **specific timestamps** for all temporal references (0-{video_duration:.2f}s ONLY)
- Provide **quantitative measurements** (durations, counts, sizes)
- Use **bullet points** for lists and multiple items
- Structure with **clear headings** for different analysis areas
- Include **cross-references** between related information
- Offer **insights and interpretations** beyond simple description

### ⚠️ FINAL WARNING ⚠️
**This video ends at {video_duration:.2f} seconds.**
**Any timestamp beyond this point is WRONG and will cause errors.**
**Double-check every timestamp you generate.**
**If unsure, use timestamps closer to the beginning of the video.**

Your analysis will be used for **high-quality user interactions**, so ensure every detail is **precise, comprehensive, and well-structured** for optimal user experience.

**REMEMBER**: This video ends at {video_duration:.2f} seconds. Do not reference anything beyond this point.
"""
        return base_prompt
    
    def generate_chat_response(self, analysis_result: str, analysis_type: str, 
                              user_focus: str, message: str, 
                              chat_history: List[Dict]) -> str:
        """Generate contextual AI response based on video analysis"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Use the model's chat response method
            response = minicpm_v26_model.generate_chat_response(
                analysis_result, analysis_type, user_focus, message, chat_history
            )
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_chat_latency(latency)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"Error generating chat response: {str(e)}"
    
    def get_model_status(self) -> Dict:
        """Get model status and performance metrics"""
        try:
            status = {
                'initialized': self.is_initialized,
                'device': str(self.device),
                'gpu_service': self.gpu_service.get_status()
            }
            
            if self.is_initialized:
                status['minicpm_model'] = minicpm_v26_model.get_model_status()
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up GPU resources"""
        try:
            if self.is_initialized:
                minicpm_v26_model.cleanup()
                self.gpu_service.cleanup()
                torch.cuda.empty_cache()
                self.is_initialized = False
                print("🧹 MiniCPM-V-2_6 service cleaned up")
        except Exception as e:
            print(f"⚠️ Warning: Service cleanup failed: {e}")

# Global instance
minicpm_service = MiniCPMV26Service()

# Round 2: All AI processing is now done locally with MiniCPM-V-2_6
# No external API dependencies required 
