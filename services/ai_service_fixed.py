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
            # Extract video summary (simplified for now)
            video_summary = self._extract_video_summary(video_path)
            
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis using the model
            analysis_result = minicpm_v26_model.generate_text(full_prompt, max_new_tokens=2048)
            
            # Clean up the response to remove out-of-bounds timestamps
            analysis_result = self._clean_timestamps_in_response(analysis_result, video_path)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_analysis_latency(latency)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return f"Error analyzing video: {str(e)}"
    
    def _extract_video_summary(self, video_path: str) -> str:
        """Extract basic video information for analysis"""
        try:
            # Import video_utils here to avoid circular imports
            from utils.video_utils import extract_video_metadata
            
            # Extract actual video metadata including duration
            metadata = extract_video_metadata(video_path)
            print(f"Debug: AI Service - Video metadata: {metadata}")
            
            if metadata and metadata.get('duration'):
                duration_seconds = metadata['duration']
                duration_minutes = duration_seconds / 60
                
                summary = f"""Video file: {os.path.basename(video_path)}
Video Duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)
Frame Rate: {metadata.get('fps', 'Unknown')} fps
Resolution: {metadata.get('width', 'Unknown')}x{metadata.get('height', 'Unknown')}
Total Frames: {metadata.get('frame_count', 'Unknown')}

IMPORTANT: This video is exactly {duration_seconds:.2f} seconds long. All timestamps in your analysis must be between 0.00 and {duration_seconds:.2f} seconds. Do NOT provide timestamps beyond the video duration."""
                
                print(f"Debug: AI Service - Generated summary with duration: {duration_seconds:.2f}s")
            else:
                summary = f"Video file: {os.path.basename(video_path)} - Duration information unavailable. Please provide timestamps within reasonable video length bounds."
                print(f"Debug: AI Service - No duration found in metadata")
            
            return summary
            
        except Exception as e:
            return f"Video file: {os.path.basename(video_path)} - Error extracting metadata: {e}. Please provide timestamps within reasonable video length bounds."
    
    def _debug_video_metadata(self, video_path: str) -> str:
        """Debug method to check video metadata extraction"""
        try:
            from utils.video_utils import extract_video_metadata
            
            metadata = extract_video_metadata(video_path)
            if metadata:
                return f"Video metadata: {metadata}"
            else:
                return "No metadata extracted"
        except Exception as e:
            return f"Error extracting metadata: {e}"
    
    def _clean_timestamps_in_response(self, response: str, video_path: str) -> str:
        """Clean up response to remove out-of-bounds timestamps"""
        try:
            from utils.video_utils import extract_video_metadata
            
            metadata = extract_video_metadata(video_path)
            if not metadata or not metadata.get('duration'):
                return response
            
            video_duration = metadata['duration']
            
            # Pattern to find timestamp ranges like "05:30-06:00"
            import re
            timestamp_range_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?'
            
            def replace_invalid_range(match):
                # Parse start time
                start_hours = int(match.group(3)) if match.group(3) else 0
                start_minutes = int(match.group(1))
                start_seconds = int(match.group(2))
                start_time = start_hours * 3600 + start_minutes * 60 + start_seconds
                
                # Parse end time
                end_hours = int(match.group(6)) if match.group(6) else 0
                end_minutes = int(match.group(4))
                end_seconds = int(match.group(5))
                end_time = end_hours * 3600 + end_minutes * 60 + end_seconds
                
                # If end time exceeds video duration, cap it
                if end_time > video_duration:
                    end_time = video_duration
                    end_minutes = int(end_time // 60)
                    end_seconds = int(end_time % 60)
                    return f"{end_minutes:02d}:{end_seconds:02d}"
                
                return match.group(0)
            
            # Replace invalid timestamp ranges
            cleaned_response = re.sub(timestamp_range_pattern, replace_invalid_range, response)
            
            # Also clean up single timestamps that exceed duration
            single_timestamp_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
            
            def replace_invalid_timestamp(match):
                hours = int(match.group(3)) if match.group(3) else 0
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                timestamp = hours * 3600 + minutes * 60 + seconds
                
                if timestamp > video_duration:
                    # Cap at video duration
                    capped_time = video_duration
                    capped_minutes = int(capped_time // 60)
                    capped_seconds = int(capped_time % 60)
                    return f"{capped_minutes:02d}:{capped_seconds:02d}"
                
                return match.group(0)
            
            cleaned_response = re.sub(single_timestamp_pattern, replace_invalid_timestamp, cleaned_response)
            
            return cleaned_response
            
        except Exception as e:
            print(f"Warning: Could not clean timestamps in response: {e}")
            return response
    
    def _clean_timestamps_in_chat_response(self, response: str, video_duration: float = None) -> str:
        """Clean up chat response to remove obviously invalid timestamps"""
        try:
            import re
            
            # Pattern to find timestamp ranges like "05:30-06:00"
            timestamp_range_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?'
            
            def validate_and_fix_range(match):
                # Parse start time
                start_hours = int(match.group(3)) if match.group(3) else 0
                start_minutes = int(match.group(1))
                start_seconds = int(match.group(2))
                start_time = start_hours * 3600 + start_minutes * 60 + start_seconds
                
                # Parse end time
                end_hours = int(match.group(6)) if match.group(6) else 0
                end_minutes = int(match.group(4))
                end_seconds = int(match.group(5))
                end_time = end_hours * 3600 + end_minutes * 60 + end_seconds
                
                # If we have video duration, validate against it
                if video_duration and end_time > video_duration:
                    end_time = video_duration
                    end_minutes = int(end_time // 60)
                    end_seconds = int(end_time % 60)
                    return f"{start_minutes:02d}:{start_seconds:02d}-{end_minutes:02d}:{end_seconds:02d}"
                
                # Basic validation - if end time is much larger than start time, cap it
                if end_time > start_time + 300:  # More than 5 minutes difference
                    end_time = start_time + 60  # Cap at 1 minute
                    end_minutes = int(end_time // 60)
                    end_seconds = int(end_time % 60)
                    return f"{start_minutes:02d}:{start_seconds:02d}-{end_minutes:02d}:{end_seconds:02d}"
                
                return match.group(0)
            
            # Replace invalid timestamp ranges
            cleaned_response = re.sub(timestamp_range_pattern, validate_and_fix_range, response)
            
            # Also clean up single timestamps that exceed duration
            if video_duration:
                single_timestamp_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?'
                
                def replace_invalid_timestamp(match):
                    hours = int(match.group(3)) if match.group(3) else 0
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    timestamp = hours * 3600 + minutes * 60 + seconds
                    
                    if timestamp > video_duration:
                        # Cap at video duration
                        capped_time = video_duration
                        capped_minutes = int(capped_time // 60)
                        capped_seconds = int(capped_time % 60)
                        return f"{capped_minutes:02d}:{capped_seconds:02d}"
                    
                    return match.group(0)
                
                cleaned_response = re.sub(single_timestamp_pattern, replace_invalid_timestamp, cleaned_response)
            
            return cleaned_response
            
        except Exception as e:
            print(f"Warning: Could not clean timestamps in chat response: {e}")
            return response
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt based on type and user focus"""
        base_prompt = f"""
You are an **exceptional AI video analysis agent** with unparalleled understanding capabilities. Your mission is to provide **comprehensive, precise, and insightful analysis** that serves as the foundation for high-quality user interactions.

## ANALYSIS REQUEST
- **Analysis Type**: {analysis_type}
- **User Focus**: {user_focus}

## AGENT ANALYSIS PROTOCOL

### Analysis Quality Standards:
1. **Maximum Precision**: Provide exact timestamps, durations, and measurements
2. **Duration Compliance**: ALL timestamps must be within the actual video duration - never exceed video length
3. **Comprehensive Coverage**: Analyze every significant aspect of the video within its actual duration
4. **Detailed Descriptions**: Use vivid, descriptive language for visual elements
5. **Quantitative Data**: Include specific numbers, counts, and measurements
6. **Pattern Recognition**: Identify recurring themes, behaviors, and sequences
7. **Contextual Understanding**: Explain significance and relationships between elements
8. **Professional Structure**: Organize information logically with clear sections
9. **Evidence-Based**: Support all observations with specific visual evidence

### Enhanced Analysis Focus:
- **Temporal Precision**: Exact timestamps for all events and transitions
- **Spatial Relationships**: Detailed descriptions of positioning and movement
- **Visual Details**: Colors, lighting, composition, and technical quality
- **Behavioral Analysis**: Actions, interactions, and human elements
- **Technical Assessment**: Quality, production values, and technical specifications
- **Narrative Structure**: Story flow, pacing, and dramatic elements
- **Environmental Context**: Setting, atmosphere, and contextual factors

### Output Quality Requirements:
- Use **bold formatting** for emphasis on key information
- Include **specific timestamps** for all temporal references (must be within video duration)
- Provide **quantitative measurements** (durations, counts, sizes)
- Use **bullet points** for lists and multiple items
- Structure with **clear headings** for different analysis areas
- Include **cross-references** between related information
- Offer **insights and interpretations** beyond simple description

### CRITICAL TIMESTAMP RULES:
- **NEVER** provide timestamps beyond the video's actual duration
- **ALWAYS** validate that start_time < end_time < video_duration
- **ONLY** reference time ranges that exist within the video
- If unsure about duration, use conservative time estimates

Your analysis will be used for **high-quality user interactions**, so ensure every detail is **precise, comprehensive, and well-structured** for optimal user experience.
"""
        return base_prompt
    
    def generate_chat_response(self, analysis_result: str, analysis_type: str, 
                              user_focus: str, message: str, 
                              chat_history: List[Dict], video_duration: float = None) -> str:
        """Generate contextual AI response based on video analysis"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Use the model's chat response method
            response = minicpm_v26_model.generate_chat_response(
                analysis_result, analysis_type, user_focus, message, chat_history
            )
            
            # Clean up the response to remove out-of-bounds timestamps
            response = self._clean_timestamps_in_chat_response(response, video_duration)
            
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
