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
            # This is a simplified summary - in a real implementation, you'd extract frames
            # and analyze them with the vision model
            return f"Video file: {os.path.basename(video_path)} - Ready for comprehensive analysis."
        except Exception as e:
            return f"Video file available for analysis. Error extracting details: {e}"
    
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
2. **Comprehensive Coverage**: Analyze every significant aspect of the video
3. **Detailed Descriptions**: Use vivid, descriptive language for visual elements
4. **Quantitative Data**: Include specific numbers, counts, and measurements
5. **Pattern Recognition**: Identify recurring themes, behaviors, and sequences
6. **Contextual Understanding**: Explain significance and relationships between elements
7. **Professional Structure**: Organize information logically with clear sections
8. **Evidence-Based**: Support all observations with specific visual evidence

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
- Include **specific timestamps** for all temporal references
- Provide **quantitative measurements** (durations, counts, sizes)
- Use **bullet points** for lists and multiple items
- Structure with **clear headings** for different analysis areas
- Include **cross-references** between related information
- Offer **insights and interpretations** beyond simple description

Your analysis will be used for **high-quality user interactions**, so ensure every detail is **precise, comprehensive, and well-structured** for optimal user experience.
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
