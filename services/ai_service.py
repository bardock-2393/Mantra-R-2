"""
AI Service Module for Round 2 - GPU-powered local AI
Handles MiniCPM-V 2.6 local inference and GPU optimization
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

class MiniCPMV26Service:
    """Local GPU-powered MiniCPM-V 2.6 service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the MiniCPM-V 2.6 model on GPU"""
        try:
            print(f"🚀 Initializing MiniCPM-V 2.6 on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Initialize GPU service
            self.gpu_service.initialize()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MINICPM_MODEL_PATH,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MINICPM_MODEL_PATH,
                torch_dtype=torch.float16 if Config.GPU_CONFIG['precision'] == 'float16' else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                use_flash_attention_2=Config.MINICPM_CONFIG['use_flash_attention'],
                load_in_8bit=Config.MINICPM_CONFIG['quantization'] == 'int8'
            )
            
            # Move to GPU
            self.model.to(self.device)
            self.model.eval()
            
            # Warm up the model
            self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V 2.6 initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V 2.6: {e}")
            raise
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V 2.6 model...")
        
        # Create dummy input for warmup
        dummy_text = "Hello, this is a warmup message for the AI model."
        inputs = self.tokenizer(dummy_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            for _ in range(3):  # Run 3 warmup iterations
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
        
        print("✅ Model warmup completed")
    
    def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using local GPU-powered MiniCPM-V 2.6"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Process video frames (simplified for now - will be enhanced with DeepStream)
            video_summary = self._extract_video_summary(video_path)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis using MiniCPM-V 2.6
            analysis_result = self._generate_analysis(full_prompt)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.performance_monitor.record_analysis_latency(latency)
            
            if latency > Config.PERFORMANCE_TARGETS['latency_target']:
                print(f"⚠️ Warning: Analysis latency ({latency:.2f}ms) exceeds target ({Config.PERFORMANCE_TARGETS['latency_target']}ms)")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return f"Error analyzing video: {str(e)}"
    
    def _extract_video_summary(self, video_path: str) -> str:
        """Extract key information from video for analysis"""
        # This will be enhanced with DeepStream integration
        # For now, return a basic summary
        return f"Video file: {os.path.basename(video_path)}\nDuration: [To be extracted]\nResolution: [To be extracted]"
    
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
    
    def _generate_analysis(self, prompt: str) -> str:
        """Generate analysis using MiniCPM-V 2.6"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=Config.MINICPM_CONFIG['max_length']).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.MINICPM_CONFIG['max_length'],
                    temperature=Config.MINICPM_CONFIG['temperature'],
                    top_p=Config.MINICPM_CONFIG['top_p'],
                    top_k=Config.MINICPM_CONFIG['top_k'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated content
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Analysis generation failed: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate contextual AI response based on video analysis"""
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        try:
            # Build context from chat history
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create chat prompt
            chat_prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            response = self._generate_analysis(chat_prompt)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_chat_latency(latency)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"Error generating chat response: {str(e)}"
    
    def _build_chat_context(self, analysis_result: str, analysis_type: str, user_focus: str, chat_history: List[Dict]) -> str:
        """Build context for chat responses"""
        context = f"""
## VIDEO ANALYSIS CONTEXT
**Analysis Type**: {analysis_type}
**User Focus**: {user_focus}

## ANALYSIS SUMMARY
{analysis_result}

## CONVERSATION HISTORY
"""
        
        # Add recent chat history (last 5 messages)
        for msg in chat_history[-5:]:
            role = "User" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')
            context += f"{role}: {content}\n"
        
        return context
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        self.gpu_service.cleanup()
        torch.cuda.empty_cache()
        print("🧹 MiniCPM-V 2.6 service cleaned up")

# Global instance
minicpm_service = MiniCPMV26Service()

# Round 2: All AI processing is now done locally with MiniCPM-V 2.6
# No external API dependencies required 