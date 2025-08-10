"""
MiniCPM-V-2_6 Model Management for Round 2
Handles model loading, GPU optimization, and inference for local AI processing
Based on official MiniCPM-V-2_6 implementation
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from config import Config

class MiniCPMV26Model:
    """MiniCPM-V-2_6 model manager with GPU optimization"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_initialized = False
        self.model_path = Config.MINICPM_MODEL_PATH
        
    def initialize(self):
        """Initialize the MiniCPM-V-2_6 model on GPU"""
        try:
            print(f"🚀 Initializing MiniCPM-V-2_6 on cuda:0...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Set device
            self.device = torch.device('cuda:0')
            print(f"📱 Using device: {self.device}")
            
            # Load model with correct parameters
            print(f"🔍 Model path: {self.model_path}")
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                attn_implementation='sdpa',  # Use SDPA for better performance
                torch_dtype=torch.bfloat16
            )
            
            # Move to GPU and set to eval mode
            self.model = self.model.eval().cuda()
            
            # Load tokenizer
            print(f"📝 Loading processor from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Verify components loaded
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer")
            
            print(f"✅ Processor loaded successfully: {type(self.tokenizer).__name__}")
            print(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            print(f"✅ Model loaded successfully: {type(self.model).__name__}")
            
            # Warm up the model
            self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V-2_6 initialized successfully on {self.device}")
            
            # Print model info
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V-2_6: {e}")
            raise
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V-2_6 model...")
        
        try:
            # Try vision-language warmup first
            print("🖼️ Using vision-language warmup...")
            try:
                # Create a dummy image for warmup
                dummy_image = Image.new('RGB', (224, 224), color='red')
                question = "What color is this image?"
                msgs = [{'role': 'user', 'content': [dummy_image, question]}]
                
                with torch.no_grad():
                    _ = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.tokenizer
                    )
                print("✅ Vision-language warmup completed")
            except Exception as e:
                print(f"❌ VL warmup failed: {e}")
                print("ℹ️ Skipping text-only warmup; continuing without it.")
            
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
    
    def _print_model_info(self):
        """Print model information and memory usage"""
        try:
            if self.model:
                # Get model parameters count
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                print(f"📊 Model Information:")
                print(f"   Total Parameters: {total_params:,}")
                print(f"   Trainable Parameters: {trainable_params:,}")
                print(f"   Model Size: {total_params * 4 / (1024**3):.2f} GB (FP32)")
                
                # Get GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                    print(f"   GPU Memory Allocated: {allocated:.2f} GB")
                    print(f"   GPU Memory Reserved: {reserved:.2f} GB")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not print model info: {e}")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.2, top_p: float = 0.9, 
                     top_k: int = 40) -> str:
        """Generate text using MiniCPM-V-2_6"""
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Create messages format
            msgs = [{'role': 'user', 'content': prompt}]
            
            # Generate response using the model's chat method
            with torch.no_grad():
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    sampling=True
                )
            
            return response
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return f"Error generating text: {str(e)}"
    
    def analyze_video_content(self, video_summary: str, analysis_type: str, 
                             user_focus: str) -> str:
        """Analyze video content using MiniCPM-V-2_6"""
        try:
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis
            analysis_result = self.generate_text(full_prompt, max_new_tokens=2048)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video content analysis failed: {e}")
            return f"Error analyzing video content: {str(e)}"
    
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
        try:
            # Build context from chat history
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create chat prompt
            chat_prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            response = self.generate_text(chat_prompt, max_new_tokens=2048)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"Error generating chat response: {str(e)}"
    
    def _build_chat_context(self, analysis_result: str, analysis_type: str, 
                           user_focus: str, chat_history: List[Dict]) -> str:
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
            role = "User" if msg.get('user') else "Assistant"
            content = msg.get('user', msg.get('ai', ''))
            context += f"{role}: {content}\n"
        
        return context
    
    def get_model_status(self) -> Dict:
        """Get model status and performance metrics"""
        try:
            status = {
                'initialized': self.is_initialized,
                'device': str(self.device) if self.device else None,
                'model_path': self.model_path
            }
            
            if self.is_initialized and torch.cuda.is_available():
                # Get GPU memory usage
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                
                status['gpu_memory'] = {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2)
                }
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            print("🧹 Cleaning up MiniCPM-V-2_6 model...")
            
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            print("✅ MiniCPM-V-2_6 model cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Model cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.is_initialized:
                self.cleanup()
        except:
            pass

# Global instance for easy access
minicpm_v26_model = MiniCPMV26Model() 