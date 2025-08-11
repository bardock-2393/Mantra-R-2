"""
AI Service Module for Round 2 - GPU-powered local AI
Handles MiniCPM-V-2_6 local inference and GPU optimization
Based on official Hugging Face documentation: https://huggingface.co/openbmb/MiniCPM-V-2_6
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

class MiniCPMV26Service:
    """Local GPU-powered MiniCPM-V-2_6 service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the MiniCPM-V-2_6 model on GPU"""
        try:
            print(f"Initializing MiniCPM-V-2_6 on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Check model path
            print(f"Model path: {Config.MINICPM_MODEL_PATH}")
            if not Config.MINICPM_MODEL_PATH:
                raise RuntimeError("MINICPM_MODEL_PATH is empty or None")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Load processor (handles both text and image inputs)
            print(f"Loading processor from {Config.MINICPM_MODEL_PATH}...")
            try:
                # Get HF token for MiniCPM-V-2_6 access
                hf_token = Config.MINICPM_CONFIG.get('hf_token', '')
                
                self.processor = AutoProcessor.from_pretrained(
                    Config.MINICPM_MODEL_PATH,
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                
                # Verify processor loaded successfully
                if self.processor is None:
                    raise RuntimeError("Processor failed to load - returned None")
                print(f"Processor loaded successfully: {type(self.processor).__name__}")
                
            except Exception as e:
                print(f"Processor loading failed: {e}")
                print(f"   Model path: {Config.MINICPM_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load processor: {e}")
            
            # Load tokenizer as fallback
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.MINICPM_MODEL_PATH,
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                print(f"Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            except Exception as e:
                print(f"Tokenizer loading failed, using processor only: {e}")
                self.tokenizer = None
            
            # Load model with optimizations
            print(f"Loading model from {Config.MINICPM_MODEL_PATH}...")
            try:
                # Get HF token for MiniCPM-V-2_6 access
                hf_token = Config.MINICPM_CONFIG.get('hf_token', '')
                if not hf_token:
                    print("⚠️ Warning: No HF_TOKEN provided. MiniCPM-V-2_6 may require authentication.")
                
                self.model = AutoModel.from_pretrained(
                    Config.MINICPM_MODEL_PATH,
                    torch_dtype=torch.float16 if Config.GPU_CONFIG['precision'] == 'float16' else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                
                # Verify model loaded successfully
                if self.model is None:
                    raise RuntimeError("Model failed to load - returned None")
                print(f"Model loaded successfully: {type(self.model).__name__}")
                
            except Exception as e:
                print(f"Model loading failed: {e}")
                print(f"   Model path: {Config.MINICPM_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Move to GPU
            print(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            # Warm up the model
            print("Warming up model...")
            self._warmup_model()
            
            # Start performance monitoring
            self.performance_monitor.start()
            
            self.is_initialized = True
            print("MiniCPM-V-2_6 service initialized successfully!")
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.cleanup()
            raise
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("Warming up MiniCPM-V-2_6 model...")
        
        try:
            if self.processor is None:
                raise RuntimeError("Processor is None - cannot perform warmup")
            if self.model is None:
                raise RuntimeError("Model is None - cannot perform warmup")
            
            # MiniCPM-V-2_6 is a vision-language model, use image + text warmup
            print("Using vision-language warmup for MiniCPM-V-2_6...")
            self._warmup_model_vision_language()
                
        except Exception as e:
            print(f"Model warmup failed: {e}")
            print(f"   Processor type: {type(self.processor)}")
            print(f"   Model type: {type(self.model)}")
            raise
    
    def _warmup_model_vision_language(self):
        """Warmup method using dummy image + text for vision-language model"""
        print("Using vision-language warmup...")
        
        try:
            # Create a dummy image using PIL as shown in the official documentation
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_text = "Hello, this is a warmup message."
            
            # Use the proper chat method as shown in the official documentation
            msgs = [{'role': 'user', 'content': [dummy_image, dummy_text]}]
            
            with torch.no_grad():
                for i in range(3):
                    print(f"Vision-language warmup iteration {i+1}/3...")
                    response = self.model.chat(
                        msgs=msgs,
                        tokenizer=self.processor.tokenizer if self.processor else self.tokenizer,
                        max_new_tokens=8,
                        do_sample=False
                    )
                    
                    # Handle response (could be string or generator)
                    if response is not None:
                        if isinstance(response, str):
                            print(f"  Vision-language warmup iteration {i+1} successful: {response[:50]}...")
                        else:
                            # Handle generator response
                            text_parts = []
                            for part in response:
                                if part is not None:
                                    text_parts.append(str(part))
                            if text_parts:
                                print(f"  Vision-language warmup iteration {i+1} successful: {''.join(text_parts)[:50]}...")
                            else:
                                print(f"  Vision-language warmup iteration {i+1} completed")
                    else:
                        print(f"  ⚠️ Vision-language warmup iteration {i+1} returned None")
                    
        except Exception as e:
            print(f"Vision-language warmup failed: {e}")
            # Fallback to text-only warmup if vision fails
            print("Falling back to text-only warmup...")
            self._warmup_model_text_only()
    
    def _warmup_model_text_only(self):
        """Fallback warmup method using text-only input"""
        print("Using text-only warmup fallback...")
        
        try:
            dummy_text = "Hello, this is a warmup message."
            
            # Create a dummy image as required by the vision-language model
            dummy_image = Image.new('RGB', (224, 224), color='blue')
            
            # Use the proper chat method for text-only input
            msgs = [{'role': 'user', 'content': [dummy_image, dummy_text]}]
            
            with torch.no_grad():
                for i in range(3):
                    print(f"Text warmup iteration {i+1}/3...")
                    response = self.model.chat(
                        msgs=msgs,
                        tokenizer=self.processor.tokenizer if self.processor else self.tokenizer,
                        max_new_tokens=8,
                        do_sample=False
                    )
                    
                    # Handle response (could be string or generator)
                    if response is not None:
                        if isinstance(response, str):
                            print(f"  Text warmup iteration {i+1} successful: {response[:50]}...")
                        else:
                            # Handle generator response
                            text_parts = []
                            for part in response:
                                if part is not None:
                                    text_parts.append(str(part))
                            if text_parts:
                                print(f"  Text warmup iteration {i+1} successful: {''.join(text_parts)[:50]}...")
                            else:
                                print(f"  Text warmup iteration {i+1} completed")
                    else:
                        print(f"  ⚠️ Text warmup iteration {i+1} returned None")
                    
        except Exception as e:
            print(f"Text warmup failed: {e}")
            raise
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using local GPU-powered MiniCPM-V-2_6"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract video summary for context
            video_summary = self._extract_video_summary(video_path)
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Add video context to prompt
            full_prompt = f"{prompt}\n\nVideo Context: {video_summary}\n\nAnalysis:"
            
            # Generate analysis
            analysis_result = self._generate_analysis(full_prompt)
            
            # Record performance metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_analysis_latency(latency)
            
            return analysis_result
            
        except Exception as e:
            print(f"Video analysis failed: {e}")
            return f"Error analyzing video: {str(e)}"
    
    def _extract_video_summary(self, video_path: str) -> str:
        """Extract basic video information for context"""
        try:
            # For now, return basic file info
            # In a full implementation, you'd extract frames, metadata, etc.
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            return f"Video file: {os.path.basename(video_path)}, Size: {file_size:.1f} MB"
        except Exception as e:
            return f"Video file: {os.path.basename(video_path)}"
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt based on type and user focus"""
        base_prompt = f"""You are an AI video analysis expert. Analyze the following video content comprehensively.

Analysis Type: {analysis_type}
User Focus: {user_focus}

Please provide a detailed analysis including:
1. Key events and activities
2. Important objects and people
3. Temporal sequence of events
4. Notable patterns or behaviors
5. Any concerning or interesting observations

Be thorough and professional in your analysis."""
        
        return base_prompt
    
    def _generate_analysis(self, prompt: str) -> str:
        """Generate analysis using MiniCPM-V-2_6"""
        try:
            # MiniCPM-V-2_6 uses a chat interface, not generate
            # Create a dummy image as required by the vision-language model
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            # Format messages for the chat interface
            msgs = [{'role': 'user', 'content': [dummy_image, prompt]}]
            
            # Use the chat method as per official documentation
            with torch.no_grad():
                response = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.processor.tokenizer if self.processor else self.tokenizer,
                    max_new_tokens=1000,  # Reasonable output length
                    temperature=Config.MINICPM_CONFIG['temperature'],
                    top_p=Config.MINICPM_CONFIG['top_p'],
                    top_k=Config.MINICPM_CONFIG['top_k'],
                    do_sample=True
                )
            
            # Handle the response - it could be a string or generator
            if response is None:
                raise RuntimeError("Model chat returned None")
            
            # If response is a string, return it directly
            if isinstance(response, str):
                return response
            
            # If response is a generator (streaming), collect the text
            if hasattr(response, '__iter__') and not isinstance(response, str):
                generated_text = ""
                for new_text in response:
                    if new_text is not None:
                        generated_text += str(new_text)
                return generated_text if generated_text else "No text generated"
            
            # Fallback: convert to string
            return str(response) if response else "Text generation failed"
            
        except Exception as e:
            print(f"❌ Analysis generation failed: {e}")
            return f"Error generating analysis: {str(e)}"
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate contextual AI response based on video analysis"""
        if not self.is_initialized:
            await self.initialize()
        
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
            print(f"Chat response generation failed: {e}")
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
        if self.processor:
            del self.processor
        
        self.gpu_service.cleanup()
        torch.cuda.empty_cache()
        print("MiniCPM-V-2_6 service cleaned up")

# Global instance
minicpm_service = MiniCPMV26Service()

# Round 2: All AI processing is now done locally with MiniCPM-V-2_6
# No external API dependencies required 