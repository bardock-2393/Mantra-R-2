"""
AI Service Module for Round 2 - GPU-powered local AI
Handles Qwen2.5-VL-7B local inference and GPU optimization
Based on official Hugging Face documentation: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

class Qwen25VL7BService:
    """Local GPU-powered Qwen2.5-VL-7B service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the Qwen2.5-VL-7B model on GPU"""
        try:
            print(f"Initializing Qwen2.5-VL-7B on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Check model path
            print(f"Model path: {Config.QWEN25VL_MODEL_PATH}")
            if not Config.QWEN25VL_MODEL_PATH:
                raise RuntimeError("QWEN25VL_MODEL_PATH is empty or None")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Load processor (handles both text and image inputs)
            print(f"Loading processor from {Config.QWEN25VL_MODEL_PATH}...")
            try:
                # Get HF token for Qwen2.5-VL-7B access
                hf_token = Config.QWEN25VL_CONFIG.get('hf_token', '')
                
                self.processor = AutoProcessor.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                
                # Verify processor loaded successfully
                if self.processor is None:
                    raise RuntimeError("Processor failed to load - returned None")
                print(f"Processor loaded successfully: {type(self.processor).__name__}")
                
            except Exception as e:
                print(f"Processor loading failed: {e}")
                print(f"   Model path: {Config.QWEN25VL_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load processor: {e}")
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                print(f"Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            except Exception as e:
                print(f"Tokenizer loading failed: {e}")
                raise RuntimeError(f"Failed to load tokenizer: {e}")
            
            # Load model with optimizations
            print(f"Loading model from {Config.QWEN25VL_MODEL_PATH}...")
            try:
                # Get HF token for Qwen2.5-VL-7B access
                hf_token = Config.QWEN25VL_CONFIG.get('hf_token', '')
                if not hf_token:
                    print("⚠️ Warning: No HF_TOKEN provided. Qwen2.5-VL-7B may require authentication.")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    torch_dtype=torch.float16 if Config.GPU_CONFIG['precision'] == 'float16' else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token if hf_token else None
                )
                
                # Verify model loaded successfully
                if self.model is None:
                    raise RuntimeError("Model failed to load - returned None")
                
                print(f"Model loaded successfully: {type(self.model).__name__}")
                
                # Move model to GPU if not already there
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(self.device)
                    print(f"Model moved to device: {self.device}")
                
                # Warm up the model
                await self._warmup_model()
                
                self.is_initialized = True
                print("✅ Qwen2.5-VL-7B initialized successfully")
                
            except Exception as e:
                print(f"Model loading failed: {e}")
                print(f"   Model path: {Config.QWEN25VL_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load model: {e}")
                
        except Exception as e:
            print(f"❌ Failed to initialize Qwen2.5-VL-7B: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the model with sample inputs"""
        try:
            print("🔥 Warming up Qwen2.5-VL-7B model...")
            
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='red')
            
            # Create test messages
            test_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": "What color is this image?"}
                    ]
                }
            ]
            
            # Process the test input
            inputs = self.processor(
                test_messages,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate a test response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=Config.QWEN25VL_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k']
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Model warmup successful. Test response: {response[:100]}...")
            
        except Exception as e:
            print(f"⚠️ Model warmup failed: {e}")
            # Don't fail initialization if warmup fails
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using Qwen2.5-VL-7B"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Extract video frames for analysis
            frames = self._extract_video_frames(video_path)
            if not frames:
                return "❌ Failed to extract video frames for analysis."
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Analyze with the model
            analysis_result = await self._generate_analysis(prompt, frames)
            
            processing_time = (time.time() - start_time) * 1000
            print(f"🎬 Video analysis completed in {processing_time:.2f}ms")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return f"❌ Analysis failed: {str(e)}"
    
    async def analyze_stream_frame(self, frame: np.ndarray, analysis_type: str = "realtime", user_focus: str = "") -> str:
        """Analyze a single frame from a video stream using 7B model - NEW FOR STREAMING"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Convert numpy array to PIL Image
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Create analysis prompt for real-time
            prompt = self._generate_stream_analysis_prompt(analysis_type, user_focus)
            
            # Create messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process the input
            inputs = self.processor(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate response with faster settings for real-time
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Shorter for real-time
                    do_sample=True,
                    temperature=0.1,  # Lower temperature for consistency
                    top_p=0.95,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated text
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            processing_time = (time.time() - start_time) * 1000
            print(f"🎬 Stream frame analysis completed in {processing_time:.2f}ms")
            
            return response if response else "No analysis generated."
            
        except Exception as e:
            print(f"❌ Stream frame analysis failed: {e}")
            return f"❌ Analysis failed: {str(e)}"
    
    async def analyze_stream_frames_batch(self, frames: List[np.ndarray], analysis_type: str = "realtime", user_focus: str = "") -> List[str]:
        """Analyze multiple frames from a stream in batch - NEW FOR STREAMING"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            results = []
            
            # Process frames in batches for efficiency
            batch_size = min(Config.STREAMING_CONFIG['frame_buffer_size'], len(frames))
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_results = []
                
                for frame in batch_frames:
                    result = await self.analyze_stream_frame(frame, analysis_type, user_focus)
                    batch_results.append(result)
                
                results.extend(batch_results)
            
            processing_time = (time.time() - start_time) * 1000
            print(f"🎬 Batch stream analysis completed in {processing_time:.2f}ms ({len(frames)} frames)")
            
            return results
            
        except Exception as e:
            print(f"❌ Batch stream analysis failed: {e}")
            return [f"❌ Analysis failed: {str(e)}"] * len(frames)
    
    def _extract_video_frames(self, video_path: str) -> List[Image.Image]:
        """Extract key frames from video for analysis"""
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                return []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            # Extract frames at regular intervals
            frame_interval = max(1, total_frames // 8)  # Extract 8 frames
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    
                    if len(frames) >= 8:  # Limit to 8 frames
                        break
            
            cap.release()
            print(f"📸 Extracted {len(frames)} frames for analysis")
            return frames
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            return []
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt based on type and focus"""
        base_prompt = f"Analyze this video with focus on {analysis_type}."
        
        if user_focus:
            base_prompt += f" Pay special attention to: {user_focus}"
        
        analysis_prompts = {
            "content": "Provide a comprehensive analysis of the video content, including objects, actions, scenes, and overall narrative.",
            "objects": "Identify and describe all visible objects, their properties, and spatial relationships in the video.",
            "actions": "Describe all actions, movements, and activities happening in the video with temporal context.",
            "scenes": "Analyze the visual composition, lighting, colors, and overall aesthetic of the video scenes.",
            "narrative": "Provide a narrative analysis of the video, including story elements, progression, and meaning."
        }
        
        specific_prompt = analysis_prompts.get(analysis_type, "")
        if specific_prompt:
            base_prompt += f" {specific_prompt}"
        
        return base_prompt
    
    def _generate_stream_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt for streaming frames - NEW FOR STREAMING"""
        base_prompt = f"Analyze this video frame in real-time. Focus on: {analysis_type}"
        
        if user_focus:
            base_prompt += f" Pay special attention to: {user_focus}"
        
        # Shorter, more focused prompts for real-time analysis
        stream_prompts = {
            "realtime": "Provide a quick, focused analysis of what's happening in this frame.",
            "motion": "Detect any movement, actions, or changes in this frame.",
            "objects": "Identify key objects and their current state in this frame.",
            "events": "Detect any notable events or activities in this frame.",
            "anomaly": "Identify any unusual or concerning elements in this frame."
        }
        
        specific_prompt = stream_prompts.get(analysis_type, "")
        if specific_prompt:
            base_prompt += f" {specific_prompt}"
        
        return base_prompt
    
    async def _generate_analysis(self, prompt: str, frames: List[Image.Image]) -> str:
        """Generate analysis using the model"""
        try:
            # Create messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add frames to the message
            for frame in frames:
                messages[0]["content"].insert(0, {"type": "image", "image": frame})
            
            # Process the input
            inputs = self.processor(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.QWEN25VL_CONFIG['max_length'],
                    do_sample=True,
                    temperature=Config.QWEN25VL_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k']
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated text (remove input prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response if response else "No analysis generated."
            
        except Exception as e:
            print(f"❌ Analysis generation failed: {e}")
            return f"❌ Analysis generation failed: {str(e)}"
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate chat response based on analysis and conversation history"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            # Build chat context
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create chat message
            chat_message = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            inputs = self.tokenizer(
                chat_message,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=Config.QWEN25VL_CONFIG['chat_max_length'],
                    do_sample=True,
                    temperature=Config.QWEN25VL_CONFIG['chat_temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k']
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response if response else "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"❌ Failed to generate response: {str(e)}"
    
    def _build_chat_context(self, analysis_result: str, analysis_type: str, user_focus: str, chat_history: List[Dict]) -> str:
        """Build context for chat responses"""
        context = f"Video Analysis Context:\n"
        context += f"Analysis Type: {analysis_type}\n"
        context += f"User Focus: {user_focus}\n"
        context += f"Analysis Result: {analysis_result}\n\n"
        
        if chat_history:
            context += "Conversation History:\n"
            for entry in chat_history[-5:]:  # Last 5 exchanges
                context += f"{entry['role']}: {entry['message']}\n"
        
        return context
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            "initialized": self.is_initialized,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "streaming_enabled": Config.STREAMING_CONFIG['enabled'],
            "real_time_7b_analysis": Config.STREAMING_CONFIG['real_time_7b_analysis']
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            print("🧹 Qwen2.5-VL-7B service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

# Create global service instance
ai_service = Qwen25VL7BService() 