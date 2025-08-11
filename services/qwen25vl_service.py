"""
Qwen2.5-VL Service Module
Handles Qwen2.5-VL-7B-Instruct local inference and GPU optimization
Based on official Hugging Face documentation: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Import qwen-vl-utils with proper error handling
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
    print("✅ qwen-vl-utils imported successfully")
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("⚠️ qwen-vl-utils not available, using fallback implementation")
    
    def process_vision_info(messages):
        """Fallback implementation for process_vision_info"""
        image_inputs = []
        video_inputs = []
        
        try:
            for message in messages:
                if "content" in message:
                    for content in message["content"]:
                        if content.get("type") == "image":
                            # Handle image processing
                            print("🖼️ Processing image content...")
                            # For now, just add a placeholder
                            image_inputs.append(None)
                        elif content.get("type") == "video":
                            # Handle video processing
                            video_path = content.get("video", "")
                            print(f"🎬 Processing video content: {video_path}")
                            
                            # Verify video file exists
                            if os.path.exists(video_path):
                                # For now, just add the video path as a placeholder
                                # In a real implementation, you would load and process the video
                                video_inputs.append(video_path)
                                print(f"✅ Video file found and added: {video_path}")
                            else:
                                print(f"⚠️ Video file not found: {video_path}")
                                # Add None to maintain list structure
                                video_inputs.append(None)
            
            print(f"📊 Processed: {len(image_inputs)} images, {len(video_inputs)} videos")
            
        except Exception as e:
            print(f"⚠️ Error in fallback process_vision_info: {e}")
            # Return empty lists on error
            image_inputs = []
            video_inputs = []
        
        return image_inputs, video_inputs

from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

class Qwen25VLService:
    """Local GPU-powered Qwen2.5-VL-7B-Instruct service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the Qwen2.5-VL-7B-Instruct model on GPU"""
        try:
            print(f"🚀 Initializing Qwen2.5-VL-7B-Instruct on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Qwen2.5-VL.")
            
            # Check model path
            print(f"🔍 Model path: {Config.QWEN25VL_MODEL_PATH}")
            if not Config.QWEN25VL_MODEL_PATH:
                raise RuntimeError("QWEN25VL_MODEL_PATH is empty or None")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Load processor (handles both text and image/video inputs)
            print(f"📝 Loading processor from {Config.QWEN25VL_MODEL_PATH}...")
            try:
                # Set min_pixels and max_pixels for optimal performance
                min_pixels = 256 * 28 * 28  # 256 tokens
                max_pixels = 1280 * 28 * 28  # 1280 tokens
                
                self.processor = AutoProcessor.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    trust_remote_code=True
                )
                
                # Verify processor loaded successfully
                if self.processor is None:
                    raise RuntimeError("Processor failed to load - returned None")
                print(f"✅ Processor loaded successfully: {type(self.processor).__name__}")
                
            except Exception as e:
                print(f"❌ Processor loading failed: {e}")
                print(f"   Model path: {Config.QWEN25VL_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load processor: {e}")
            
            # Load tokenizer as fallback
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    trust_remote_code=True
                )
                print(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            except Exception as e:
                print(f"⚠️ Tokenizer loading failed, using processor only: {e}")
                self.tokenizer = None
            
            # Load model with optimizations
            print(f"🤖 Loading model from {Config.QWEN25VL_MODEL_PATH}...")
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    Config.QWEN25VL_MODEL_PATH,
                    torch_dtype=torch.float16 if Config.GPU_CONFIG['precision'] == 'float16' else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Verify model loaded successfully
                if self.model is None:
                    raise RuntimeError("Model failed to load - returned None")
                print(f"✅ Model loaded successfully: {type(self.model).__name__}")
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print(f"   Model path: {Config.QWEN25VL_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Move to GPU
            print(f"🚀 Moving model to device: {self.device}")
            self.model.to(self.device)
            
            # Warm up the model
            await self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ Qwen2.5-VL-7B-Instruct initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Qwen2.5-VL-7B-Instruct: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the Qwen2.5-VL model for optimal performance"""
        try:
            print("🔥 Warming up Qwen2.5-VL model...")
            
            # Create a simple warmup prompt
            warmup_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello, how are you?"}
                    ]
                }
            ]
            
            # Process text-only warmup
            text = self.processor.apply_chat_template(
                warmup_messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate warmup response
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            print("✅ Qwen2.5-VL model warmed up successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using local GPU-powered Qwen2.5-VL-7B-Instruct"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Qwen2.5-VL service not initialized")
            
            print(f"🎬 Analyzing video with Qwen2.5-VL: {video_path}")
            
            # Debug: Check file path details
            print(f"🔍 Video path details:")
            print(f"   Original path: {video_path}")
            print(f"   Absolute path: {os.path.abspath(video_path)}")
            print(f"   File exists: {os.path.exists(video_path)}")
            print(f"   Is file: {os.path.isfile(video_path)}")
            print(f"   File size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
            
            # Check if the path is relative and convert to absolute
            if not os.path.isabs(video_path):
                # Try to resolve relative path
                current_dir = os.getcwd()
                abs_path = os.path.join(current_dir, video_path)
                print(f"   Resolved relative path: {abs_path}")
                print(f"   Resolved path exists: {os.path.exists(abs_path)}")
                
                if os.path.exists(abs_path):
                    video_path = abs_path
                    print(f"✅ Using resolved absolute path: {video_path}")
                else:
                    print(f"⚠️ Resolved path also doesn't exist: {abs_path}")
            
            # Extract video summary for context
            video_summary = self._extract_video_summary(video_path)
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Generate analysis using Qwen2.5-VL
            analysis_result = await self._generate_analysis(prompt, video_path)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            raise RuntimeError(f"Video analysis failed: {e}")
    
    def _extract_video_summary(self, video_path: str) -> str:
        """Extract basic video information for context"""
        try:
            # For now, return basic info - could be enhanced with video metadata
            return f"Video file: {os.path.basename(video_path)}"
        except Exception as e:
            print(f"⚠️ Warning: Could not extract video summary: {e}")
            return "Video file"
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt based on type and user focus"""
        base_prompts = {
            'general': "Analyze this video and provide a comprehensive overview.",
            'behavioral': "Analyze the behavior patterns and actions in this video.",
            'technical': "Provide a technical analysis of this video content.",
            'narrative': "Analyze the narrative structure and storytelling elements.",
            'forensic': "Conduct a forensic analysis of this video for evidence.",
            'commercial': "Analyze this video from a commercial and marketing perspective."
        }
        
        base_prompt = base_prompts.get(analysis_type, base_prompts['general'])
        
        if user_focus:
            return f"{base_prompt} Focus specifically on: {user_focus}"
        return base_prompt
    
    async def _generate_analysis(self, prompt: str, video_path: str) -> str:
        """Generate analysis using Qwen2.5-VL-7B-Instruct"""
        try:
            # Verify the video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get absolute path for the video
            abs_video_path = os.path.abspath(video_path)
            print(f"🎬 Processing video: {abs_video_path}")
            
            # Create messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": abs_video_path  # Use absolute path instead of file:// URL
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process vision info with proper error handling
            try:
                if QWEN_VL_UTILS_AVAILABLE:
                    image_inputs, video_inputs = process_vision_info(messages)
                    print(f"✅ Vision info processed - Images: {len(image_inputs)}, Videos: {len(video_inputs)}")
                    
                    # Validate video inputs
                    if video_inputs and len(video_inputs) > 0:
                        # Check if video inputs are valid
                        valid_videos = []
                        for video_input in video_inputs:
                            if video_input is not None and os.path.exists(str(video_input)):
                                valid_videos.append(video_input)
                            else:
                                print(f"⚠️ Invalid video input: {video_input}")
                        
                        if not valid_videos:
                            print("⚠️ No valid video inputs found, using fallback")
                            return await self._generate_text_only_analysis(prompt, video_path)
                        
                        video_inputs = valid_videos
                    else:
                        print("⚠️ No video inputs found, using fallback")
                        return await self._generate_text_only_analysis(prompt, video_path)
                        
                else:
                    print("⚠️ qwen-vl-utils not available, using fallback")
                    return await self._generate_text_only_analysis(prompt, video_path)
                    
            except Exception as e:
                print(f"⚠️ Vision info processing failed: {e}")
                # Fallback to text-only analysis
                return await self._generate_text_only_analysis(prompt, video_path)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs with proper error handling
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs if 'image_inputs' in locals() else None,
                    videos=video_inputs if 'video_inputs' in locals() else None,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
            except Exception as e:
                print(f"⚠️ Input processing failed: {e}")
                return await self._generate_text_only_analysis(prompt, video_path)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(Config.QWEN25VL_CONFIG['max_length'], 2048),
                    temperature=Config.QWEN25VL_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k'],
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else "Analysis generation failed"
            
        except Exception as e:
            print(f"❌ Analysis generation failed: {e}")
            # Try fallback text-only analysis
            try:
                return await self._generate_text_only_analysis(prompt, video_path)
            except Exception as fallback_error:
                print(f"❌ Fallback analysis also failed: {fallback_error}")
                raise RuntimeError(f"Analysis generation failed: {e}")
    
    async def _generate_text_only_analysis(self, prompt: str, video_path: str) -> str:
        """Fallback text-only analysis when video processing fails"""
        try:
            print("📝 Using fallback text-only analysis...")
            
            # Extract video metadata for context
            video_info = self._extract_video_summary(video_path)
            
            # Create enhanced prompt with video context
            enhanced_prompt = f"""
{prompt}

Video Information:
{video_info}

Please provide a comprehensive analysis based on the video description and context.
"""
            
            # Create messages for text-only generation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(Config.QWEN25VL_CONFIG['max_length'], 1024),
                    temperature=Config.QWEN25VL_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k'],
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            result = output_text[0] if output_text else "Fallback analysis failed"
            print("✅ Fallback text-only analysis completed")
            return result
            
        except Exception as e:
            print(f"❌ Fallback analysis failed: {e}")
            return f"Video analysis failed. Error: {str(e)}"
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate chat response using Qwen2.5-VL-7B-Instruct"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Qwen2.5-VL service not initialized")
            
            # Build chat context
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create messages for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{context}\n\nUser: {message}"}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(Config.QWEN25VL_CONFIG['max_length'], 1024),
                    temperature=Config.QWEN25VL_CONFIG['chat_temperature'],
                    top_p=Config.QWEN25VL_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_CONFIG['top_k'],
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, 'tokenizer') else None
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else "Chat response generation failed"
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            raise RuntimeError(f"Chat response generation failed: {e}")
    
    def _build_chat_context(self, analysis_result: str, analysis_type: str, user_focus: str, chat_history: List[Dict]) -> str:
        """Build chat context from analysis and history"""
        context_parts = [
            f"Analysis Type: {analysis_type}",
            f"User Focus: {user_focus}",
            f"Video Analysis: {analysis_result}"
        ]
        
        if chat_history:
            context_parts.append("Chat History:")
            for entry in chat_history[-3:]:  # Last 3 messages
                context_parts.append(f"- {entry.get('role', 'user')}: {entry.get('content', '')}")
        
        return "\n".join(context_parts)
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'model': 'Qwen2.5-VL-7B-Instruct',
            'initialized': self.is_initialized,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            'qwen_vl_utils_available': QWEN_VL_UTILS_AVAILABLE
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.processor:
                del self.processor
                self.processor = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            print("🧹 Qwen2.5-VL-7B-Instruct service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")

# Global service instance
qwen25vl_service = Qwen25VLService() 