"""
Qwen2.5-VL-7B Service Module
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

# Import analysis templates from the old project
try:
    from analysis_templates import generate_analysis_prompt
    ANALYSIS_TEMPLATES_AVAILABLE = True
    print("✅ Analysis templates imported successfully")
except ImportError:
    ANALYSIS_TEMPLATES_AVAILABLE = False
    print("⚠️ Analysis templates not available, using fallback prompts")

# Import new services
try:
    from video_chunking_service import VideoChunkingService
    from websocket_service import WebSocketService
    CHUNKING_AVAILABLE = True
    print("✅ Video chunking service imported successfully")
except ImportError:
    CHUNKING_AVAILABLE = False
    print("⚠️ Video chunking service not available, using fallback processing")

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
            if not messages or not isinstance(messages, list):
                print("⚠️ Invalid messages format")
                return image_inputs, video_inputs
                
            for message in messages:
                if not isinstance(message, dict) or "content" not in message:
                    continue
                    
                content_list = message["content"]
                if not isinstance(content_list, list):
                    continue
                    
                for content in content_list:
                    if not isinstance(content, dict):
                        continue
                        
                    content_type = content.get("type")
                    if content_type == "image":
                        # Handle image processing
                        print("🖼️ Processing image content...")
                        # For now, just skip images since we're focusing on video
                        pass
                    elif content_type == "video":
                        # Handle video processing
                        video_path = content.get("video", "")
                        print(f"🎬 Processing video content: {video_path}")
                        
                        # Verify video file exists and is accessible
                        if video_path and os.path.exists(str(video_path)):
                            # Check if it's a valid video file
                            try:
                                file_size = os.path.getsize(str(video_path))
                                if file_size > 0:
                                    # Convert to absolute path and ensure proper format
                                    abs_path = os.path.abspath(str(video_path))
                                    print(f"✅ Video file found and added: {abs_path} ({file_size} bytes)")
                                    
                                    # For the fallback, we need to handle video loading ourselves
                                    # since qwen-vl-utils is not available
                                    try:
                                        # Try to load video with OpenCV to create a tensor
                                        import cv2
                                        import torch
                                        
                                        cap = cv2.VideoCapture(abs_path)
                                        if cap.isOpened():
                                            # Extract a few frames to create a video tensor
                                            frames = []
                                            frame_count = 0
                                            max_frames = 16  # 7B model can handle more frames than 32B
                                            
                                            while cap.isOpened() and frame_count < max_frames:
                                                ret, frame = cap.read()
                                                if ret:
                                                    # Resize frame to 224x224 for memory efficiency
                                                    frame = cv2.resize(frame, (224, 224))
                                                    # Convert BGR to RGB
                                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                    # Convert to tensor
                                                    frame_tensor = torch.from_numpy(frame).float() / 255.0
                                                    frames.append(frame_tensor)
                                                    frame_count += 1
                                                else:
                                                    break
                                            
                                            cap.release()
                                            
                                            if frames:
                                                # Stack frames into a video tensor
                                                video_tensor = torch.stack(frames)
                                                print(f"✅ Video tensor created with {len(frames)} frames")
                                                video_inputs.append(video_tensor)
                                            else:
                                                print("⚠️ No frames extracted from video")
                                        else:
                                            print("⚠️ Could not open video file")
                                    except Exception as e:
                                        print(f"⚠️ Error processing video with OpenCV: {e}")
                                        # Add the video path anyway for potential processing
                                        video_inputs.append(abs_path)
                                else:
                                    print("⚠️ Video file is empty")
                            except Exception as e:
                                print(f"⚠️ Error checking video file: {e}")
                        else:
                            print(f"⚠️ Video file not found: {video_path}")
                    else:
                        print(f"⚠️ Unknown content type: {content_type}")
                        
        except Exception as e:
            print(f"⚠️ Error in fallback process_vision_info: {e}")
            
        return image_inputs, video_inputs

# Import configuration
try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ Config not available, using default settings")
    CONFIG_AVAILABLE = False

class Qwen25VL7BService:
    """
    Local GPU-powered Qwen2.5-VL-7B-Instruct service for video analysis
    Optimized for 7B model with balanced performance and memory usage
    """
    
    def __init__(self):
        """Initialize the Qwen2.5-VL-7B service"""
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self.is_initialized = False
        
        # 7B model configuration - balanced performance
        self.model_config = {
            'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
            'max_length': 8192,  # 7B model can handle longer context
            'temperature': 0.3,   # Balanced creativity
            'top_p': 0.9,        # Balanced sampling
            'top_k': 40,         # Balanced selection
            'chat_temperature': 0.4,  # Slightly more creative for chat
            # 7B model can handle more pixels than 32B
            'min_pixels': 512 * 28 * 28,   # 512 tokens
            'max_pixels': 1280 * 28 * 28,  # 1280 tokens
            # Balanced generation settings
            'use_cache': True,
            'do_sample': True,    # Enable sampling for better quality
            'num_beams': 2,       # 2 beams for balance
            'early_stopping': True,
            'repetition_penalty': 1.1,  # Light repetition penalty
            'length_penalty': 1.0,      # No length penalty
            # Memory optimization for 7B model
            'batch_size': 2,      # 7B can handle 2 batches
            'gradient_checkpointing': False,  # Disable for inference
            'use_flash_attention': True,  # Enable for 7B model
            'compile_mode': 'reduce-overhead',  # Speed optimization
            # Timeout settings
            'vision_timeout': 1200,  # 20 minutes for vision processing
            'generation_timeout': 1200,  # 20 minutes for text generation
            'total_timeout': 2400,  # 40 minutes total analysis time
            # Frame optimization for 7B model
            'max_frames_large': 8,    # Large videos (>500MB) - 8 frames
            'max_frames_medium': 12,  # Medium videos (100-500MB) - 12 frames
            'max_frames_small': 16,   # Small videos (<100MB) - 16 frames
            'frame_resolution': 336,  # 7B can handle higher resolution
            'use_half_precision': True,  # Use FP16 for memory efficiency
            # Generation settings
            'max_new_tokens': 1024,   # 7B can generate longer outputs
            'min_new_tokens': 200,    # Minimum 200 tokens for quality
            'no_repeat_ngram_size': 3,  # Enable n-gram blocking
            'bad_words_ids': None,   # No bad word filtering
        }
        
        # Override with config if available
        if CONFIG_AVAILABLE:
            self.model_config.update(Config.QWEN25VL_CONFIG)
        
        print(f"🚀 Qwen2.5-VL-7B service initialized with config: {self.model_config['model_name']}")
    
    async def initialize(self):
        """Initialize the Qwen2.5-VL-7B-Instruct model on GPU"""
        try:
            print(f"🚀 Initializing Qwen2.5-VL-7B-Instruct on {self.device}...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Qwen2.5-VL-7B.")
            
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"🎯 Using device: {self.device}")
            
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
                
                # 7B model requires less memory than 32B
                if gpu_memory < 12:
                    print("⚠️ Warning: GPU memory may be insufficient for optimal 7B performance")
                    print("   Recommended: 16GB+ for optimal performance")
                elif gpu_memory < 8:
                    print("❌ Error: Insufficient GPU memory for 7B model")
                    print("   Required: 8GB+, Recommended: 16GB+")
                    return False
            
            # Load tokenizer
            await self._load_tokenizer()
            
            # Load processor
            await self._load_processor()
            
            # Load model
            await self._load_model()
            
            # Warm up the model
            await self._warm_up()
            
            self.is_initialized = True
            print(f"✅ Qwen2.5-VL-7B-Instruct initialized successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Qwen2.5-VL-7B-Instruct: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _load_tokenizer(self):
        """Load the tokenizer for Qwen2.5-VL-7B"""
        try:
            print("🔤 Loading tokenizer...")
            
            # Try to load from config path first
            model_path = self.model_config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                print(f"✅ Tokenizer loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load tokenizer from {model_path}: {e}")
                
                # Try alternative path
                alternative_path = "Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"🔄 Trying alternative path: {alternative_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    alternative_path,
                    trust_remote_code=True,
                    use_fast=False
                )
                print(f"✅ Tokenizer loaded from {alternative_path}")
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 Set pad_token to eos_token")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
    
    async def _load_processor(self):
        """Load the processor for Qwen2.5-VL-7B"""
        try:
            print("🔧 Loading processor...")
            
            # Try to load from config path first
            model_path = self.model_config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            
            try:
                # Load processor with custom pixel settings for 7B model
                min_pixels = self.model_config.get('min_pixels', 512 * 28 * 28)
                max_pixels = self.model_config.get('max_pixels', 1280 * 28 * 28)
                
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                print(f"✅ Processor loaded from {model_path}")
                print(f"   Pixel range: {min_pixels} - {max_pixels}")
                
            except Exception as e:
                print(f"⚠️ Failed to load processor from {model_path}: {e}")
                
                # Try alternative path
                alternative_path = "Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"🔄 Trying alternative path: {alternative_path}")
                
                self.processor = AutoProcessor.from_pretrained(
                    alternative_path,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                print(f"✅ Processor loaded from {alternative_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load processor: {e}")
            raise
    
    async def _load_model(self):
        """Load the Qwen2.5-VL-7B model"""
        try:
            print("🤖 Loading Qwen2.5-VL-7B-Instruct model...")
            
            # Try to load from config path first
            model_path = self.model_config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            
            try:
                # Load model with 7B-optimized settings
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for 7B model
                    attn_implementation="flash_attention_2",  # Enable flash attention for 7B
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"✅ Model loaded from {model_path}")
                
            except Exception as e:
                print(f"⚠️ Failed to load model from {model_path}: {e}")
                
                # Try alternative path
                alternative_path = "Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"🔄 Trying alternative path: {alternative_path}")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    alternative_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"✅ Model loaded from {alternative_path}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Enable evaluation mode
            self.model.eval()
            
            print(f"✅ Qwen2.5-VL-7B-Instruct model loaded successfully: {type(self.model).__name__}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    async def _warm_up(self):
        """Warm up the Qwen2.5-VL-7B model for optimal performance"""
        try:
            print("🔥 Warming up Qwen2.5-VL-7B model...")
            
            # Create a simple warm-up prompt
            warm_up_text = "Hello, how are you?"
            
            # Process the text
            inputs = self.tokenizer(
                warm_up_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate a short response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            print("✅ Qwen2.5-VL-7B model warmed up successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Warm-up failed: {e}")
            # Don't fail initialization for warm-up issues
            return True
    
    async def analyze_video(self, video_path: str, analysis_type: str = "comprehensive") -> Dict:
        """Analyze video using Qwen2.5-VL-7B-Instruct - main analysis method"""
        if not self.is_initialized:
            raise RuntimeError("Qwen2.5-VL-7B service not initialized")
        
        try:
            print(f"🎬 Starting video analysis with Qwen2.5-VL-7B: {video_path}")
            start_time = time.time()
            
            # Analyze video using local GPU-powered Qwen2.5-VL-7B-Instruct
            result = await self._analyze_video_local(video_path, analysis_type)
            
            # Add timing information
            end_time = time.time()
            result['processing_time'] = end_time - start_time
            result['model_used'] = 'Qwen2.5-VL-7B-Instruct'
            
            print(f"✅ Video analysis completed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'model_used': 'Qwen2.5-VL-7B-Instruct',
                'status': 'failed'
            }
    
    async def _analyze_video_local(self, video_path: str, analysis_type: str) -> Dict:
        """Analyze video using local GPU-powered Qwen2.5-VL-7B-Instruct"""
        if not self.is_initialized:
            raise RuntimeError("Qwen2.5-VL-7B service not initialized")
        
        try:
            print(f"🎬 Analyzing video with Qwen2.5-VL-7B: {video_path}")
            
            # Verify video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get video information
            video_info = self._get_video_info(video_path)
            print(f"📹 Video info: {video_info['duration']:.2f}s, {video_info['fps']:.1f}fps, {video_info['resolution']}")
            
            # Determine frame extraction strategy based on video size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            if file_size_mb > 500:
                max_frames = self.model_config.get('max_frames_large', 8)
                print(f"📊 Large video detected ({file_size_mb:.1f}MB), extracting {max_frames} frames")
            elif file_size_mb > 100:
                max_frames = self.model_config.get('max_frames_medium', 12)
                print(f"📊 Medium video detected ({file_size_mb:.1f}MB), extracting {max_frames} frames")
            else:
                max_frames = self.model_config.get('max_frames_small', 16)
                print(f"📊 Small video detected ({file_size_mb:.1f}MB), extracting {max_frames} frames")
            
            # Extract frames
            frames = self._extract_frames(video_path, max_frames)
            print(f"🖼️ Extracted {len(frames)} frames for analysis")
            
            # Generate analysis using Qwen2.5-VL-7B
            analysis_result = await self._generate_analysis(frames, video_info, analysis_type)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Local video analysis failed: {e}")
            raise
    
    async def _generate_analysis(self, frames: List[torch.Tensor], video_info: Dict, analysis_type: str) -> Dict:
        """Generate analysis using Qwen2.5-VL-7B-Instruct"""
        try:
            print("🧠 Generating analysis with Qwen2.5-VL-7B-Instruct...")
            
            # Create messages for Qwen2.5-VL-7B with correct video format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,  # Pass frames directly
                        },
                        {
                            "type": "text",
                            "text": self._get_analysis_prompt(analysis_type, video_info)
                        }
                    ]
                }
            ]
            
            # Process vision information
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                # Use fallback processing
                image_inputs, video_inputs = process_vision_info(messages)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.get('max_new_tokens', 1024),
                    temperature=self.model_config.get('temperature', 0.3),
                    top_p=self.model_config.get('top_p', 0.9),
                    top_k=self.model_config.get('top_k', 40),
                    do_sample=self.model_config.get('do_sample', True),
                    num_beams=self.model_config.get('num_beams', 2),
                    early_stopping=self.model_config.get('early_stopping', True),
                    repetition_penalty=self.model_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.eos_token_id
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
            
            # Extract the generated text
            if output_text and len(output_text) > 0:
                analysis_text = output_text[0].strip()
            else:
                analysis_text = "No analysis generated"
            
            print(f"✅ Analysis generated: {len(analysis_text)} characters")
            
            return {
                'analysis': analysis_text,
                'analysis_type': analysis_type,
                'video_info': video_info,
                'frames_analyzed': len(frames),
                'model_config': {
                    'model': 'Qwen2.5-VL-7B-Instruct',
                    'temperature': self.model_config.get('temperature', 0.3),
                    'max_tokens': self.model_config.get('max_new_tokens', 1024)
                }
            }
            
        except Exception as e:
            print(f"❌ Analysis generation failed: {e}")
            raise
    
    def _get_analysis_prompt(self, analysis_type: str, video_info: Dict) -> str:
        """Get the appropriate analysis prompt based on type"""
        if ANALYSIS_TEMPLATES_AVAILABLE:
            try:
                return generate_analysis_prompt(analysis_type, video_info)
            except Exception as e:
                print(f"⚠️ Failed to use analysis template: {e}")
        
        # Fallback prompts
        base_prompt = f"""Analyze this video comprehensively. The video is {video_info['duration']:.2f} seconds long, recorded at {video_info['fps']:.1f} FPS, with resolution {video_info['resolution']}.

Please provide a detailed analysis including:
1. Visual content description
2. Key events and activities
3. Objects, people, and scenes identified
4. Temporal sequence of events
5. Any notable patterns or anomalies
6. Overall context and purpose of the video

Be thorough and professional in your analysis."""
        
        if analysis_type == "brief":
            base_prompt += "\n\nPlease provide a concise summary focusing on the most important elements."
        elif analysis_type == "detailed":
            base_prompt += "\n\nPlease provide an extremely detailed analysis with specific timestamps and comprehensive observations."
        elif analysis_type == "technical":
            base_prompt += "\n\nPlease focus on technical aspects, quality, and technical characteristics of the video."
        
        return base_prompt
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Extract basic video information"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'duration': duration
            }
            
        except Exception as e:
            print(f"⚠️ Could not extract video info: {e}")
            return {
                'fps': 0,
                'frame_count': 0,
                'width': 0,
                'height': 0,
                'resolution': 'unknown',
                'duration': 0
            }
    
    def _extract_frames(self, video_path: str, max_frames: int) -> List[torch.Tensor]:
        """Extract frames from video for analysis"""
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                raise RuntimeError("Invalid video file or frame count")
            
            # Calculate frame indices to extract
            if total_frames <= max_frames:
                # Extract all frames
                frame_indices = list(range(total_frames))
            else:
                # Extract evenly distributed frames
                frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame to target resolution
                    target_res = self.model_config.get('frame_resolution', 336)
                    frame = cv2.resize(frame, (target_res, target_res))
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to tensor and normalize
                    frame_tensor = torch.from_numpy(frame).float() / 255.0
                    
                    # Move to device if available
                    if self.device and hasattr(frame_tensor, 'to'):
                        frame_tensor = frame_tensor.to(self.device)
                    
                    frames.append(frame_tensor)
                else:
                    print(f"⚠️ Failed to read frame {frame_idx}")
            
            cap.release()
            
            if not frames:
                raise RuntimeError("No frames could be extracted from video")
            
            print(f"✅ Extracted {len(frames)} frames successfully")
            return frames
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            raise
    
    async def generate_text_response(self, prompt: str) -> str:
        """Generate text response using Qwen2.5-VL-7B-Instruct"""
        if not self.is_initialized:
            raise RuntimeError("Qwen2.5-VL-7B service not initialized")
        
        try:
            # Create messages for Qwen2.5-VL-7B
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
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
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.get('max_new_tokens', 1024),
                    temperature=self.model_config.get('chat_temperature', 0.4),
                    top_p=self.model_config.get('top_p', 0.9),
                    top_k=self.model_config.get('top_k', 40),
                    do_sample=self.model_config.get('do_sample', True),
                    num_beams=self.model_config.get('num_beams', 2),
                    early_stopping=self.model_config.get('early_stopping', True),
                    repetition_penalty=self.model_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.eos_token_id
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
            
            # Extract the generated text
            if output_text and len(output_text) > 0:
                response_text = output_text[0].strip()
            else:
                response_text = "No response generated"
            
            return response_text
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    async def chat(self, message: str, chat_history: List[Dict] = None) -> Dict:
        """Handle chat messages using Qwen2.5-VL-7B-Instruct"""
        if not self.is_initialized:
            raise RuntimeError("Qwen2.5-VL-7B service not initialized")
        
        try:
            # Generate chat response using Qwen2.5-VL-7B-Instruct with the same prompt system as the old project
            if not self.is_initialized:
                raise RuntimeError("Qwen2.5-VL-7B service not initialized")
            
            # Create messages for Qwen2.5-VL-7B
            messages = []
            
            # Add chat history if available
            if chat_history:
                for entry in chat_history:
                    if entry.get('role') == 'user':
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": entry.get('message', '')}]
                        })
                    elif entry.get('role') == 'assistant':
                        messages.append({
                            "role": "assistant",
                            "content": entry.get('message', '')
                        })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": message}]
            })
            
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
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.get('max_new_tokens', 1024),
                    temperature=self.model_config.get('chat_temperature', 0.4),
                    top_p=self.model_config.get('top_p', 0.9),
                    top_k=self.model_config.get('top_k', 40),
                    do_sample=self.model_config.get('do_sample', True),
                    num_beams=self.model_config.get('num_beams', 2),
                    early_stopping=self.model_config.get('early_stopping', True),
                    repetition_penalty=self.model_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.eos_token_id
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
            
            # Extract the generated text
            if output_text and len(output_text) > 0:
                response_text = output_text[0].strip()
            else:
                response_text = "No response generated"
            
            return {
                'response': response_text,
                'model': 'Qwen2.5-VL-7B-Instruct',
                'timestamp': time.time(),
                'chat_history': chat_history or []
            }
            
        except Exception as e:
            print(f"❌ Chat failed: {e}")
            return {
                'response': f"Error in chat: {str(e)}",
                'model': 'Qwen2.5-VL-7B-Instruct',
                'timestamp': time.time(),
                'chat_history': chat_history or []
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_initialized:
            return {
                'status': 'not_initialized',
                'model': 'Qwen2.5-VL-7B-Instruct',
                'error': 'Service not initialized'
            }
        
        try:
            # Get model parameters count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Get device info
            device_info = str(self.device)
            if hasattr(self.device, 'type') and self.device.type == 'cuda':
                device_info += f" ({torch.cuda.get_device_name(self.device.index)})"
            
            return {
                'status': 'initialized',
                'model': 'Qwen2.5-VL-7B-Instruct',
                'model_type': type(self.model).__name__,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': device_info,
                'config': self.model_config,
                'initialized_at': getattr(self, '_initialized_at', 'unknown')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'model': 'Qwen2.5-VL-7B-Instruct',
                'error': str(e)
            }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            print("🧹 Cleaning up Qwen2.5-VL-7B-Instruct service...")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅ CUDA cache cleared")
            
            # Reset state
            self.is_initialized = False
            self.model = None
            self.tokenizer = None
            self.processor = None
            
            print("✅ Qwen2.5-VL-7B-Instruct service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

