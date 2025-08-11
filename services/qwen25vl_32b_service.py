"""
Qwen2.5-VL-32B Service Module
Handles Qwen2.5-VL-32B-Instruct local inference and GPU optimization
Based on official Hugging Face documentation: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
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
                        # For now, just add a placeholder
                        image_inputs.append(None)
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
                                    video_inputs.append(video_path)
                                    print(f"✅ Video file found and added: {video_path} ({file_size} bytes)")
                                else:
                                    print(f"⚠️ Video file is empty: {video_path}")
                                    video_inputs.append(None)
                            except OSError as e:
                                print(f"⚠️ Error accessing video file {video_path}: {e}")
                                video_inputs.append(None)
                        else:
                            print(f"⚠️ Video file not found: {video_path}")
                            # Add None to maintain list structure
                            video_inputs.append(None)
            
            print(f"📊 Processed: {len(image_inputs)} images, {len(video_inputs)} videos")
            
        except Exception as e:
            print(f"⚠️ Error in fallback process_vision_info: {e}")
            import traceback
            traceback.print_exc()
            # Return empty lists on error
            image_inputs = []
            video_inputs = []
        
        return image_inputs, video_inputs

from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

class Qwen25VL32BService:
    """Local GPU-powered Qwen2.5-VL-32B-Instruct service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the Qwen2.5-VL-32B-Instruct model on GPU"""
        try:
            print(f"🚀 Initializing Qwen2.5-VL-32B-Instruct on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Qwen2.5-VL-32B.")
            
            # Check model path
            print(f"🔍 Model path: {Config.QWEN25VL_32B_MODEL_PATH}")
            if not Config.QWEN25VL_32B_MODEL_PATH:
                raise RuntimeError("QWEN25VL_32B_MODEL_PATH is empty or None")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Load processor (handles both text and image/video inputs)
            print(f"📝 Loading processor from {Config.QWEN25VL_32B_MODEL_PATH}...")
            try:
                # Set min_pixels and max_pixels for optimal performance
                min_pixels = Config.QWEN25VL_32B_CONFIG['min_pixels']
                max_pixels = Config.QWEN25VL_32B_CONFIG['max_pixels']
                
                self.processor = AutoProcessor.from_pretrained(
                    Config.QWEN25VL_32B_MODEL_PATH,
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
                print(f"   Model path: {Config.QWEN25VL_32B_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load processor: {e}")
            
            # Load tokenizer as fallback
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.QWEN25VL_32B_MODEL_PATH,
                    trust_remote_code=True
                )
                print(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            except Exception as e:
                print(f"⚠️ Tokenizer loading failed, using processor only: {e}")
                self.tokenizer = None
            
            # Load model with optimizations for 32B model
            print(f"🤖 Loading Qwen2.5-VL-32B-Instruct model from {Config.QWEN25VL_32B_MODEL_PATH}...")
            try:
                # Use flash attention 2 for better acceleration and memory saving
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    Config.QWEN25VL_32B_MODEL_PATH,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for 32B model
                    attn_implementation="flash_attention_2",  # Enable flash attention
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Verify model loaded successfully
                if self.model is None:
                    raise RuntimeError("Model failed to load - returned None")
                print(f"✅ Qwen2.5-VL-32B-Instruct model loaded successfully: {type(self.model).__name__}")
                
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                print(f"   Model path: {Config.QWEN25VL_32B_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Move to GPU
            print(f"🚀 Moving model to device: {self.device}")
            self.model.to(self.device)
            
            # Warm up the model
            await self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ Qwen2.5-VL-32B-Instruct initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Qwen2.5-VL-32B-Instruct: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the Qwen2.5-VL-32B model for optimal performance"""
        try:
            print("🔥 Warming up Qwen2.5-VL-32B model...")
            
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
            
            print("✅ Qwen2.5-VL-32B model warmed up successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using local GPU-powered Qwen2.5-VL-32B-Instruct"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Qwen2.5-VL-32B service not initialized")
            
            print(f"🎬 Analyzing video with Qwen2.5-VL-32B: {video_path}")
            
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
            
            # Generate analysis prompt using the same system as the old project
            if ANALYSIS_TEMPLATES_AVAILABLE:
                analysis_prompt = generate_analysis_prompt(analysis_type, user_focus)
                print("✅ Using analysis templates from old project")
            else:
                analysis_prompt = self._generate_fallback_analysis_prompt(analysis_type, user_focus)
                print("⚠️ Using fallback analysis prompt")
            
            # Generate analysis using Qwen2.5-VL-32B
            analysis_result = await self._generate_analysis(analysis_prompt, video_path)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            raise RuntimeError(f"Video analysis failed: {e}")
    
    def _generate_fallback_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate fallback analysis prompt when templates are not available"""
        base_prompts = {
            'general': "Analyze this video and provide a comprehensive overview.",
            'behavioral': "Analyze the behavior patterns and actions in this video.",
            'technical': "Provide a technical analysis of this video content.",
            'narrative': "Analyze the narrative structure and storytelling elements.",
            'forensic': "Conduct a forensic analysis of this video for evidence.",
            'commercial': "Analyze this video from a commercial and marketing perspective.",
            'comprehensive_analysis': "Provide a comprehensive multi-dimensional analysis covering all aspects.",
            'safety_investigation': "Conduct a thorough safety analysis with risk assessment.",
            'creative_review': "Provide comprehensive creative and aesthetic analysis."
        }
        
        base_prompt = base_prompts.get(analysis_type, base_prompts['general'])
        
        if user_focus:
            return f"{base_prompt} Focus specifically on: {user_focus}"
        return base_prompt
    
    async def _generate_analysis(self, prompt: str, video_path: str) -> str:
        """Generate analysis using Qwen2.5-VL-32B-Instruct"""
        try:
            # Verify the video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get absolute path for the video
            abs_video_path = os.path.abspath(video_path)
            print(f"🎬 Processing video: {abs_video_path}")
            
            # Create messages for Qwen2.5-VL-32B
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
                        print(f"✅ Using {len(video_inputs)} valid video inputs for analysis")
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
                # Use the actual video inputs from vision processing
                inputs = self.processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                print(f"✅ Inputs prepared successfully - Text: {len(text)}, Videos: {len(video_inputs) if video_inputs else 0}")
            except Exception as e:
                print(f"⚠️ Input processing failed: {e}")
                return await self._generate_text_only_analysis(prompt, video_path)
            
            # Generate response with 32B model optimizations
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(Config.QWEN25VL_32B_CONFIG['max_length'], 2048),
                    temperature=Config.QWEN25VL_32B_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_32B_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_32B_CONFIG['top_k'],
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

Please provide a comprehensive analysis based on the video description and context. Since direct video analysis is not available, focus on:
1. General insights about the video content based on the metadata
2. Potential analysis approaches for this type of video
3. Recommendations for what to look for in similar videos
4. Contextual information based on the video parameters

Please provide a detailed, helpful analysis.
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
                    max_new_tokens=min(Config.QWEN25VL_32B_CONFIG['max_length'], 1024),
                    temperature=Config.QWEN25VL_32B_CONFIG['temperature'],
                    top_p=Config.QWEN25VL_32B_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_32B_CONFIG['top_k'],
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
        """Generate chat response using Qwen2.5-VL-32B-Instruct with the same prompt system as the old project"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Qwen2.5-VL-32B service not initialized")
            
            # Build chat context using the same system as the old project
            context_prompt = self._build_enhanced_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create messages for Qwen2.5-VL-32B
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{context_prompt}\n\nUser: {message}"}
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
                    max_new_tokens=min(Config.QWEN25VL_32B_CONFIG['max_length'], 1024),
                    temperature=Config.QWEN25VL_32B_CONFIG['chat_temperature'],
                    top_p=Config.QWEN25VL_32B_CONFIG['top_p'],
                    top_k=Config.QWEN25VL_32B_CONFIG['top_k'],
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
    
    def _build_enhanced_chat_context(self, analysis_result: str, analysis_type: str, user_focus: str, chat_history: List[Dict]) -> str:
        """Build enhanced chat context using the same system as the old project"""
        # Enhanced agentic conversation prompt with advanced capabilities (same as old project)
        context_prompt = f"""
You are an advanced AI video analysis agent with comprehensive understanding capabilities. You are engaging in a multi-turn conversation about a video that has been analyzed.

## AGENT CONVERSATION PROTOCOL

### Current Context:
- Analysis Type: {analysis_type.replace('_', ' ').title()}
- Original Analysis Focus: {user_focus}
- Conversation History: Available for context awareness

### Video Analysis Context:
{analysis_result}
            
### Agent Capabilities:
- Autonomous Analysis: Provide comprehensive insights beyond the immediate question
- Multi-Modal Understanding: Reference visual, audio, temporal, and spatial elements
- Context Awareness: Adapt responses based on conversation history and user intent
- Proactive Insights: Offer additional relevant information and observations
- Comprehensive Reporting: Provide detailed, structured responses
- Adaptive Focus: Adjust response depth based on question complexity

### Response Quality Standards:
1. **Precision & Accuracy**: Provide exact, verifiable information with specific timestamps
2. **Comprehensive Coverage**: Address all aspects of the question thoroughly
3. **Evidence-Based**: Support every claim with specific evidence from the analysis
4. **Clear Structure**: Use logical organization with clear headings and bullet points
5. **Professional Tone**: Maintain engaging yet professional communication
6. **Proactive Insights**: Offer additional relevant observations beyond the direct question
7. **Visual Clarity**: Use formatting to enhance readability (bold, italics, lists)
8. **Contextual Awareness**: Reference previous conversation context when relevant

### Response Format Guidelines:
- **Start with a direct answer** to the user's question
- **Use clear headings** for different sections (e.g., "**Key Findings:**", "**Timeline:**", "**Additional Insights:**")
- **Include specific timestamps** when discussing events (e.g., "At **00:15-00:17**")
- **Use bullet points** for lists and multiple items
- **Bold important information** for emphasis
- **Provide quantitative data** when available (durations, counts, measurements)
- **Include relevant context** that enhances understanding
- **End with actionable insights** or additional observations when relevant

### Specialized Response Areas:
- **Safety Analysis**: Focus on specific safety concerns, violations, and recommendations
- **Timeline Events**: Provide chronological details with precise timestamps
- **Pattern Recognition**: Highlight recurring behaviors, trends, and anomalies
- **Performance Assessment**: Discuss quality, efficiency, and optimization opportunities
- **Creative Elements**: Analyze artistic, aesthetic, and creative aspects
- **Technical Details**: Provide technical specifications and quality assessments
- **Behavioral Insights**: Analyze human behavior, interactions, and social dynamics
- **Environmental Factors**: Consider context, setting, and environmental conditions

### Quality Enhancement Techniques:
- **Quantify responses**: Use specific numbers, durations, and measurements
- **Cross-reference information**: Connect related details across different sections
- **Provide context**: Explain why certain details are significant
- **Use descriptive language**: Make responses vivid and engaging
- **Structure complex information**: Break down complex topics into digestible sections
- **Highlight patterns**: Identify and explain recurring themes or behaviors
- **Offer insights**: Provide analysis beyond simple description

Your mission is to provide **exceptional quality responses** that demonstrate deep understanding of the video content, offer precise information with timestamps, and deliver insights that exceed user expectations. Every response should be comprehensive, well-structured, and highly informative.
"""
        
        # Include conversation history for context awareness (same as old project)
        conversation_context = ""
        if len(chat_history) > 2:  # More than just current message
            recent_messages = chat_history[-6:]  # Last 6 messages for context
            conversation_context = "\n\n### Recent Conversation Context:\n"
            for msg in recent_messages:
                if 'user' in msg:
                    conversation_context += f"User: {msg['user']}\n"
                elif 'ai' in msg:
                    conversation_context += f"Agent: {msg['ai'][:200]}...\n"  # Truncate for context
        
        enhanced_context_prompt = context_prompt + conversation_context
        return enhanced_context_prompt
    
    def _extract_video_summary(self, video_path: str) -> str:
        """Extract basic video information for context"""
        try:
            if not os.path.exists(video_path):
                return f"Video file: {os.path.basename(video_path)} (file not found)"
            
            # Get file information
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Try to extract video metadata using video_utils
            try:
                from utils.video_utils import extract_video_metadata
                metadata = extract_video_metadata(video_path)
                if metadata:
                    duration = metadata.get('duration', 0)
                    fps = metadata.get('fps', 0)
                    width = metadata.get('width', 0)
                    height = metadata.get('height', 0)
                    frame_count = metadata.get('frame_count', 0)
                    
                    summary = f"""
Video File: {os.path.basename(video_path)}
File Size: {file_size_mb:.2f} MB
Duration: {duration:.2f} seconds
Frame Rate: {fps:.2f} fps
Resolution: {width}x{height} pixels
Total Frames: {frame_count}
"""
                else:
                    summary = f"Video File: {os.path.basename(video_path)}\nFile Size: {file_size_mb:.2f} MB\nDuration: Unknown"
            except ImportError:
                # Fallback if video_utils not available
                summary = f"Video File: {os.path.basename(video_path)}\nFile Size: {file_size_mb:.2f} MB"
            
            return summary.strip()
            
        except Exception as e:
            print(f"⚠️ Warning: Could not extract video summary: {e}")
            return f"Video file: {os.path.basename(video_path)}"
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'model': 'Qwen2.5-VL-32B-Instruct',
            'initialized': self.is_initialized,
            'device': str(self.device),
            'gpu_available': torch.cuda.is_available(),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            'qwen_vl_utils_available': QWEN_VL_UTILS_AVAILABLE,
            'analysis_templates_available': ANALYSIS_TEMPLATES_AVAILABLE
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
            print("🧹 Qwen2.5-VL-32B-Instruct service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")

# Global service instance
qwen25vl_32b_service = Qwen25VL32BService() 