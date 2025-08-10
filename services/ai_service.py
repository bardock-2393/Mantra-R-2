"""
AI Service Module for Round 2 - GPU-powered local AI
Handles MiniCPM-V-2_6 local inference and GPU optimization
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from config import Config
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

class MiniCPMV26Service:
    """Local GPU-powered MiniCPM-V-2_6 service for video analysis"""
    
    def __init__(self):
        self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.gpu_service = GPUService()
        self.performance_monitor = PerformanceMonitor()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the MiniCPM-V-2_6 model on GPU"""
        try:
            print(f"🚀 Initializing MiniCPM-V-2_6 on {self.device}...")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Check model path
            print(f"🔍 Model path: {Config.MINICPM_MODEL_PATH}")
            if not Config.MINICPM_MODEL_PATH:
                raise RuntimeError("MINICPM_MODEL_PATH is empty or None")
            
            # Initialize GPU service
            await self.gpu_service.initialize()
            
            # Load tokenizer
            print(f"📝 Loading tokenizer from {Config.MINICPM_MODEL_PATH}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    Config.MINICPM_MODEL_PATH,
                    trust_remote_code=True
                )
                
                # Verify tokenizer loaded successfully
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer failed to load - returned None")
                print(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
                
            except Exception as e:
                print(f"❌ Tokenizer loading failed: {e}")
                print(f"   Model path: {Config.MINICPM_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load tokenizer: {e}")
            
            # Load model with optimizations
            print(f"🤖 Loading model from {Config.MINICPM_MODEL_PATH}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    Config.MINICPM_MODEL_PATH,
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
                print(f"   Model path: {Config.MINICPM_MODEL_PATH}")
                print(f"   Error type: {type(e).__name__}")
                raise RuntimeError(f"Failed to load model: {e}")
            
            # Move to GPU
            print(f"🚀 Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            # Warm up the model
            self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V-2_6 initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V-2_6: {e}")
            raise
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V-2_6 model...")
        
        try:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer is None - cannot perform warmup")
            if self.model is None:
                raise RuntimeError("Model is None - cannot perform warmup")
            
            # Create dummy image and text for vision-language model warmup
            try:
                from PIL import Image
                import numpy as np
                
                # Create a 224x224 black dummy image
                dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                dummy_text = "Describe the image in one short word."
                
                print(f"🖼️  Created dummy image: {dummy_img.size}")
                print(f"📝 Using dummy text: '{dummy_text}'")
                
                # Use the processor to handle both image and text
                if hasattr(self.tokenizer, 'processor'):
                    processor = self.tokenizer.processor
                else:
                    # If no processor attribute, try to use the tokenizer directly
                    processor = self.tokenizer
                
                # Process image + text inputs
                inputs = processor(images=dummy_img, text=dummy_text, return_tensors="pt")
                if inputs is None:
                    raise RuntimeError("Processor returned None inputs")
                
                print(f"✅ Processing successful, input keys: {list(inputs.keys())}")
                
                # Move inputs to device
                inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                
                # Warm up the model with multiple forward passes
                with torch.no_grad():
                    for i in range(3):  # Run 3 warmup iterations
                        print(f"🔥 Warmup iteration {i+1}/3...")
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=8,  # Small output for warmup
                            do_sample=False,
                            pad_token_id=getattr(self.tokenizer, 'eos_token_id', 0),
                            eos_token_id=getattr(self.tokenizer, 'eos_token_id', 0)
                        )
                        
                        # Verify output is valid
                        if output is None or len(output) == 0:
                            raise RuntimeError(f"Warmup iteration {i+1} returned None or empty output")
                        print(f"  ✅ Warmup iteration {i+1} successful, output shape: {output.shape}")
                
                print("✅ Model warmup completed")
                
            except ImportError as e:
                print(f"⚠️  PIL not available, falling back to text-only warmup: {e}")
                # Fallback to text-only if PIL is not available
                self._warmup_model_text_only()
                
        except Exception as e:
            print(f"❌ Model warmup failed: {e}")
            print(f"   Tokenizer type: {type(self.tokenizer)}")
            print(f"   Model type: {type(self.model)}")
            raise
    
    def _warmup_model_text_only(self):
        """Fallback warmup method using text-only input"""
        print("📝 Using text-only warmup fallback...")
        
        dummy_text = "Hello, this is a warmup message."
        inputs = self.tokenizer(dummy_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            for i in range(3):
                print(f"🔥 Text warmup iteration {i+1}/3...")
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, 'eos_token_id', 0),
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', 0)
                )
                print(f"  ✅ Text warmup iteration {i+1} successful")
    
    def _extract_first_frame(self, video_path: str) -> Optional[Image.Image]:
        """Extract first frame from video for analysis"""
        try:
            import av
            
            container = av.open(video_path)
            for frame in container.decode(video=0):
                img = frame.to_image().convert("RGB")
                container.close()
                return img
                
        except ImportError:
            print("⚠️  PyAV not available for video frame extraction")
            return None
        except Exception as e:
            print(f"⚠️  Failed to extract video frame: {e}")
            return None
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using local GPU-powered MiniCPM-V-2_6"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Process video frames (simplified for now - will be enhanced with DeepStream)
            video_summary = self._extract_video_summary(video_path)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis using MiniCPM-V-2_6
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
        """Generate analysis using MiniCPM-V-2_6"""
        try:
            # Use the processor for proper tokenization
            if hasattr(self.tokenizer, 'processor'):
                processor = self.tokenizer.processor
            else:
                processor = self.tokenizer
            
            # Tokenize input (text-only for analysis)
            inputs = processor(
                text=prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=Config.MINICPM_CONFIG['max_length']
            )
            
            # Move to device
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,  # Reasonable output length
                    max_length=min(Config.MINICPM_CONFIG['max_length'], inputs['input_ids'].shape[1] + 1000),  # Total length limit
                    temperature=Config.MINICPM_CONFIG['temperature'],
                    top_p=Config.MINICPM_CONFIG['top_p'],
                    top_k=Config.MINICPM_CONFIG['top_k'],
                    do_sample=True,
                    pad_token_id=getattr(self.tokenizer, 'eos_token_id', 0),
                    eos_token_id=getattr(self.tokenizer, 'eos_token_id', 0)
                )
            
            # Verify outputs are valid
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Model generation returned None or empty output")
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated content
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
            
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
        print("🧹 MiniCPM-V-2_6 service cleaned up")

# Global instance
minicpm_service = MiniCPMV26Service()

# Round 2: All AI processing is now done locally with MiniCPM-V-2_6
# No external API dependencies required 