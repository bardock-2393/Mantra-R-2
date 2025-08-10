"""
MiniCPM-V 2.6 Model Management for Round 2
Handles model loading, GPU optimization, and inference for local AI processing
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import Config

class MiniCPMV26Model:
    """MiniCPM-V 2.6 model manager with GPU optimization"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_initialized = False
        self.model_path = Config.MINICPM_MODEL_PATH
        self.config = Config.MINICPM_CONFIG
        
    async def initialize(self):
        """Initialize the MiniCPM-V 2.6 model on GPU"""
        try:
            print(f"🚀 Initializing MiniCPM-V 2.6 on GPU...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Set device
            self.device = torch.device(Config.GPU_CONFIG['device'])
            print(f"📱 Using device: {self.device}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model path not found: {self.model_path}")
                print("🔍 Attempting to download from Hugging Face...")
                await self._download_model()
            
            # Load tokenizer
            print("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                token=Config.MINICPM_CONFIG.get('hf_token')
            )
            
            # Configure quantization
            quantization_config = self._get_quantization_config()
            
            # Load model with optimizations
            print("🤖 Loading model with GPU optimizations...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if Config.GPU_CONFIG['precision'] == 'float16' else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                use_flash_attention_2=self.config['use_flash_attention'],
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                token=Config.MINICPM_CONFIG.get('hf_token')
            )
            
            # Move to GPU if not already there
            if not hasattr(self.model, 'device_map'):
                self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Warm up the model
            await self._warmup_model()
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V 2.6 initialized successfully on {self.device}")
            
            # Print model info
            await self._print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V 2.6: {e}")
            raise
    
    def _get_quantization_config(self):
        """Get quantization configuration based on settings"""
        if self.config['quantization'] == 'int8':
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        elif self.config['quantization'] == 'int4':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            return None
    
    async def _download_model(self):
        """Download model from Hugging Face if not available locally"""
        try:
            print("📥 Downloading MiniCPM-V 2.6 from Hugging Face...")
            
            # This would download the model to the specified path
            # For now, we'll use a placeholder
            model_name = "openbmb/MiniCPM-V-2.6"
            
            print(f"📥 Model {model_name} would be downloaded here")
            print(f"📁 Target path: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Model download failed: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V 2.6 model...")
        
        try:
            # Create dummy input for warmup
            dummy_text = "Hello, this is a warmup message for the AI model."
            inputs = self.tokenizer(dummy_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                for i in range(3):  # Run 3 warmup iterations
                    start_time = time.time()
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    warmup_time = (time.time() - start_time) * 1000
                    print(f"🔥 Warmup iteration {i+1}: {warmup_time:.2f}ms")
            
            print("✅ Model warmup completed")
            
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
    
    async def _print_model_info(self):
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
    
    async def generate_text(self, prompt: str, max_new_tokens: int = None, 
                           temperature: float = None, top_p: float = None, 
                           top_k: int = None) -> str:
        """Generate text using MiniCPM-V 2.6"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Use default values if not specified
            max_new_tokens = max_new_tokens or self.config['max_length']
            temperature = temperature or self.config['temperature']
            top_p = top_p or self.config['top_p']
            top_k = top_k or self.config['top_k']
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config['max_length']
            ).to(self.device)
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = (time.time() - start_time) * 1000
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated content
            if prompt in response:
                response = response[len(prompt):].strip()
            
            print(f"⚡ Text generation completed in {generation_time:.2f}ms")
            
            return response
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return f"Error generating text: {str(e)}"
    
    async def analyze_video_content(self, video_summary: str, analysis_type: str, 
                                   user_focus: str) -> str:
        """Analyze video content using MiniCPM-V 2.6"""
        try:
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Combine prompt with video summary
            full_prompt = f"{analysis_prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis
            analysis_result = await self.generate_text(full_prompt)
            
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
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, 
                                   user_focus: str, message: str, 
                                   chat_history: List[Dict]) -> str:
        """Generate contextual AI response based on video analysis"""
        try:
            # Build context from chat history
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create chat prompt
            chat_prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            response = await self.generate_text(chat_prompt, max_new_tokens=2048)
            
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
    
    async def get_model_status(self) -> Dict:
        """Get model status and performance metrics"""
        try:
            status = {
                'initialized': self.is_initialized,
                'device': str(self.device) if self.device else None,
                'model_path': self.model_path,
                'config': self.config
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
    
    async def cleanup(self):
        """Clean up model resources"""
        try:
            print("🧹 Cleaning up MiniCPM-V 2.6 model...")
            
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
            print("✅ MiniCPM-V 2.6 model cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Model cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.is_initialized:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
        except:
            pass

# Global instance for easy access
minicpm_v26_model = MiniCPMV26Model() 