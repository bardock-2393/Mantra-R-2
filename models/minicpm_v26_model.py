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
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from config import Config

class MiniCPMV26Model:
    """MiniCPM-V-2_6 model manager with GPU optimization"""
    
    def __init__(self):
        self.model = None
        self.processor = None
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
            
            # Get HF token for MiniCPM-V-2_6 access
            hf_token = Config.MINICPM_CONFIG.get('hf_token', '')
            if not hf_token:
                print("⚠️ Warning: No HF_TOKEN provided. MiniCPM-V-2_6 may require authentication.")
            
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                attn_implementation='sdpa',  # Use SDPA for better performance
                torch_dtype=torch.bfloat16,
                token=hf_token if hf_token else None
            ).eval().cuda()
            
            # ✅ Load processor first (has .image_processor and .tokenizer)
            print(f"📝 Loading processor from {self.model_path}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                token=hf_token if hf_token else None
            )
            
            # Keep a tokenizer handle for fallbacks
            self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                token=hf_token if hf_token else None
            )
            
            # Verify components loaded
            if self.model is None or self.processor is None:
                raise RuntimeError("Failed to load model or processor")
            
            print(f"✅ Processor loaded successfully: {type(self.processor).__name__}")
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
            # MiniCPM-V-2_6 is a vision-language model, use proper chat interface
            print("📝 Using vision-language warmup with chat interface...")
            
            # Create a dummy image and text for warmup
            dummy_image = Image.new('RGB', (224, 224), color='blue')
            warmup_text = "Hello, this is a warmup message."
            
            # Format messages for the chat interface
            msgs = [{'role': 'user', 'content': [dummy_image, warmup_text]}]
            
            with torch.no_grad():
                for i in range(3):
                    print(f"  Warmup iteration {i+1}/3...")
                    try:
                        # ✅ Pass image and processor first
                        resp = self.model.chat(
                            image=dummy_image, 
                            msgs=msgs, 
                            processor=self.processor,
                            sampling=False, 
                            stream=False, 
                            max_new_tokens=8
                        )
                        print(f"    ✅ Warmup {i+1} successful: {str(resp)[:50]}...")
                    except TypeError:
                        # Fallback for older signature
                        resp = self.model.chat(
                            image=dummy_image, 
                            msgs=msgs, 
                            tokenizer=self.tokenizer,
                            sampling=False, 
                            stream=False, 
                            max_new_tokens=8
                        )
                        print(f"    ✅ Warmup {i+1} successful (fallback): {str(resp)[:50]}...")
                    except Exception as e:
                        print(f"    ❌ Warmup {i+1} failed: {e}")
                        # Continue with next iteration
                        continue
                        
            print("✅ Model warmup completed successfully")
                
        except Exception as e:
            print(f"❌ Model warmup failed: {e}")
            # Don't raise here, just log the error
            print(f"⚠️ Continuing without warmup...")
    
    def _print_model_info(self):
        """Print model information and memory usage"""
        try:
            if self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                print(f"📊 Model Parameters: {total_params:,}")
                print(f"📊 Trainable Parameters: {trainable_params:,}")
                
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"📊 GPU Memory Allocated: {memory_allocated:.2f} GB")
                    print(f"📊 GPU Memory Reserved: {memory_reserved:.2f} GB")
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not print model info: {e}")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.2, top_p: float = 0.9, 
                     top_k: int = 40) -> str:
        """Generate text using the MiniCPM-V-2_6 model"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Model not initialized")
            
            # MiniCPM-V-2_6 uses a chat interface, not generate
            # Create a dummy image as required by the vision-language model
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            # Format messages for the chat interface
            msgs = [{'role': 'user', 'content': [dummy_image, prompt]}]
            
            # Use the chat method as per official documentation
            with torch.no_grad():
                try:
                    # ✅ Primary path: processor
                    resp = self.model.chat(
                        image=dummy_image, 
                        msgs=msgs, 
                        processor=self.processor,
                        sampling=True, 
                        stream=False,
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature, 
                        top_p=top_p, 
                        top_k=top_k
                    )
                except TypeError:
                    # ✅ Fallback path: tokenizer
                    resp = self.model.chat(
                        image=dummy_image, 
                        msgs=msgs, 
                        tokenizer=self.tokenizer,
                        sampling=True, 
                        stream=False,
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature, 
                        top_p=top_p, 
                        top_k=top_k
                    )
            
            # Handle the response - it could be a string or generator
            if resp is None:
                raise RuntimeError("Model chat returned None")
            
            # If response is a string, return it directly
            if isinstance(resp, str):
                return resp
            
            # If response is a generator (streaming), collect the text
            if hasattr(resp, '__iter__') and not isinstance(resp, str):
                generated_text = ""
                for new_text in resp:
                    if new_text is not None:
                        generated_text += str(new_text)
                return generated_text if generated_text else "No text generated"
            
            # Fallback: convert to string
            return str(resp) if resp else "Text generation failed"
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return f"Error generating text: {str(e)}"
    
    def analyze_video_content(self, video_summary: str, analysis_type: str, 
                             user_focus: str) -> str:
        """Analyze video content using the model"""
        try:
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Combine with video summary
            full_prompt = f"{prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis
            analysis = self.generate_text(full_prompt, max_new_tokens=2048)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Video content analysis failed: {e}")
            return f"Error analyzing video content: {str(e)}"
    
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
    
    def generate_chat_response(self, analysis_result: str, analysis_type: str, 
                              user_focus: str, message: str, 
                              chat_history: List[Dict]) -> str:
        """Generate chat response using the model"""
        try:
            # Build chat context
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create full prompt
            full_prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            response = self.generate_text(full_prompt, max_new_tokens=1024, temperature=0.3)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"Error generating chat response: {str(e)}"
    
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
    
    def get_model_status(self) -> Dict:
        """Get model status and information"""
        try:
            status = {
                'model_name': 'MiniCPM-V-2_6',
                'initialized': self.is_initialized,
                'device': str(self.device) if self.device else None,
                'model_path': self.model_path
            }
            
            if self.is_initialized and self.model:
                total_params = sum(p.numel() for p in self.model.parameters())
                status['total_parameters'] = total_params
                
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    status['gpu_memory_allocated_gb'] = round(memory_allocated, 2)
                    status['gpu_memory_reserved_gb'] = round(memory_reserved, 2)
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up model resources"""
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            print("🧹 MiniCPM-V-2_6 model cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Model cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

# Global model instance
minicpm_v26_model = MiniCPMV26Model() 