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
        
    def _validate_model_state(self) -> bool:
        """Validate that the model is in a working state"""
        try:
            if not self.is_initialized:
                print("❌ Model not initialized")
                return False
            
            if self.model is None:
                print("❌ Model object is None")
                return False
            
            if self.tokenizer is None:
                print("❌ Tokenizer object is None")
                return False
            
            if self.device is None:
                print("❌ Device is None")
                return False
            
            # Test basic functionality
            try:
                test_text = "test"
                test_inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
                
                if test_inputs.input_ids is None or len(test_inputs.input_ids) == 0:
                    print("❌ Tokenizer not producing valid output")
                    return False
                
                print("✅ Model state validation passed")
                return True
                
            except Exception as e:
                print(f"❌ Model functionality test failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
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
            
            # Load model with memory optimization
            print(f"🔍 Model path: {self.model_path}")
            
            # Try different loading strategies based on available memory
            loading_strategies = [
                self._load_model_full_precision,
                self._load_model_8bit,
                self._load_model_4bit,
                self._load_model_cpu_fallback
            ]
            
            model_loaded = False
            for strategy in loading_strategies:
                try:
                    print(f"🔄 Trying loading strategy: {strategy.__name__}")
                    strategy()
                    model_loaded = True
                    print(f"✅ Strategy {strategy.__name__} succeeded")
                    break
                except Exception as e:
                    print(f"⚠️ Strategy {strategy.__name__} failed: {e}")
                    if strategy == loading_strategies[-1]:  # Last strategy
                        raise RuntimeError(f"All loading strategies failed: {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load model with any strategy")
            
            # Load tokenizer
            print(f"📝 Loading processor from {self.model_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
                raise
            
            # Verify components loaded
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer")
            
            print(f"✅ Model loaded successfully: {type(self.model).__name__}")
            
            # Warm up the model
            self._warmup_model()
            
            # Validate model state
            if not self._validate_model_state():
                raise RuntimeError("Model validation failed after initialization")
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V-2_6 initialized successfully on {self.device}")
            
            # Print model info
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V-2_6: {e}")
            # Reset state on failure
            self.is_initialized = False
            self.model = None
            self.tokenizer = None
            raise
    
    def _load_model_full_precision(self):
        """Load model with full precision (highest quality, highest memory usage)"""
        print("📥 Loading model with full precision...")
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=Config.MINICPM_CONFIG['low_cpu_mem_usage'],
            device_map=Config.MINICPM_CONFIG['device_map']
        )
        
        # Move to GPU and set to eval mode
        if self.device.type == 'cuda':
            self.model = self.model.eval().cuda()
        else:
            self.model = self.model.eval()
    
    def _load_model_8bit(self):
        """Load model with 8-bit quantization (reduced memory usage)"""
        print("📥 Loading model with 8-bit quantization...")
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=Config.MINICPM_CONFIG['low_cpu_mem_usage'],
                device_map=Config.MINICPM_CONFIG['device_map']
            )
            
            if self.device.type == 'cuda':
                self.model = self.model.eval().cuda()
            else:
                self.model = self.model.eval()
                
        except ImportError:
            raise RuntimeError("BitsAndBytes not available for 8-bit quantization")
    
    def _load_model_4bit(self):
        """Load model with 4-bit quantization (lowest memory usage)"""
        print("📥 Loading model with 4-bit quantization...")
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=Config.MINICPM_CONFIG['low_cpu_mem_usage'],
                device_map=Config.MINICPM_CONFIG['device_map']
            )
            
            if self.device.type == 'cuda':
                self.model = self.model.eval().cuda()
            else:
                self.model = self.model.eval()
                
        except ImportError:
            raise RuntimeError("BitsAndBytes not available for 4-bit quantization")
    
    def _load_model_cpu_fallback(self):
        """Load model on CPU as last resort"""
        print("📥 Loading model on CPU (fallback)...")
        self.device = torch.device('cpu')
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=Config.MINICPM_CONFIG['low_cpu_mem_usage']
        )
        self.model = self.model.eval()
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V-2_6 model...")
        
        try:
            # Validate model and tokenizer
            if self.model is None:
                print("❌ Model is None during warmup")
                return
            if self.tokenizer is None:
                print("❌ Tokenizer is None during warmup")
                return
            
            # Simple text warmup
            print("📝 Using text-only warmup...")
            try:
                # Create a simple warmup prompt
                warmup_text = "Hello, how are you?"
                print(f"🔍 Warmup prompt: {warmup_text}")
                
                # Test tokenization first
                try:
                    warmup_inputs = self.tokenizer(warmup_text, return_tensors="pt").to(self.device)
                    print(f"✅ Warmup tokenization successful: {warmup_inputs.input_ids.shape}")
                except Exception as e:
                    print(f"❌ Warmup tokenization failed: {e}")
                    return
                
                # Test generation
                with torch.no_grad():
                    warmup_output = self.model.generate(
                        **warmup_inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                if warmup_output is not None:
                    print(f"✅ Warmup generation successful: {warmup_output.shape}")
                    
                    # Test decoding
                    try:
                        warmup_decoded = self.tokenizer.batch_decode(
                            warmup_output,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        if warmup_decoded and len(warmup_decoded) > 0:
                            print(f"✅ Warmup decoding successful: {warmup_decoded[0]}")
                        else:
                            print("⚠️ Warmup decoding returned empty result")
                    except Exception as e:
                        print(f"⚠️ Warmup decoding failed: {e}")
                else:
                    print("❌ Warmup generation returned None")
                    
                print("✅ Text warmup completed")
                
            except Exception as e:
                print(f"❌ Text warmup failed: {e}")
                # Don't raise the error, just log it
                
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
            # Don't raise the error, just log it
    
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
    
    def is_ready_for_inference(self) -> bool:
        """Check if the model is ready for inference"""
        try:
            if not self.is_initialized:
                return False
            
            if self.model is None or self.tokenizer is None:
                return False
            
            # Quick validation test
            return self._validate_model_state()
            
        except Exception as e:
            print(f"❌ Inference readiness check failed: {e}")
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.2, top_p: float = 0.9, 
                     top_k: int = 40) -> str:
        """Generate text using the MiniCPM-V-2_6 model"""
        try:
            # Check if model is ready
            if not self.is_ready_for_inference():
                return "Error: Model is not ready for inference. Please reinitialize the model."
            
            if not prompt or not prompt.strip():
                return "Error: Empty or invalid prompt provided"
            
            print(f"🔍 Generating text with prompt length: {len(prompt)}")
            print(f"📝 Prompt preview: {prompt[:100]}...")
            
            # Tokenize input
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                print(f"✅ Tokenization successful: {inputs.input_ids.shape}")
            except Exception as e:
                print(f"❌ Tokenization failed: {e}")
                return f"Error during tokenization: {str(e)}"
            
            # Generate response
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                print(f"✅ Generation successful: {generated_ids.shape}")
            except Exception as e:
                print(f"❌ Text generation failed: {e}")
                return f"Error during text generation: {str(e)}"
            
            # Validate generated output
            if generated_ids is None:
                print("❌ Generated IDs is None")
                return "Error: Model returned no output"
            
            # Decode response with safety checks
            try:
                # Ensure we have valid input and output tensors
                if inputs.input_ids is None or generated_ids is None:
                    print("❌ Input or output tensors are None")
                    return "Error: Invalid tensor data"
                
                # Check tensor shapes
                if len(inputs.input_ids) == 0 or len(generated_ids) == 0:
                    print("❌ Empty input or output tensors")
                    return "Error: Empty tensor data"
                
                # Safely trim generated IDs
                generated_ids_trimmed = []
                for i, (in_ids, out_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
                    if in_ids is not None and out_ids is not None:
                        try:
                            trimmed = out_ids[len(in_ids):]
                            generated_ids_trimmed.append(trimmed)
                        except Exception as e:
                            print(f"❌ Error trimming tensor {i}: {e}")
                            continue
                
                if not generated_ids_trimmed:
                    print("❌ No valid trimmed tensors")
                    return "Error: Failed to process generated output"
                
                # Decode the trimmed tensors
                output_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                print(f"✅ Decoding successful: {len(output_text)} outputs")
                
                # Validate decoded output
                if not output_text or len(output_text) == 0:
                    print("❌ Empty decoded output")
                    return "Error: No text was generated"
                
                # Return the first valid output
                result = output_text[0] if output_text else "Text generation failed"
                
                if not result or result.strip() == "":
                    print("❌ Empty result text")
                    return "Error: Generated text is empty"
                
                print(f"✅ Text generation completed: {len(result)} characters")
                return result
                
            except Exception as e:
                print(f"❌ Decoding failed: {e}")
                return f"Error during text decoding: {str(e)}"
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            return f"Error generating text: {str(e)}"
    
    def analyze_video_content(self, video_summary: str, analysis_type: str, 
                             user_focus: str) -> str:
        """Analyze video content using the model"""
        try:
            # Check if model is ready
            if not self.is_ready_for_inference():
                return "Error: Model is not ready for inference. Please reinitialize the model."
            
            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(analysis_type, user_focus)
            
            # Combine with video summary
            full_prompt = f"{prompt}\n\nVideo Summary:\n{video_summary}\n\nAnalysis:"
            
            # Generate analysis
            analysis = self.generate_text(full_prompt, max_new_tokens=2048)
            
            # Validate the result
            if analysis.startswith("Error:"):
                print(f"❌ Analysis generation failed: {analysis}")
                return f"Failed to analyze video content: {analysis}"
            
            return analysis
            
        except Exception as e:
            print(f"❌ Video content analysis failed: {e}")
            return f"Error analyzing video content: {str(e)}"
    
    def _generate_analysis_prompt(self, analysis_type: str, user_focus: str) -> str:
        """Generate analysis prompt based on type and user focus"""
        try:
            base_prompts = {
                'general': "Analyze this video and provide a comprehensive overview.",
                'behavioral': "Analyze the behavior patterns and actions in this video.",
                'technical': "Provide a technical analysis of this video content.",
                'narrative': "Analyze the narrative structure and storytelling elements.",
                'forensic': "Conduct a forensic analysis of this video for evidence.",
                'commercial': "Analyze this video from a commercial and marketing perspective."
            }
            
            base_prompt = base_prompts.get(analysis_type, base_prompts['general'])
            
            if user_focus and user_focus.strip():
                return f"{base_prompt} Focus specifically on: {user_focus}"
            return base_prompt
            
        except Exception as e:
            print(f"❌ Failed to generate analysis prompt: {e}")
            return "Analyze this video and provide a comprehensive overview."
    
    def generate_chat_response(self, analysis_result: str, analysis_type: str, 
                              user_focus: str, message: str, 
                              chat_history: List[Dict]) -> str:
        """Generate chat response using the model"""
        try:
            # Check if model is ready
            if not self.is_ready_for_inference():
                return "Error: Model is not ready for inference. Please reinitialize the model."
            
            # Build chat context
            context = self._build_chat_context(analysis_result, analysis_type, user_focus, chat_history)
            
            # Create full prompt
            full_prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response
            response = self.generate_text(full_prompt, max_new_tokens=1024, temperature=0.3)
            
            # Validate the result
            if response.startswith("Error:"):
                print(f"❌ Chat response generation failed: {response}")
                return f"Failed to generate chat response: {response}"
            
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
    
    def safe_reinitialize(self) -> bool:
        """Safely reinitialize the model if it's in a bad state"""
        try:
            print("🔄 Attempting to reinitialize the model...")
            
            # Clean up current state
            self.cleanup()
            
            # Wait a bit for cleanup to take effect
            import time
            time.sleep(2)
            
            # Try to initialize again
            self.initialize()
            
            if self.is_ready_for_inference():
                print("✅ Model reinitialization successful")
                return True
            else:
                print("❌ Model reinitialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Model reinitialization failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            print("🧹 Cleaning up MiniCPM-V-2_6 model...")
            
            # Clear GPU memory if using CUDA
            if self.device and self.device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"⚠️ GPU cleanup warning: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset state
            self.is_initialized = False
            self.model = None
            self.tokenizer = None
            self.device = None
            
            print("✅ Model cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Warning: Model cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def get_model_status(self) -> Dict:
        """Get detailed model status information"""
        try:
            status = {
                'model_name': 'MiniCPM-V-2_6',
                'initialized': self.is_initialized,
                'model_loaded': self.model is not None,
                'tokenizer_loaded': self.tokenizer is not None,
                'device': str(self.device) if self.device else None,
                'ready_for_inference': self.is_ready_for_inference() if self.is_initialized else False,
                'model_path': self.model_path
            }
            
            if self.is_initialized and self.model:
                try:
                    total_params = sum(p.numel() for p in self.model.parameters())
                    status['total_parameters'] = total_params
                except Exception as e:
                    status['total_parameters'] = f"Error: {e}"
                
                if self.device and self.device.type == 'cuda':
                    try:
                        memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                        status['gpu_memory_allocated_gb'] = round(memory_allocated, 2)
                        status['gpu_memory_reserved_gb'] = round(memory_reserved, 2)
                    except Exception as e:
                        status['gpu_memory_error'] = str(e)
            
            return status
            
        except Exception as e:
            return {
                'error': f"Failed to get status: {e}",
                'initialized': False
            }

# Global model instance
minicpm_v26_model = MiniCPMV26Model() 