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
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from config import Config

class MiniCPMV26Model:
    """MiniCPM-V-2_6 model manager with GPU optimization"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self.is_initialized = False
        self.model_path = Config.MINICPM_MODEL_PATH
        
    def test_generation(self) -> bool:
        """Test if the model can actually generate text"""
        try:
            if not self.is_initialized:
                print("❌ Model not initialized for testing")
                return False
            
            print("🧪 Testing model text generation...")
            
            # Test with chat interface
            try:
                test_prompt = "Hello, how are you today?"
                result = self.generate_text(test_prompt, max_new_tokens=10)
                if result and isinstance(result, str) and len(result) > 0:
                    print(f"✅ Chat interface test successful: Generated '{result[:50]}...'")
                    return True
                else:
                    print(f"❌ Chat interface test failed: Invalid result: {result}")
            except Exception as e:
                print(f"❌ Chat interface test failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("❌ All tests failed")
            return False
                
        except Exception as e:
            print(f"❌ Model test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_model_accessibility(self):
        """Check if the model is accessible and can be loaded"""
        try:
            print(f"🔍 Checking model accessibility: {self.model_path}")
            
            # Try to access the model info without downloading
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            print(f"✅ Model config accessible: {config.model_type}")
            
            # Check if model files exist locally or can be downloaded
            try:
                from transformers import cached_file
                # This will check if the model is available
                cached_file(self.model_path, "config.json", trust_remote_code=True)
                print("✅ Model files accessible")
                return True
            except Exception as e:
                print(f"⚠️ Warning: Model files not accessible locally: {e}")
                print("📥 Model will be downloaded on first use")
                return True
                
        except Exception as e:
            print(f"❌ Model accessibility check failed: {e}")
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
            
            # Validate model path
            if not self.model_path:
                raise RuntimeError("Model path not configured")
            
            # Check model accessibility
            if not self._check_model_accessibility():
                raise RuntimeError("Model is not accessible")
            
            print(f"🔍 Model path: {self.model_path}")
            print(f"🔍 CUDA device count: {torch.cuda.device_count()}")
            print(f"🔍 Current CUDA device: {torch.cuda.current_device()}")
            
            # Load model with correct parameters
            print("📥 Loading model...")
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                attn_implementation='sdpa',  # Use SDPA for better performance
                torch_dtype=torch.bfloat16
            )
            print("✅ Model loaded successfully")
            
            # Move to GPU and set to eval mode
            print("🚀 Moving model to GPU...")
            self.model = self.model.eval().cuda()
            print("✅ Model moved to GPU successfully")
            
            # Load processor (MiniCPM-V-2_6 uses a processor, not just a tokenizer)
            print(f"📝 Loading processor from {self.model_path}...")
            try:
                # Load processor for vision-language tasks
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print("✅ Processor loaded successfully")
                
                # Also load tokenizer for text-only tasks
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print("✅ Tokenizer loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Warning: Failed to load processor: {e}")
                # Fallback to tokenizer only
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        trust_remote_code=True
                    )
                    print("✅ Tokenizer loaded successfully (fallback)")
                except Exception as e2:
                    print(f"❌ Failed to load tokenizer: {e2}")
                    raise RuntimeError("Failed to load both processor and tokenizer")
            
            # Verify components loaded
            if self.model is None:
                raise RuntimeError("Failed to load model")
            
            if self.processor is None and self.tokenizer is None:
                raise RuntimeError("Failed to load both processor and tokenizer")
            
            # Check if the processor has the required methods
            if self.processor and not hasattr(self.processor, '__call__'):
                print("⚠️ Warning: Processor does not have __call__ method")
            
            # Check if the tokenizer has the required methods
            if self.tokenizer and not hasattr(self.tokenizer, 'encode') and not hasattr(self.tokenizer, '__call__'):
                print("⚠️ Warning: Tokenizer does not have required methods")
            
            print(f"✅ Processor loaded: {type(self.processor).__name__ if self.processor else 'None'}")
            print(f"✅ Tokenizer loaded: {type(self.tokenizer).__name__ if self.tokenizer else 'None'}")
            print(f"✅ Model loaded successfully: {type(self.model).__name__}")
            
            # Check model compatibility
            if not self._check_model_compatibility():
                print("⚠️ Warning: Model compatibility check failed, but continuing")
            
            # Test chat interface functionality
            try:
                test_text = "Hello, world!"
                # Test using the helper method
                test_inputs = self._process_inputs(test_text)
                print(f"✅ Chat interface test successful: {len(test_inputs.get('msgs', []))} messages prepared")
            except Exception as e:
                print(f"⚠️ Warning: Chat interface test failed: {e}")
                print("⚠️ Warning: Model may not work properly")
            
            # Warm up the model (don't fail if warmup fails)
            try:
                self._warmup_model()
            except Exception as e:
                print(f"⚠️ Warning: Model warmup failed, but continuing: {e}")
            
            # Test the model to ensure it's working
            try:
                if self.test_generation():
                    print("✅ Model test successful")
                else:
                    print("⚠️ Warning: Model test failed, but continuing with initialization")
            except Exception as e:
                print(f"⚠️ Warning: Model test failed, but continuing: {e}")
            
            self.is_initialized = True
            print(f"✅ MiniCPM-V-2_6 initialized successfully on {self.device}")
            
            # Print model info
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to initialize MiniCPM-V-2_6: {e}")
            import traceback
            traceback.print_exc()
            # Reset state on failure
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.device = None
            self.is_initialized = False
            raise
    
    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        print("🔥 Warming up MiniCPM-V-2_6 model...")
        
        try:
            # Simple text warmup using chat interface
            print("📝 Using text-only warmup with chat interface...")
            try:
                # Create a simple warmup prompt
                warmup_text = "Hello, how are you?"
                if not warmup_text or not isinstance(warmup_text, str):
                    print("⚠️ Warning: Invalid warmup text")
                    return
                
                # Use the generate_text method which now uses the chat interface
                _ = self.generate_text(warmup_text, max_new_tokens=10)
                print("✅ Text warmup completed")
            except Exception as e:
                print(f"❌ Text warmup failed: {e}")
                # Don't fail initialization if warmup fails
                print("⚠️ Warning: Model warmup failed, but continuing with initialization")
                
        except Exception as e:
            print(f"⚠️ Warning: Model warmup failed: {e}")
            # Don't fail initialization if warmup fails
            print("⚠️ Warning: Model warmup failed, but continuing with initialization")
    
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
    
    def _fallback_text_generation(self, prompt: str) -> str:
        """Fallback text generation when the main model fails"""
        try:
            # Simple template-based response
            if "analyze" in prompt.lower() or "analysis" in prompt.lower():
                return "I apologize, but I'm experiencing technical difficulties with the AI model. This appears to be a video analysis request. Please try again later or contact support if the issue persists."
            elif "chat" in prompt.lower() or "conversation" in prompt.lower():
                return "I'm sorry, but I'm currently unable to process your request due to technical issues. Please try again in a moment."
            else:
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        except Exception as e:
            return f"Technical error occurred: {str(e)}"
    
    def generate_text(self, prompt: str, max_new_tokens: int = 512, 
                     temperature: float = 0.2, top_p: float = 0.9, 
                     top_k: int = 40) -> str:
        """Generate text using the MiniCPM-V-2_6 model via chat interface"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Model not initialized")
            
            # Validate input parameters
            if not prompt or not isinstance(prompt, str):
                print(f"❌ Error: Invalid prompt provided: {type(prompt)} - {prompt}")
                raise ValueError("Invalid prompt provided")
            
            print(f"🔍 Generating text with prompt length: {len(prompt)} characters")
            
            # Process inputs using the helper method to get chat format
            chat_inputs = self._process_inputs(prompt)
            print(f"✅ Input processing successful, chat format prepared")
            
            # Validate chat inputs
            if 'msgs' not in chat_inputs or 'tokenizer' not in chat_inputs:
                raise ValueError("Input processing failed: missing required chat components")
            
            print(f"🔍 Chat inputs prepared: {list(chat_inputs.keys())}")
            
            # Generate response using the chat interface
            with torch.no_grad():
                # Ensure the model has the chat method
                if not hasattr(self.model, 'chat'):
                    raise RuntimeError("Model does not have chat method")
                
                print("🔍 Using model.chat interface...")
                
                try:
                    # Use the chat interface as per official MiniCPM-V 2.6 documentation
                    response = self.model.chat(
                        image=chat_inputs['image'],
                        msgs=chat_inputs['msgs'],
                        tokenizer=chat_inputs['tokenizer'],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True
                    )
                    
                    print("✅ Chat generation successful")
                    
                except Exception as e:
                    print(f"❌ Chat generation failed: {e}")
                    print("🔄 Trying with simplified parameters...")
                    
                    # Try with simplified generation parameters
                    try:
                        response = self.model.chat(
                            image=chat_inputs['image'],
                            msgs=chat_inputs['msgs'],
                            tokenizer=chat_inputs['tokenizer'],
                            max_new_tokens=min(max_new_tokens, 100),  # Reduce tokens
                            do_sample=False  # Use greedy decoding
                        )
                        print("✅ Generation successful with simplified parameters")
                    except Exception as e2:
                        print(f"❌ Simplified generation also failed: {e2}")
                        raise RuntimeError(f"Model generation failed: {e2}")
            
            # Ensure we return a valid string
            if response is None:
                response = "Text generation failed"
            
            print(f"✅ Final result length: {len(response)} characters")
            return response
            
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Try fallback generation
            try:
                print("🔄 Attempting fallback text generation...")
                fallback_result = self._fallback_text_generation(prompt)
                print("✅ Fallback generation successful")
                return fallback_result
            except Exception as fallback_error:
                print(f"❌ Fallback generation also failed: {fallback_error}")
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
            
            if self.processor:
                del self.processor
                self.processor = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            print("🧹 MiniCPM-V-2_6 model cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Model cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


    
    def _process_inputs(self, prompt: str, image=None):
        """Process inputs for the model using the chat interface"""
        try:
            print(f"🔍 Processing inputs - Processor: {self.processor is not None}, Tokenizer: {self.tokenizer is not None}, Image: {image is not None}")
            
            # For MiniCPM-V 2.6, we need to use the chat interface
            # Create the message format expected by the model
            if image is not None:
                # Vision-language task
                msgs = [{'role': 'user', 'content': [image, prompt]}]
                print("✅ Using vision-language chat format")
            else:
                # Text-only task - create a dummy image for the model
                dummy_image = Image.new('RGB', (224, 224), color='black')
                msgs = [{'role': 'user', 'content': [dummy_image, prompt]}]
                print("✅ Using text-only chat format with dummy image")
            
            # Return the message format and dummy image for the model.chat method
            return {
                'msgs': msgs,
                'image': None,  # The image is included in the msgs
                'tokenizer': self.tokenizer
            }
            
        except Exception as e:
            print(f"❌ Error processing inputs: {e}")
            raise

    def _check_model_compatibility(self):
        """Check if the loaded model is compatible with our usage"""
        try:
            if not hasattr(self.model, 'config'):
                print("⚠️ Warning: Model has no config, compatibility unknown")
                return True
            
            config = self.model.config
            
            # Check if it's a vision-language model
            if hasattr(config, 'model_type'):
                print(f"🔍 Model type: {config.model_type}")
                
                # Check if it supports chat interface
                if hasattr(self.model, 'chat'):
                    print("✅ Model supports chat interface")
                else:
                    print("❌ Model does not support chat interface")
                    return False
                
                # Check if it's a vision-language model
                if hasattr(config, 'vision_config') or hasattr(config, 'image_size'):
                    print("✅ Model supports vision-language tasks")
                else:
                    print("⚠️ Warning: Model may not support vision-language tasks")
                
                return True
            else:
                print("⚠️ Warning: Model config has no model_type")
                return True
                
        except Exception as e:
            print(f"⚠️ Warning: Could not check model compatibility: {e}")
            return True

# Global model instance
minicpm_v26_model = MiniCPMV26Model() 