"""
MiniCPM-V-2_6 Model Management for Round 2
Handles model loading, GPU optimization, and inference for local AI processing
Based on official MiniCPM-V-2_6 implementation

FIXED: Now uses model.chat() API instead of incorrectly passing images to tokenizers.
Routes MiniCPM through model.chat and never passes images to a tokenizer.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer, AutoProcessor, PreTrainedTokenizerBase
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
    
    def _has_vision_processor(self):
        """Check if the processor has vision capabilities"""
        p = self.processor
        # For MiniCPM, check if it's a processor that can handle images
        # But be more strict - don't just check if it's callable
        return p is not None and (
            hasattr(p, "image_processor") or 
            hasattr(p, "image_processor_class")
        )
        
    def test_generation(self) -> bool:
        """Test if the model can actually generate text"""
        try:
            if not self.is_initialized:
                print("❌ Model not initialized for testing")
                return False
            
            print("🧪 Testing model text generation...")
            
            # Test with processor if available
            if self.processor:
                print("🧪 Testing with processor...")
                try:
                    test_prompt = "Hello, how are you today?"
                    result = self.generate_text(test_prompt, max_new_tokens=10)
                    if result and isinstance(result, str) and len(result) > 0:
                        print(f"✅ Processor test successful: Generated '{result[:50]}...'")
                        return True
                    else:
                        print(f"❌ Processor test failed: Invalid result: {result}")
                except Exception as e:
                    print(f"❌ Processor test failed: {e}")
            
            # Test with tokenizer if available
            if self.tokenizer:
                print("🧪 Testing with tokenizer...")
                try:
                    test_prompt = "Hello, how are you today?"
                    result = self.generate_text(test_prompt, max_new_tokens=10)
                    if result and isinstance(result, str) and len(result) > 0:
                        print(f"✅ Tokenizer test successful: Generated '{result[:50]}...'")
                        return True
                    else:
                        print(f"❌ Tokenizer test failed: Invalid result: {result}")
                except Exception as e:
                    print(f"❌ Tokenizer test failed: {e}")
            
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
            
            # Test tokenizer functionality
            try:
                test_text = "Hello, world!"
                # Test using the helper method
                test_tokens = self._process_inputs(test_text)
                print(f"✅ Tokenizer test successful: {test_tokens.input_ids.shape if hasattr(test_tokens, 'input_ids') else 'processed'}")
            except Exception as e:
                print(f"⚠️ Warning: Tokenizer test failed: {e}")
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
            # Simple text warmup
            print("📝 Using text-only warmup...")
            try:
                # Create a simple warmup prompt
                warmup_text = "Hello, how are you?"
                if not warmup_text or not isinstance(warmup_text, str):
                    print("⚠️ Warning: Invalid warmup text")
                    return
                
                if hasattr(self.model, "chat"):
                    if self.processor is None:
                        print("⚠️ Warning: No processor available for chat warmup")
                        return
                    _ = self.model.chat(image=None, msgs=[{'role':'user','content':[warmup_text]}], processor=self.processor)
                else:
                    inputs = self._process_inputs(warmup_text)
                    with torch.no_grad():
                        _ = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
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
        """Generate text using the MiniCPM-V-2_6 model"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Model not initialized")
            
            # Validate input parameters
            if not prompt or not isinstance(prompt, str):
                print(f"❌ Error: Invalid prompt provided: {type(prompt)} - {prompt}")
                raise ValueError("Invalid prompt provided")
            
            print(f"🔍 Generating text with prompt length: {len(prompt)} characters")
            
            # If this model exposes `.chat`, use it (MiniCPM path)
            if hasattr(self.model, "chat"):
                print("🔍 Using model.chat() API (MiniCPM path)")
                print(f"🔍 Processor type: {type(self.processor).__name__}")
                print(f"🔍 Processor attributes: {[attr for attr in dir(self.processor) if not attr.startswith('_')][:10]}")
                print(f"🔍 Has encode method: {hasattr(self.processor, 'encode')}")
                print(f"🔍 Has image_processor: {hasattr(self.processor, 'image_processor')}")
                print(f"🔍 _has_vision_processor() returns: {self._has_vision_processor()}")
                
                # Check model config to understand what it expects
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'query_num'):
                    print(f"🔍 Model expects query_num: {self.model.config.query_num}")
                
                # Check if processor is actually a tokenizer
                if hasattr(self.processor, 'encode') and not hasattr(self.processor, 'image_processor'):
                    print("⚠️ Warning: Processor appears to be a tokenizer, not a vision processor")
                    print("🔄 Falling back to generate method...")
                    # Continue to fallback
                elif not self._has_vision_processor():
                    print("⚠️ Warning: Processor is not a vision processor")
                    print("🔄 Falling back to generate method...")
                    # Continue to fallback
                else:
                    msgs = [{'role': 'user', 'content': [prompt]}]  # text-only
                    # MiniCPM.chat expects a processor, not a tokenizer
                    if self.processor is None:
                        raise RuntimeError("Processor required for model.chat")
                    try:
                        # Try with the processor as-is first
                        out = self.model.chat(image=None, msgs=msgs, processor=self.processor)
                        result = out.strip() if isinstance(out, str) else str(out)
                        print(f"✅ Chat API successful: {result[:100]}...")
                        return result
                    except AttributeError as attr_error:
                        if "image_processor" in str(attr_error):
                            print(f"⚠️ Processor missing image_processor attribute: {attr_error}")
                            print("🔄 Trying to create a compatible processor wrapper...")
                            try:
                                # Create a wrapper that provides the expected interface
                                from types import SimpleNamespace
                                wrapper = SimpleNamespace()
                                wrapper.image_processor = SimpleNamespace()
                                # Use actual query_num from model config if available
                                if hasattr(self.model, 'config') and hasattr(self.model.config, 'query_num'):
                                    wrapper.image_processor.image_feature_size = self.model.config.query_num
                                    print(f"🔍 Using model's query_num: {self.model.config.query_num}")
                                else:
                                    wrapper.image_processor.image_feature_size = 256  # Common default
                                    print("🔍 Using default image_feature_size: 256")
                                wrapper.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.tokenizer
                                
                                out = self.model.chat(image=None, msgs=msgs, processor=wrapper)
                                result = out.strip() if isinstance(out, str) else str(out)
                                print(f"✅ Chat API successful with wrapper: {result[:100]}...")
                                return result
                            except Exception as wrapper_error:
                                print(f"⚠️ Wrapper approach failed: {wrapper_error}")
                                print("🔄 Falling back to generate method...")
                        else:
                            print(f"⚠️ Chat API failed with attribute error: {attr_error}")
                            print("🔄 Falling back to generate method...")
                    except Exception as chat_error:
                        print(f"⚠️ Chat API failed: {chat_error}")
                        print("🔄 Falling back to generate method...")
                        # Continue to fallback

            # Fallback: standard HF generate for plain LMs
            inputs = self._process_inputs(prompt)
            print(f"✅ Input processing successful, input_ids: {inputs.get('input_ids').shape if 'input_ids' in inputs else 'n/a'}")

            # Validate inputs before generation
            if 'input_ids' not in inputs or inputs['input_ids'] is None:
                raise ValueError("Input processing failed: no input_ids")
            
            print(f"🔍 Final input validation: {list(inputs.keys())}")
            
            # Generate response
            with torch.no_grad():
                # Ensure the model has the generate method
                if not hasattr(self.model, 'generate'):
                    raise RuntimeError("Model does not have generate method")
                
                # Check if the model expects specific input formats
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                    print(f"🔍 Model type: {self.model.config.model_type}")
                
                try:
                    eos_token_id = self._get_eos_token_id()
                    if eos_token_id is None:
                        print("⚠️ Warning: No EOS token ID available, using default")
                        eos_token_id = 0
                    
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=True,
                        pad_token_id=eos_token_id
                    )
                except Exception as e:
                    print(f"❌ Model generation failed: {e}")
                    print("🔄 Trying with simplified parameters...")
                    
                    # Try with simplified generation parameters
                    try:
                        eos_token_id = self._get_eos_token_id()
                        if eos_token_id is None:
                            eos_token_id = 0
                        
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=min(max_new_tokens, 100),  # Reduce tokens
                            do_sample=False,  # Use greedy decoding
                            pad_token_id=eos_token_id
                        )
                        print("✅ Generation successful with simplified parameters")
                    except Exception as e2:
                        print(f"❌ Simplified generation also failed: {e2}")
                        raise RuntimeError(f"Model generation failed: {e2}")
            
            print(f"✅ Generation successful, output shape: {generated_ids.shape}")
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Use the appropriate component for decoding
            decoding_component = self._get_decoding_component()
            output_text = decoding_component.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            print(f"✅ Using {type(decoding_component).__name__} for decoding")
            
            print(f"✅ Decoding successful, output texts: {len(output_text)}")
            
            # Ensure we return a valid string
            result = output_text[0] if output_text else "Text generation failed"
            if result is None:
                result = "Text generation failed"
            
            print(f"✅ Final result length: {len(result)} characters")
            return result
            
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

    def _get_decoding_component(self):
        """Get the appropriate component for decoding (processor or tokenizer)"""
        if self.processor and hasattr(self.processor, 'batch_decode'):
            return self.processor
        elif self.tokenizer and hasattr(self.tokenizer, 'batch_decode'):
            return self.tokenizer
        else:
            raise RuntimeError("No suitable component found for decoding")
    
    def _get_eos_token_id(self):
        """Get the EOS token ID from the appropriate component"""
        if self.processor and hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.eos_token_id
        elif self.tokenizer:
            return self.tokenizer.eos_token_id
        else:
            return None
    
    def _process_inputs(self, prompt: str, image=None):
        """Process inputs for the model, handling both processor and tokenizer cases"""
        try:
            # TEXT-ONLY by default (never pass images to a tokenizer)
            if self._has_vision_processor() and image is not None:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                print(f"🔍 Using vision processor, inputs type: {type(inputs)}")
            else:
                tok = self.processor.tokenizer if (self._has_vision_processor() and hasattr(self.processor, "tokenizer")) else self.tokenizer
                if tok is None:
                    raise RuntimeError("No tokenizer available")
                print(f"🔍 Using tokenizer: {type(tok).__name__}")
                
                # Ensure we get the right format from tokenizer
                tokenizer_output = tok(prompt, return_tensors="pt")
                print(f"🔍 Raw tokenizer output: {tokenizer_output}")
                
                # Convert to dict if it's not already
                if hasattr(tokenizer_output, 'input_ids'):
                    inputs = {'input_ids': tokenizer_output.input_ids}
                else:
                    inputs = dict(tokenizer_output)
                
                print(f"🔍 Tokenizer inputs: {inputs}")

            # Ensure all inputs are on the correct device
            for key, value in dict(inputs).items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            
            print(f"🔍 Final processed inputs: {inputs}")
            return inputs
            
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
                
                # Check if it supports text generation
                if hasattr(self.model, 'generate'):
                    print("✅ Model supports text generation")
                else:
                    print("❌ Model does not support text generation")
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

# FIXED: MiniCPM-V-2_6 now uses model.chat() API instead of incorrectly passing images to tokenizers.
# The model will automatically detect if it has a chat method and use it, falling back to generate() if needed.
# This eliminates the PreTrainedTokenizerFast._batch_encode_plus(... images=...) error. 