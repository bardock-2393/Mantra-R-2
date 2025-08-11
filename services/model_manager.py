"""
Model Manager Service - OPTIMIZED FOR 7B MODEL ONLY
Handles Qwen2.5-VL-7B-Instruct model for MAXIMUM performance video analysis
"""

import asyncio
from typing import Dict, Optional
from config import Config
from services.ai_service import ai_service

class ModelManager:
    """Manages Qwen2.5-VL-7B-Instruct model for MAXIMUM performance"""
    
    def __init__(self):
        self.current_model = 'qwen25vl_7b'  # DEFAULT TO OPTIMIZED 7B MODEL
        self.available_models = {
            'qwen25vl_7b': {
                'name': 'Qwen2.5-VL-7B-Instruct (OPTIMIZED)',
                'description': 'OPTIMIZED 7B model with MAXIMUM performance, accuracy, and speed',
                'service': ai_service,
                'initialized': False
            }
        }
        self._initialization_lock = asyncio.Lock()  # Prevent concurrent initialization
    
    async def initialize_model(self, model_name: str = None) -> bool:
        """Initialize the specified model or current model"""
        async with self._initialization_lock:
            try:
                if model_name:
                    self.current_model = model_name
                
                if self.current_model not in self.available_models:
                    raise ValueError(f"Unknown model: {self.current_model}")
                
                model_info = self.available_models[self.current_model]
                
                # Check if already initialized
                if model_info['initialized']:
                    print(f"ℹ️ {model_info['name']} is already initialized")
                    return True
                
                print(f"🚀 Initializing {model_info['name']}...")
                
                # Initialize the service
                await model_info['service'].initialize()
                model_info['initialized'] = True
                
                print(f"✅ {model_info['name']} initialized successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to initialize {self.current_model}: {e}")
                # Mark as not initialized on failure
                if self.current_model in self.available_models:
                    self.available_models[self.current_model]['initialized'] = False
                return False
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model (only 7B model available)"""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Unknown model: {model_name}. Only Qwen2.5-VL-7B is available.")
            
            if model_name == self.current_model:
                print(f"ℹ️ Already using {model_name}")
                return True
            
            print(f"🔄 Switching from {self.current_model} to {model_name}...")
            
            # Store the previous model name
            previous_model = self.current_model
            
            # Cleanup current model if initialized
            if self.available_models[self.current_model]['initialized']:
                try:
                    self.available_models[self.current_model]['service'].cleanup()
                    self.available_models[self.current_model]['initialized'] = False
                    print(f"🧹 Cleaned up {self.available_models[self.current_model]['name']}")
                except Exception as e:
                    print(f"⚠️ Warning: Cleanup failed for {self.current_model}: {e}")
            
            # Switch to new model
            self.current_model = model_name
            
            # Initialize the new model
            success = await self.initialize_model()
            
            if success:
                print(f"✅ Successfully switched to {model_name}")
                return True
            else:
                # Revert to previous model on failure
                print(f"❌ Failed to initialize {model_name}, reverting to {previous_model}")
                self.current_model = previous_model
                return False
                
        except Exception as e:
            print(f"❌ Model switch failed: {e}")
            return False
    
    def get_current_model(self) -> Dict:
        """Get information about the currently active model"""
        if self.current_model in self.available_models:
            model_info = self.available_models[self.current_model].copy()
            model_info['current'] = True
            return model_info
        return {}
    
    def get_available_models(self) -> Dict:
        """Get information about all available models"""
        return {
            name: {
                'name': info['name'],
                'description': info['description'],
                'current': name == self.current_model,
                'initialized': info['initialized']
            }
            for name, info in self.available_models.items()
        }
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using the current model"""
        try:
            if self.current_model not in self.available_models:
                raise RuntimeError("No model available for analysis")
            
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                await self.initialize_model()
            
            return await model_info['service'].analyze_video(video_path, analysis_type, user_focus)
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return f"❌ Analysis failed: {str(e)}"
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: list) -> str:
        """Generate chat response using the current model"""
        try:
            if self.current_model not in self.available_models:
                raise RuntimeError("No model available for chat")
            
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                await self.initialize_model()
            
            return await model_info['service'].generate_chat_response(
                analysis_result, analysis_type, user_focus, message, chat_history
            )
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"❌ Failed to generate response: {str(e)}"
    
    def get_status(self) -> Dict:
        """Get overall model manager status"""
        return {
            'current_model': self.current_model,
            'available_models': list(self.available_models.keys()),
            'initialized_models': [
                name for name, info in self.available_models.items() 
                if info['initialized']
            ]
        }
    
    def cleanup(self):
        """Clean up all models"""
        try:
            for model_name, model_info in self.available_models.items():
                if model_info['initialized']:
                    try:
                        model_info['service'].cleanup()
                        model_info['initialized'] = False
                        print(f"🧹 Cleaned up {model_info['name']}")
                    except Exception as e:
                        print(f"⚠️ Warning: Cleanup failed for {model_name}: {e}")
            
            print("🧹 All models cleaned up")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    async def health_check(self) -> Dict:
        """Check health of all models"""
        health_status = {
            'overall_status': 'healthy',
            'models': {},
            'timestamp': asyncio.get_event_loop().time()
        }
        
        try:
            for model_name, model_info in self.available_models.items():
                model_health = {
                    'name': model_info['name'],
                    'initialized': model_info['initialized'],
                    'status': 'healthy' if model_info['initialized'] else 'not_initialized'
                }
                
                # Get detailed status from service if available
                if hasattr(model_info['service'], 'get_status'):
                    try:
                        service_status = model_info['service'].get_status()
                        model_health.update(service_status)
                    except Exception as e:
                        model_health['status'] = 'error'
                        model_health['error'] = str(e)
                
                health_status['models'][model_name] = model_health
                
                # Update overall status
                if model_health['status'] != 'healthy':
                    health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            return health_status

# Create global model manager instance
model_manager = ModelManager() 