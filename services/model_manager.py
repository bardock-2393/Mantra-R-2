"""
Model Manager Service
Handles switching between different AI models for video analysis
"""

import asyncio
from typing import Dict, Optional
from config import Config
from services.ai_service_fixed import minicpm_service
from services.qwen25vl_service import qwen25vl_service

class ModelManager:
    """Manages different AI models and provides unified interface"""
    
    def __init__(self):
        self.current_model = 'minicpm'  # Default model
        self.available_models = {
            'minicpm': {
                'name': 'MiniCPM-V-2_6',
                'description': 'Fast, efficient vision-language model',
                'service': minicpm_service,
                'initialized': False
            },
            'qwen25vl': {
                'name': 'Qwen2.5-VL-7B-Instruct',
                'description': 'Advanced multimodal model with enhanced video understanding',
                'service': qwen25vl_service,
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
        """Switch to a different model"""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Unknown model: {model_name}")
            
            if model_name == self.current_model:
                print(f"ℹ️ Already using {model_name}")
                return True
            
            print(f"🔄 Switching from {self.current_model} to {model_name}...")
            
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
            
            # Initialize new model
            success = await self.initialize_model()
            if success:
                print(f"✅ Successfully switched to {model_name}")
                return True
            else:
                print(f"❌ Failed to switch to {model_name}")
                # Revert to previous model
                previous_model = self.current_model
                self.current_model = 'minicpm'
                print(f"🔄 Reverting to {self.current_model}")
                await self.initialize_model()
                return False
            
        except Exception as e:
            print(f"❌ Model switch failed: {e}")
            # Try to revert to minicpm on failure
            try:
                self.current_model = 'minicpm'
                await self.initialize_model()
            except Exception as revert_error:
                print(f"❌ Failed to revert to minicpm: {revert_error}")
            return False
    
    def get_current_model(self) -> Dict:
        """Get information about the current model"""
        if self.current_model not in self.available_models:
            return {}
        
        model_info = self.available_models[self.current_model]
        return {
            'name': model_info['name'],
            'description': model_info['description'],
            'initialized': model_info['initialized'],
            'status': model_info['service'].get_status() if model_info['initialized'] else {}
        }
    
    def get_available_models(self) -> Dict:
        """Get information about all available models"""
        return {
            name: {
                'name': info['name'],
                'description': info['description'],
                'initialized': info['initialized'],
                'current': name == self.current_model
            }
            for name, info in self.available_models.items()
        }
    
    async def analyze_video(self, video_path: str, analysis_type: str, user_focus: str) -> str:
        """Analyze video using the current model"""
        try:
            if not self.available_models[self.current_model]['initialized']:
                print(f"🔄 Initializing {self.current_model} for video analysis...")
                await self.initialize_model()
            
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                raise RuntimeError(f"Failed to initialize {self.current_model}")
            
            return await model_info['service'].analyze_video(video_path, analysis_type, user_focus)
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            raise RuntimeError(f"Video analysis failed: {e}")
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: list) -> str:
        """Generate chat response using the current model"""
        try:
            if not self.available_models[self.current_model]['initialized']:
                print(f"🔄 Initializing {self.current_model} for chat response...")
                await self.initialize_model()
            
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                raise RuntimeError(f"Failed to initialize {self.current_model}")
            
            return await model_info['service'].generate_chat_response(
                analysis_result, analysis_type, user_focus, message, chat_history
            )
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            raise RuntimeError(f"Chat response generation failed: {e}")
    
    def get_status(self) -> Dict:
        """Get overall status of the model manager"""
        return {
            'current_model': self.current_model,
            'available_models': self.get_available_models(),
            'current_model_status': self.get_current_model()
        }
    
    def cleanup(self):
        """Cleanup all models"""
        try:
            for name, model_info in self.available_models.items():
                if model_info['initialized']:
                    try:
                        model_info['service'].cleanup()
                        model_info['initialized'] = False
                        print(f"🧹 Cleaned up {model_info['name']}")
                    except Exception as e:
                        print(f"⚠️ Warning: Cleanup failed for {name}: {e}")
            
            print("🧹 All models cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")
    
    async def health_check(self) -> Dict:
        """Perform health check on all models"""
        health_status = {}
        
        for name, model_info in self.available_models.items():
            try:
                if model_info['initialized']:
                    status = model_info['service'].get_status()
                    health_status[name] = {
                        'status': 'healthy',
                        'details': status
                    }
                else:
                    health_status[name] = {
                        'status': 'not_initialized',
                        'details': {}
                    }
            except Exception as e:
                health_status[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_status

# Global model manager instance
model_manager = ModelManager() 