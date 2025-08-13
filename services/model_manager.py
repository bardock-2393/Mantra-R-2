"""
Enhanced Model Manager for AI Video Detective
Handles both Qwen2.5-VL-7B and Qwen2.5-VL-32B model integration
"""

import asyncio
from typing import Dict, Optional, List
from config import Config

# Import both services
try:
    from services.qwen25vl_7b_service import Qwen25VL7BService
    QWEN_7B_AVAILABLE = True
except ImportError:
    QWEN_7B_AVAILABLE = False
    print("⚠️ Qwen2.5-VL-7B service not available")

try:
    from services.qwen25vl_32b_service import Qwen25VL32BService
    QWEN_32B_AVAILABLE = True
except ImportError:
    QWEN_32B_AVAILABLE = False
    print("⚠️ Qwen2.5-VL-32B service not available")

class ModelManager:
    """Enhanced model manager for both 7B and 32B services"""
    
    def __init__(self):
        # Default to 7B model for better performance/memory balance
        self.current_model = 'qwen25vl_7b'
        self.available_models = {}
        
        # Initialize available models
        if QWEN_7B_AVAILABLE:
            self.available_models['qwen25vl_7b'] = {
                'name': 'Qwen2.5-VL-7B-Instruct',
                'description': 'Balanced 7B parameter model with excellent video analysis capabilities',
                'service_class': Qwen25VL7BService,
                'service_instance': None,
                'initialized': False,
                'memory_requirement': '8GB+',
                'performance': 'Balanced',
                'color': '#3b82f6'  # Blue
            }
        
        if QWEN_32B_AVAILABLE:
            self.available_models['qwen25vl_32b'] = {
                'name': 'Qwen2.5-VL-32B-Instruct',
                'description': 'High-performance 32B parameter model with superior video analysis capabilities',
                'service_class': Qwen25VL32BService,
                'service_instance': None,
                'initialized': False,
                'memory_requirement': '40GB+',
                'performance': 'High',
                'color': '#8b5cf6'  # Purple
            }
        
        # If no models available, raise error
        if not self.available_models:
            raise RuntimeError("No Qwen2.5-VL models available. Please check your installation.")
        
        # If 7B not available, fallback to 32B
        if self.current_model not in self.available_models:
            self.current_model = list(self.available_models.keys())[0]
            print(f"⚠️ 7B model not available, falling back to {self.current_model}")
    
    async def initialize_model(self, model_name: str = None) -> bool:
        """Initialize the specified model"""
        try:
            if model_name:
                if model_name not in self.available_models:
                    raise ValueError(f"Unknown model: {model_name}. Available: {list(self.available_models.keys())}")
                self.current_model = model_name
            
            model_info = self.available_models[self.current_model]
            
            # Check if already initialized
            if model_info['initialized'] and model_info['service_instance']:
                print(f"ℹ️ {model_info['name']} is already initialized")
                return True
            
            print(f"🚀 Initializing {model_info['name']}...")
            
            # Create service instance if not exists
            if not model_info['service_instance']:
                model_info['service_instance'] = model_info['service_class']()
            
            # Initialize the service
            success = await model_info['service_instance'].initialize()
            if success:
                model_info['initialized'] = True
                print(f"✅ {model_info['name']} initialized successfully")
                return True
            else:
                print(f"❌ Failed to initialize {model_info['name']}")
                model_info['initialized'] = False
                return False
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.current_model}: {e}")
            if self.current_model in self.available_models:
                self.available_models[self.current_model]['initialized'] = False
            return False
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(self.available_models.keys())}")
            
            # Cleanup current model if initialized
            if self.available_models[self.current_model]['initialized']:
                await self.cleanup_current_model()
            
            # Switch to new model
            self.current_model = model_name
            print(f"🔄 Switched to {model_info['name']}")
            
            # Initialize new model
            return await self.initialize_model()
            
        except Exception as e:
            print(f"❌ Failed to switch to {model_name}: {e}")
            return False
    
    async def cleanup_current_model(self):
        """Cleanup the currently loaded model"""
        try:
            model_info = self.available_models[self.current_model]
            if model_info['service_instance'] and model_info['initialized']:
                model_info['service_instance'].cleanup()
                model_info['initialized'] = False
                print(f"🧹 Cleaned up {model_info['name']}")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate chat response using the current model"""
        try:
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                await self.initialize_model()
            
            if not model_info['service_instance']:
                raise RuntimeError("Service instance not available")
            
            return await model_info['service_instance'].chat(message, chat_history)
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def get_current_model(self):
        """Get current model info"""
        return self.available_models[self.current_model]
    
    def get_available_models(self):
        """Get list of all available models"""
        return {
            name: {
                'name': info['name'],
                'description': info['description'],
                'memory_requirement': info['memory_requirement'],
                'performance': info['performance'],
                'color': info['color'],
                'available': True
            }
            for name, info in self.available_models.items()
        }
    
    async def health_check(self):
        """Check model health"""
        try:
            model_info = self.available_models[self.current_model]
            service = model_info['service_instance']
            
            health_status = {
                'current_model': self.current_model,
                'name': model_info['name'],
                'initialized': model_info['initialized'],
                'ready': False
            }
            
            if service and hasattr(service, 'is_initialized'):
                health_status['ready'] = service.is_initialized
                
                # Get additional model info if available
                if hasattr(service, 'get_model_info'):
                    try:
                        model_details = service.get_model_info()
                        health_status['model_details'] = model_details
                    except Exception as e:
                        health_status['model_details_error'] = str(e)
            
            return health_status
            
        except Exception as e:
            return {
                'current_model': self.current_model,
                'error': str(e)
            }
    
    async def analyze_video(self, video_path: str, analysis_type: str = "comprehensive") -> Dict:
        """Analyze video using the current model"""
        try:
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                await self.initialize_model()
            
            if not model_info['service_instance']:
                raise RuntimeError("Service instance not available")
            
            return await model_info['service_instance'].analyze_video(video_path, analysis_type)
            
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return {
                'error': str(e),
                'model_used': model_info['name'] if 'model_info' in locals() else 'Unknown',
                'status': 'failed'
            }

# Global model manager instance
model_manager = ModelManager()
