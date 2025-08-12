"""
Simple Model Manager for 32B Service
Handles the Qwen2.5-VL-32B model integration
"""

import asyncio
from typing import Dict, Optional, List
from config import Config
from services.qwen25vl_32b_service import qwen25vl_32b_service

class ModelManager:
    """Simple model manager for 32B service"""
    
    def __init__(self):
        self.current_model = 'qwen25vl_32b'
        self.available_models = {
            'qwen25vl_32b': {
                'name': 'Qwen2.5-VL-32B-Instruct',
                'description': 'High-performance 32B parameter model with superior video analysis capabilities',
                'service': qwen25vl_32b_service,
                'initialized': False
            }
        }
    
    async def initialize_model(self, model_name: str = None) -> bool:
        """Initialize the 32B model"""
        try:
            if model_name:
                self.current_model = model_name
            
            if self.current_model not in self.available_models:
                raise ValueError(f"Unknown model: {self.current_model}")
            
            model_info = self.available_models[self.current_model]
            
            # Check if already initialized
            if model_info['service'].is_initialized:
                print(f"ℹ️ {model_info['name']} is already initialized")
                model_info['initialized'] = True
                return True
            
            print(f"🚀 Initializing {model_info['name']}...")
            
            # Initialize the service
            await model_info['service'].initialize()
            model_info['initialized'] = True
            
            print(f"✅ {model_info['name']} initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.current_model}: {e}")
            if self.current_model in self.available_models:
                self.available_models[self.current_model]['initialized'] = False
            return False
    
    async def generate_chat_response(self, analysis_result: str, analysis_type: str, user_focus: str, message: str, chat_history: List[Dict]) -> str:
        """Generate chat response using the 32B model"""
        try:
            model_info = self.available_models[self.current_model]
            if not model_info['initialized']:
                await self.initialize_model()
            
            return await model_info['service'].generate_chat_response(
                analysis_result, analysis_type, user_focus, message, chat_history
            )
            
        except Exception as e:
            print(f"❌ Chat response generation failed: {e}")
            # Fallback to simple text generation
            try:
                return model_info['service']._generate_text_sync(
                    f"Based on this video analysis: {analysis_result}\n\nUser question: {message}\n\nPlease provide a detailed, helpful response.",
                    max_new_tokens=1024
                )
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def get_current_model(self):
        """Get current model info"""
        return self.available_models[self.current_model]
    
    async def health_check(self):
        """Check model health"""
        try:
            model_info = self.available_models[self.current_model]
            return {
                'current_model': self.current_model,
                'name': model_info['name'],
                'initialized': model_info['initialized'],
                'ready': model_info['service'].is_ready() if hasattr(model_info['service'], 'is_ready') else False
            }
        except Exception as e:
            return {
                'current_model': self.current_model,
                'error': str(e)
            }

# Global model manager instance
model_manager = ModelManager()
