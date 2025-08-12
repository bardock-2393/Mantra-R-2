"""
Model Manager Service - Optimized for speed
Only essential functions kept to reduce overhead
"""

import os
import time
from typing import Dict
from config import Config

class ModelManager:
    """Simplified model manager - only essential functions kept for speed"""
    
    def __init__(self):
        self.current_model = 'qwen25vl_32b'  # Default model
        self.models = {
            'qwen25vl_32b': {
                'name': 'Qwen2.5-VL-32B',
                'description': 'Large vision-language model for comprehensive analysis',
                'status': 'available'
            }
        }
        self.last_switch = time.time()
    
    async def initialize_model(self, model_name: str = None) -> bool:
        """Initialize model - simplified placeholder"""
        try:
            if model_name:
                self.current_model = model_name
            print(f"✅ Model initialized: {self.current_model}")
            return True
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch between available AI models"""
        try:
            if model_name not in self.models:
                print(f"❌ Model {model_name} not available")
                return False
            
            # Simple model switching - no actual model loading to reduce overhead
            self.current_model = model_name
            self.last_switch = time.time()
            print(f"✅ Switched to model: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Model switching failed: {e}")
            return False

# Create global instance
model_manager = ModelManager() 