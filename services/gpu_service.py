"""
GPU Service - Optimized for speed
Only essential functions kept to reduce overhead
"""

# Make torch import optional for server deployment
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available in GPU service, using CPU fallback")

import time
from typing import Dict
from config import Config

class GPUService:
    """Simplified GPU service - only essential functions kept for speed"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device(Config.GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.gpu_info = None
        self.memory_info = None
        self.last_update = 0
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize GPU service - simplified"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device(Config.GPU_CONFIG['device'])
                self.is_initialized = True
                print(f"✅ GPU service initialized on {self.device}")
            elif TORCH_AVAILABLE:
                print("⚠️ CUDA not available, using CPU")
                self.device = torch.device('cpu')
                self.is_initialized = False
            else:
                print("⚠️ PyTorch not available, using CPU")
                self.device = 'cpu'
                self.is_initialized = False
        except Exception as e:
            print(f"❌ GPU service initialization failed: {e}")
            self.is_initialized = False
    
    def get_status(self) -> Dict:
        """Get basic GPU status - simplified for speed"""
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return {
                    'available': False,
                    'device': 'cpu',
                    'memory_total': 0,
                    'memory_used': 0,
                    'memory_free': 0
                }
            
            # Only update GPU info every 5 seconds to reduce overhead
            current_time = time.time()
            if current_time - self.last_update > 5:
                self.gpu_info = torch.cuda.get_device_properties(0)
                self.memory_info = torch.cuda.memory_stats()
                self.last_update = current_time
            
            if self.gpu_info and self.memory_info:
                memory_total = self.gpu_info.total_memory
                memory_allocated = self.memory_info.get('allocated_bytes.all.current', 0)
                memory_free = memory_total - memory_allocated
                
                return {
                    'available': True,
                    'device': str(self.device),
                    'name': self.gpu_info.name,
                    'memory_total': memory_total // (1024**3),  # GB
                    'memory_used': memory_allocated // (1024**3),  # GB
                    'memory_free': memory_free // (1024**3),  # GB
                    'compute_capability': f"{self.gpu_info.major}.{self.gpu_info.minor}"
                }
            else:
                return {
                    'available': True,
                    'device': str(self.device),
                    'memory_total': 0,
                    'memory_used': 0,
                    'memory_free': 0
                }
                
        except Exception as e:
            print(f"❌ Error getting GPU status: {e}")
            return {
                'available': False,
                'error': str(e)
            }

# Create global instance
gpu_service = GPUService() 