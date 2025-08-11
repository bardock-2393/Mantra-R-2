"""
GPU Service Module for Round 2
Handles GPU resource management, memory optimization, and CUDA operations
"""

import os
import time
import torch
import pynvml
from typing import Dict, List, Optional, Tuple
from config import Config

class GPUService:
    """GPU resource management and optimization service"""
    
    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self.memory_allocated = 0
        self.memory_reserved = 0
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize GPU service and check resources"""
        try:
            print("🚀 Initializing GPU service...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. GPU is required for Round 2.")
            
            # Initialize NVML for GPU monitoring
            try:
                pynvml.nvmlInit()
                self.is_initialized = True
                print("✅ NVML initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: NVML initialization failed: {e}")
                self.is_initialized = False
            
            # Set device
            self.device = torch.device(Config.GPU_CONFIG['device'])
            
            # Get GPU information
            self._update_gpu_info()
            
            # Set PyTorch memory management
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            print(f"✅ GPU service initialized on {self.device}")
            memory_gb = self.gpu_info.get('total_memory_gb')
            if memory_gb is not None:
                print(f"📊 GPU Memory: {memory_gb:.1f}GB")
            else:
                print("📊 GPU Memory: Unknown")
            print(f"📊 CUDA Version: {torch.version.cuda}")
            
        except Exception as e:
            print(f"❌ GPU service initialization failed: {e}")
            raise
    
    def _update_gpu_info(self):
        """Update GPU information and memory status"""
        try:
            if self.is_initialized:
                # Get GPU handle
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get GPU info
                name_raw = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name_raw, bytes):
                    try:
                        name = name_raw.decode('utf-8', errors='ignore')
                    except (UnicodeDecodeError, AttributeError):
                        name = str(name_raw)
                else:
                    name = str(name_raw)
                total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                self.gpu_info = {
                    'name': name,
                    'total_memory_bytes': total_memory.total,
                    'total_memory_gb': total_memory.total / (1024**3),
                    'free_memory_bytes': total_memory.free,
                    'free_memory_gb': total_memory.free / (1024**3),
                    'used_memory_bytes': total_memory.used,
                    'used_memory_gb': total_memory.used / (1024**3)
                }
                
                # Get PyTorch memory info
                if torch.cuda.is_available():
                    self.memory_allocated = torch.cuda.memory_allocated(self.device)
                    self.memory_reserved = torch.cuda.memory_reserved(self.device)
                    
        except Exception as e:
            print(f"⚠️ Warning: Failed to update GPU info: {e}")
    
    def get_status(self) -> Dict:
        """Get current GPU status"""
        try:
            self._update_gpu_info()
            
            status = {
                'initialized': self.is_initialized,
                'device': str(self.device) if self.device else None,
                'gpu_info': self.gpu_info,
                'memory_allocated_gb': self.memory_allocated / (1024**3),
                'memory_reserved_gb': self.memory_reserved / (1024**3)
            }
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        try:
            if not self.is_initialized:
                return False
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Update memory info
                self._update_gpu_info()
                
                print(f"🧹 GPU memory optimized. Allocated: {self.memory_allocated / (1024**3):.2f}GB")
                return True
                
        except Exception as e:
            print(f"⚠️ Warning: Memory optimization failed: {e}")
            return False
    
    def check_memory_availability(self, required_memory_gb: float) -> bool:
        """Check if required memory is available"""
        try:
            if not self.is_initialized:
                return False
            
            self._update_gpu_info()
            available_memory = self.gpu_info.get('free_memory_gb', 0)
            
            return available_memory >= required_memory_gb
            
        except Exception as e:
            print(f"⚠️ Warning: Memory availability check failed: {e}")
            return False
    
    def allocate_memory(self, size_gb: float) -> bool:
        """Allocate GPU memory (placeholder for future implementation)"""
        try:
            if not self.check_memory_availability(size_gb):
                print(f"❌ Insufficient GPU memory. Required: {size_gb}GB")
                return False
            
            print(f"✅ GPU memory allocation successful: {size_gb}GB")
            return True
            
        except Exception as e:
            print(f"⚠️ Warning: Memory allocation failed: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict:
        """Get GPU performance metrics"""
        try:
            if not self.is_initialized:
                return {'error': 'GPU service not initialized'}
            
            self._update_gpu_info()
            
            metrics = {
                'gpu_name': self.gpu_info.get('name', 'Unknown'),
                'total_memory_gb': self.gpu_info.get('total_memory_gb', 0),
                'free_memory_gb': self.gpu_info.get('free_memory_gb', 0),
                'used_memory_gb': self.gpu_info.get('used_memory_gb', 0),
                'memory_utilization_percent': 0,
                'pytorch_allocated_gb': self.memory_allocated / (1024**3),
                'pytorch_reserved_gb': self.memory_reserved / (1024**3)
            }
            
            # Calculate memory utilization
            total = self.gpu_info.get('total_memory_gb', 0)
            used = self.gpu_info.get('used_memory_gb', 0)
            if total > 0:
                metrics['memory_utilization_percent'] = (used / total) * 100
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def set_optimization_mode(self, mode: str):
        """Set GPU optimization mode"""
        try:
            if mode == 'speed':
                # Optimize for speed
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("🚀 GPU optimized for speed")
                
            elif mode == 'memory':
                # Optimize for memory
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                print("💾 GPU optimized for memory")
                
            elif mode == 'balanced':
                # Balanced optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("⚖️ GPU optimized for balanced performance")
                
            else:
                print(f"⚠️ Unknown optimization mode: {mode}")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to set optimization mode: {e}")
    
    def get_gpu_status_message(self) -> str:
        """Get human-readable GPU status message"""
        try:
            if not self.is_initialized:
                return "GPU service not initialized"
            
            self._update_gpu_info()
            
            gpu_name = self.gpu_info.get('name', 'Unknown GPU')
            total_memory = self.gpu_info.get('total_memory_gb', 0)
            free_memory = self.gpu_info.get('free_memory_gb', 0)
            used_memory = self.gpu_info.get('used_memory_gb', 0)
            
            status_msg = f"GPU: {gpu_name}\n"
            status_msg += f"Memory: {used_memory:.1f}GB / {total_memory:.1f}GB (Free: {free_memory:.1f}GB)\n"
            status_msg += f"Device: {self.device}"
            
            return status_msg
            
        except Exception as e:
            return f"GPU status error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU service resources"""
        try:
            if self.is_initialized:
                # Clear PyTorch cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Shutdown NVML
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                
                self.is_initialized = False
                print("🧹 GPU service cleaned up")
                
        except Exception as e:
            print(f"⚠️ Warning: GPU service cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass 