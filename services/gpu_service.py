"""
GPU Service Module for Round 2
Handles GPU resource management, memory optimization, and CUDA operations
"""

import os
import time
import asyncio
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
            await self._update_gpu_info()
            
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
    
    async def _update_gpu_info(self):
        """Update GPU information and memory status"""
        try:
            if self.is_initialized:
                # Get GPU handle
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get GPU info
                name_raw = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name_raw, bytes):
                    name = name_raw.decode('utf-8')
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
    
    async def get_memory_status(self) -> Dict:
        """Get current GPU memory status"""
        await self._update_gpu_info()
        
        return {
            'gpu_info': self.gpu_info,
            'pytorch_memory': {
                'allocated_gb': self.memory_allocated / (1024**3),
                'reserved_gb': self.memory_reserved / (1024**3),
                'allocated_mb': self.memory_allocated / (1024**2),
                'reserved_mb': self.memory_reserved / (1024**2)
            },
            'memory_usage_percent': (self.gpu_info.get('used_memory_bytes', 0) / 
                                    self.gpu_info.get('total_memory_bytes', 1)) * 100
        }
    
    async def optimize_memory(self):
        """Optimize GPU memory usage"""
        try:
            print("🔧 Optimizing GPU memory...")
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            import gc
            gc.collect()
            
            # Update memory status
            await self._update_gpu_info()
            
            print(f"✅ Memory optimization completed")
            print(f"📊 Memory after optimization: {self.memory_allocated / (1024**3):.2f}GB allocated")
            
        except Exception as e:
            print(f"⚠️ Warning: Memory optimization failed: {e}")
    
    async def check_memory_availability(self, required_memory_gb: float) -> bool:
        """Check if required memory is available"""
        await self._update_gpu_info()
        
        available_memory_gb = self.gpu_info.get('free_memory_gb', 0)
        return available_memory_gb >= required_memory_gb
    
    async def allocate_memory(self, size_gb: float) -> bool:
        """Allocate GPU memory for processing"""
        try:
            if not await self.check_memory_availability(size_gb):
                print(f"⚠️ Warning: Insufficient GPU memory. Required: {size_gb:.2f}GB, Available: {self.gpu_info.get('free_memory_gb', 0):.2f}GB")
                return False
            
            # Create a dummy tensor to reserve memory
            size_bytes = int(size_gb * 1024**3)
            dummy_tensor = torch.zeros(size_bytes // 4, dtype=torch.float32, device=self.device)
            
            print(f"✅ Allocated {size_gb:.2f}GB GPU memory")
            return True
            
        except Exception as e:
            print(f"❌ Memory allocation failed: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict:
        """Get GPU performance metrics"""
        try:
            if not self.is_initialized:
                return {}
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            except:
                power = 0
            
            return {
                'gpu_utilization_percent': utilization.gpu,
                'memory_utilization_percent': utilization.memory,
                'temperature_celsius': temperature,
                'power_usage_watts': power,
                'memory_status': await self.get_memory_status()
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to get performance metrics: {e}")
            return {}
    
    async def set_optimization_mode(self, mode: str):
        """Set GPU optimization mode"""
        try:
            if mode == 'speed':
                # Optimize for speed
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✅ GPU optimization mode set to SPEED")
                
            elif mode == 'memory':
                # Optimize for memory
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                print("✅ GPU optimization mode set to MEMORY")
                
            elif mode == 'balanced':
                # Balanced optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print("✅ GPU optimization mode set to BALANCED")
                
        except Exception as e:
            print(f"⚠️ Warning: Failed to set optimization mode: {e}")
    
    async def cleanup(self):
        """Clean up GPU resources"""
        try:
            print("🧹 Cleaning up GPU service...")
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Shutdown NVML
            if self.is_initialized:
                pynvml.nvmlShutdown()
            
            print("✅ GPU service cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning: GPU service cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.is_initialized:
                pynvml.nvmlShutdown()
        except:
            pass 