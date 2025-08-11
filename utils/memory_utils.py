"""
Memory Management Utilities for GPU Optimization
Handles memory cleanup, fragmentation, and optimization
"""

import os
import gc
import time
import torch
import psutil
from typing import Dict, Optional, Tuple
import subprocess
import platform

def get_system_memory_info() -> Dict:
    """Get system memory information"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    except Exception as e:
        print(f"⚠️ Warning: Could not get system memory info: {e}")
        return {}

def get_gpu_memory_info() -> Dict:
    """Get GPU memory information"""
    try:
        if not torch.cuda.is_available():
            return {}
        
        device = torch.device('cuda:0')
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        memory_cached = torch.cuda.memory_reserved(device) / (1024**3)
        
        return {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'cached_gb': memory_cached,
            'device': str(device)
        }
    except Exception as e:
        print(f"⚠️ Warning: Could not get GPU memory info: {e}")
        return {}

def force_gpu_cleanup() -> bool:
    """Force cleanup of GPU memory"""
    try:
        if not torch.cuda.is_available():
            return True
        
        print("🧹 Forcing GPU memory cleanup...")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Wait for cleanup to take effect
        time.sleep(1)
        
        print("✅ GPU memory cleanup completed")
        return True
        
    except Exception as e:
        print(f"❌ GPU memory cleanup failed: {e}")
        return False

def check_memory_fragmentation() -> Dict:
    """Check for memory fragmentation issues"""
    try:
        if not torch.cuda.is_available():
            return {'fragmented': False, 'reason': 'CUDA not available'}
        
        device = torch.device('cuda:0')
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        
        # Calculate fragmentation ratio
        if memory_reserved > 0:
            fragmentation_ratio = memory_allocated / memory_reserved
            fragmented = fragmentation_ratio < 0.7  # If less than 70% of reserved memory is allocated
        else:
            fragmented = False
            fragmentation_ratio = 0
        
        return {
            'fragmented': fragmented,
            'fragmentation_ratio': fragmentation_ratio,
            'allocated_bytes': memory_allocated,
            'reserved_bytes': memory_reserved,
            'allocated_gb': memory_allocated / (1024**3),
            'reserved_gb': memory_reserved / (1024**3)
        }
        
    except Exception as e:
        print(f"⚠️ Warning: Could not check memory fragmentation: {e}")
        return {'fragmented': False, 'reason': str(e)}

def optimize_memory_settings():
    """Set optimal memory management environment variables"""
    try:
        # Set PyTorch memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Set memory fraction if CUDA is available
        if torch.cuda.is_available():
            # Use 80% of GPU memory to leave room for other processes
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception as e:
                print(f"⚠️ Warning: Could not enable memory efficient attention: {e}")
        
        print("✅ Memory settings optimized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to optimize memory settings: {e}")
        return False

def get_process_memory_usage() -> Dict:
    """Get current process memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except Exception as e:
        print(f"⚠️ Warning: Could not get process memory usage: {e}")
        return {}

def kill_gpu_processes(except_pid: Optional[int] = None) -> bool:
    """Kill GPU processes to free memory (use with caution)"""
    try:
        if platform.system() == "Windows":
            # Windows command
            cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
        else:
            # Linux command
            cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("⚠️ Warning: Could not get GPU process list")
            return False
        
        pids = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and line.strip().isdigit():
                pid = int(line.strip())
                if except_pid is None or pid != except_pid:
                    pids.append(pid)
        
        if not pids:
            print("ℹ️ No GPU processes to kill")
            return True
        
        print(f"🔄 Found {len(pids)} GPU processes, killing them...")
        
        for pid in pids:
            try:
                if platform.system() == "Windows":
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
                else:
                    subprocess.run(f"kill -9 {pid}", shell=True)
                print(f"✅ Killed process {pid}")
            except Exception as e:
                print(f"⚠️ Warning: Could not kill process {pid}: {e}")
        
        # Wait for processes to be killed
        time.sleep(2)
        
        # Force cleanup
        force_gpu_cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to kill GPU processes: {e}")
        return False

def print_memory_summary():
    """Print a comprehensive memory summary"""
    print("\n" + "="*50)
    print("📊 MEMORY STATUS SUMMARY")
    print("="*50)
    
    # System memory
    sys_memory = get_system_memory_info()
    if sys_memory:
        print(f"💻 System Memory:")
        print(f"   Total: {sys_memory['total_gb']:.1f}GB")
        print(f"   Available: {sys_memory['available_gb']:.1f}GB")
        print(f"   Used: {sys_memory['used_gb']:.1f}GB ({sys_memory['percent_used']:.1f}%)")
    
    # GPU memory
    gpu_memory = get_gpu_memory_info()
    if gpu_memory:
        print(f"🚀 GPU Memory:")
        print(f"   Allocated: {gpu_memory['allocated_gb']:.2f}GB")
        print(f"   Reserved: {gpu_memory['reserved_gb']:.2f}GB")
        print(f"   Cached: {gpu_memory['cached_gb']:.2f}GB")
    
    # Process memory
    proc_memory = get_process_memory_usage()
    if proc_memory:
        print(f"🔧 Process Memory:")
        print(f"   RSS: {proc_memory['rss_gb']:.2f}GB")
        print(f"   VMS: {proc_memory['vms_gb']:.2f}GB")
        print(f"   Percent: {proc_memory['percent']:.1f}%")
    
    # Memory fragmentation
    fragmentation = check_memory_fragmentation()
    if fragmentation:
        print(f"🧩 Memory Fragmentation:")
        print(f"   Fragmented: {fragmentation['fragmented']}")
        if 'fragmentation_ratio' in fragmentation:
            print(f"   Ratio: {fragmentation['fragmentation_ratio']:.2f}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test the memory utilities
    print_memory_summary()
    optimize_memory_settings()
    force_gpu_cleanup()
    print_memory_summary() 