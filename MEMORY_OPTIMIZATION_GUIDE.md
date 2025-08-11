# Memory Optimization Guide for AI Video Detective

## Overview

This guide explains the memory optimization features implemented to resolve CUDA out of memory errors and improve GPU memory management.

## Problem Description

The application was experiencing CUDA out of memory errors when trying to initialize the MiniCPM-V-2_6 model, even with an 80GB GPU. This was caused by:

1. **Memory fragmentation** - GPU memory was fragmented due to other processes
2. **Insufficient contiguous memory** - Large models need contiguous memory blocks
3. **No fallback mechanisms** - Application failed completely when GPU memory was insufficient

## Solutions Implemented

### 1. Environment Variable Optimization

The following environment variables are automatically set for optimal memory management:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
CUDA_LAUNCH_BLOCKING=1
CUDA_CACHE_DISABLE=0
CUDA_CACHE_MAXSIZE=2147483648
PYTORCH_CUDA_MEMORY_FRACTION=0.8
```

**Benefits:**
- `expandable_segments:True` - Reduces memory fragmentation
- `max_split_size_mb:128` - Limits memory block sizes
- `memory_fraction=0.8` - Uses 80% of GPU memory instead of 90%

### 2. Progressive Model Loading Strategy

The application now tries multiple loading strategies in order of preference:

1. **Full Precision** - Highest quality, highest memory usage
2. **8-bit Quantization** - Reduced memory usage with minimal quality loss
3. **4-bit Quantization** - Lowest memory usage with some quality loss
4. **CPU Fallback** - Runs on CPU if GPU memory is insufficient

### 3. Memory Management and Cleanup

#### Automatic Cleanup
- GPU memory is automatically cleaned up before model initialization
- PyTorch cache is cleared and garbage collection is forced
- Memory fragmentation is detected and addressed

#### Retry Logic
- Up to 3 retry attempts with memory cleanup between attempts
- Automatic fallback to CPU if all GPU attempts fail

### 4. Memory Monitoring

Real-time memory monitoring provides:
- GPU memory usage (allocated, reserved, cached)
- Memory fragmentation detection
- Process memory usage
- System memory status

## Usage Instructions

### Option 1: Use the Memory-Optimized Startup Script (Recommended)

```bash
# Navigate to the project directory
cd ai_video_detective\ copy

# Run the memory-optimized startup script
python startup_memory_optimized.py
```

This script will:
1. Set optimal environment variables
2. Check GPU status
3. Clean up existing GPU processes
4. Install dependencies if needed
5. Start the application with memory optimization

### Option 2: Manual Memory Optimization

If you prefer to start manually, ensure these environment variables are set:

```bash
# Windows (PowerShell)
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
$env:CUDA_LAUNCH_BLOCKING="1"
$env:PYTORCH_CUDA_MEMORY_FRACTION="0.8"

# Linux/Mac
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING="1"
export PYTORCH_CUDA_MEMORY_FRACTION="0.8"

# Then start the application
python main.py
```

### Option 3: Use Memory Utilities

The `utils/memory_utils.py` module provides utility functions:

```python
from utils.memory_utils import *

# Print memory summary
print_memory_summary()

# Force GPU cleanup
force_gpu_cleanup()

# Check memory fragmentation
frag_info = check_memory_fragmentation()
print(f"Memory fragmented: {frag_info['fragmented']}")

# Optimize memory settings
optimize_memory_settings()
```

## Configuration Options

### GPU Configuration (config.py)

```python
GPU_CONFIG = {
    'enabled': True,
    'device': 'cuda:0',
    'memory_limit': 80 * 1024 * 1024 * 1024,  # 80GB
    'memory_fraction': 0.8,  # Use 80% of GPU memory
    'fallback_to_cpu': True,  # Allow CPU fallback
    'memory_cleanup_threshold': 0.1,  # Cleanup when <10% free
    'max_retry_attempts': 3
}
```

### Model Configuration

```python
MINICPM_CONFIG = {
    'use_8bit': True,  # Enable 8-bit quantization
    'use_4bit': False,  # Disable 4-bit (more aggressive)
    'low_cpu_mem_usage': True,  # Reduce CPU memory usage
    'device_map': 'auto'  # Automatic device placement
}
```

## Troubleshooting

### Still Getting Memory Errors?

1. **Check GPU processes:**
   ```bash
   nvidia-smi
   ```

2. **Kill other GPU processes:**
   ```bash
   # Windows
   nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
   taskkill /PID <PID> /F
   
   # Linux
   nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
   kill -9 <PID>
   ```

3. **Restart the system** to clear all GPU memory

4. **Use CPU fallback** by setting `fallback_to_cpu: True` in config

### Performance Impact

- **8-bit quantization**: ~5-10% quality loss, ~50% memory reduction
- **4-bit quantization**: ~15-20% quality loss, ~75% memory reduction
- **CPU fallback**: ~80-90% slower, but guaranteed to work

## Monitoring and Debugging

### Memory Status Endpoint

The application provides memory status information:

```python
# Get GPU service status
gpu_status = gpu_service.get_status()
print(f"GPU Memory: {gpu_status['memory_info']}")

# Get model status
model_status = model_manager.get_status()
print(f"Model: {model_status['current_model']}")
```

### Log Analysis

Look for these log messages:
- `🧹 Forcing GPU memory cleanup...` - Memory cleanup in progress
- `🔄 Trying loading strategy: _load_model_8bit` - Quantization being used
- `⚠️ GPU memory still insufficient, falling back to CPU` - CPU fallback activated

## Best Practices

1. **Start fresh** - Use the startup script to clear GPU memory
2. **Monitor memory** - Check GPU memory usage before starting
3. **Close other applications** - Free up GPU memory from other processes
4. **Use quantization** - Enable 8-bit quantization for better memory efficiency
5. **Enable CPU fallback** - Ensures the application always works

## Dependencies

Additional dependencies added:
- `psutil>=5.9.0` - System memory monitoring
- `bitsandbytes>=0.41.0` - Quantization support (already in requirements)

## Support

If you continue to experience memory issues:

1. Check the logs for specific error messages
2. Verify GPU memory availability with `nvidia-smi`
3. Try the CPU fallback option
4. Consider reducing model precision or using smaller models

## Technical Details

### Memory Management Flow

1. **Initialization Check** - Verify sufficient GPU memory
2. **Memory Cleanup** - Clear PyTorch cache and force garbage collection
3. **Progressive Loading** - Try different precision levels
4. **Fallback Handling** - Switch to CPU if GPU fails
5. **Retry Logic** - Multiple attempts with cleanup between tries

### Quantization Implementation

- **8-bit**: Uses `BitsAndBytesConfig` with `load_in_8bit=True`
- **4-bit**: Uses `BitsAndBytesConfig` with `load_in_4bit=True`
- **Fallback**: Automatic detection and graceful degradation

This comprehensive memory optimization system ensures the AI Video Detective application can run reliably even under challenging GPU memory conditions. 