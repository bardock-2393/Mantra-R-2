# AI Video Detective - Optimization Guide

## Overview
This guide documents the comprehensive optimizations implemented to handle 120+ minute videos efficiently with GPU acceleration, real-time progress tracking, and improved performance.

## 🚀 Key Optimizations Implemented

### 1. Timeout Issues Fixed
- **Cloudflare Timeout**: Increased from 5 minutes to 60 minutes
- **Flask Timeouts**: Extended session lifetime and request timeouts
- **Processing Timeouts**: Vision processing: 30min, Generation: 30min, Total: 60min

### 2. Video Chunking System
- **Chunk Duration**: 5-10 minutes per chunk (configurable)
- **Parallel Processing**: Process chunks in parallel on GPU
- **Memory Management**: Clean GPU memory between chunks
- **Result Merging**: Automatically merge results from all chunks

### 3. GPU Memory Management
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Automatic Cleanup**: Trigger cleanup at 80% memory threshold
- **Chunk Isolation**: Process chunks independently to prevent memory leaks
- **Memory Optimization**: Use FP16 precision and model quantization

### 4. Frame Processing Speed
- **Reduced Frames**: From 8 to 4-6 frames per chunk
- **Decord Integration**: Fast video decoding (replaces OpenCV for video)
- **Resolution Scaling**: Default 720p for speed vs. quality balance
- **Parallel Extraction**: Extract frames from chunks in parallel

### 5. AI Model Optimization
- **torch.compile**: Enabled 'max-autotune' mode for maximum speed
- **Model Quantization**: INT8/FP16 quantization for memory efficiency
- **Batch Optimization**: Single batch processing for memory efficiency
- **Precision Control**: BFloat16 for optimal 32B model performance

### 6. Progress Bar System
- **WebSocket Integration**: Real-time progress updates
- **Chunk Progress**: Individual chunk progress (Chunk 1/10, 2/10, etc.)
- **Time Estimates**: Per-chunk and overall time remaining
- **Status Updates**: Real-time processing status and messages

### 7. Async Processing
- **Background Processing**: Non-blocking video analysis
- **Queue System**: Handle multiple videos simultaneously
- **Progress Tracking**: Real-time updates without blocking UI
- **Error Handling**: Graceful failure handling per chunk

### 8. Video Preprocessing
- **Compression**: Optimize video format before chunking
- **Resolution Scaling**: 720p/480p for processing speed
- **Format Optimization**: Support for long video formats
- **Memory Efficiency**: Reduced frame count for large videos

## 📁 New Files Created

### Core Services
- `services/video_chunking_service.py` - Video splitting and parallel processing
- `services/websocket_service.py` - Real-time progress updates
- `services/performance_service.py` - Enhanced GPU monitoring and optimization

### Configuration
- `requirements_optimized.txt` - New dependencies for optimization
- `start_optimized.py` - Enhanced startup script with dependency checks
- `OPTIMIZATION_GUIDE.md` - This comprehensive guide

## 🔧 Configuration Updates

### Video Chunking Configuration
```python
VIDEO_CHUNKING_CONFIG = {
    'enabled': True,
    'chunk_duration': 300,  # 5 minutes per chunk
    'max_workers': 2,  # Parallel processing workers
    'frame_rate': 1,  # 1 fps for analysis
    'resolution': (720, 480),  # 720p resolution
    'use_decord': True,  # Use decord instead of OpenCV
    'memory_cleanup': True,  # Clean GPU memory between chunks
}
```

### WebSocket Configuration
```python
WEBSOCKET_CONFIG = {
    'enabled': True,
    'ping_timeout': 60,
    'ping_interval': 25,
    'cors_allowed_origins': '*',
    'async_mode': 'threading'
}
```

### Performance Optimization
```python
PERFORMANCE_OPTIMIZATION = {
    'torch_compile_mode': 'max-autotune',
    'quantization_enabled': True,
    'memory_cleanup_threshold': 0.8,
    'parallel_chunk_processing': True,
}
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements_optimized.txt
```

### 2. Run Optimized Startup
```bash
python start_optimized.py
```

### 3. Monitor Progress
- Real-time WebSocket updates
- GPU memory monitoring
- Performance metrics
- Chunk processing status

## 📊 Performance Improvements

### Before Optimization
- **Max Video Length**: ~30 minutes
- **Frame Count**: 8 frames per video
- **Processing Time**: 5-10 minutes
- **Memory Usage**: High, potential crashes
- **Progress Tracking**: None

### After Optimization
- **Max Video Length**: 120+ minutes
- **Frame Count**: 4-6 frames per chunk
- **Processing Time**: 20-40 minutes (scalable)
- **Memory Usage**: Controlled, stable
- **Progress Tracking**: Real-time WebSocket updates

## 🎯 Use Cases

### Long Video Analysis
- **120+ minute videos**: Split into 5-10 minute chunks
- **Parallel processing**: Multiple chunks simultaneously
- **Memory efficiency**: Controlled GPU memory usage
- **Progress tracking**: Real-time updates for long processing

### Batch Processing
- **Multiple videos**: Queue system for multiple uploads
- **Resource management**: Efficient GPU utilization
- **Error isolation**: Failed chunks don't affect others
- **Scalability**: Add more workers as needed

### Real-time Monitoring
- **WebSocket updates**: Live progress without page refresh
- **Performance metrics**: GPU memory and utilization
- **Status tracking**: Processing state and time estimates
- **Error reporting**: Immediate feedback on issues

## 🔍 Monitoring and Debugging

### GPU Memory Monitoring
```python
from services.performance_service import PerformanceMonitor
monitor = PerformanceMonitor(config)
summary = monitor.get_performance_summary()
recommendations = monitor.get_recommendations()
```

### WebSocket Status
```python
from services.websocket_service import WebSocketService
status = websocket_service.get_session_status(session_id)
active_count = websocket_service.get_active_sessions_count()
```

### Video Chunking Status
```python
from services.video_chunking_service import VideoChunkingService
chunking = VideoChunkingService(config)
memory_usage = chunking.get_memory_usage()
```

## ⚠️ Important Notes

### Memory Requirements
- **Minimum GPU**: 40GB+ for 32B model
- **Recommended**: 80GB+ for optimal performance
- **Memory Cleanup**: Automatic at 80% threshold
- **Chunk Size**: Adjust based on available memory

### Performance Tuning
- **Chunk Duration**: Balance between memory and speed
- **Frame Count**: Reduce for memory, increase for quality
- **Resolution**: Lower resolution = faster processing
- **Workers**: More workers = more memory usage

### Error Handling
- **Chunk Failures**: Individual chunks can fail without affecting others
- **Memory Issues**: Automatic cleanup and retry mechanisms
- **Timeout Handling**: Extended timeouts for long videos
- **Fallback Processing**: OpenCV fallback if decord fails

## 🚀 Future Enhancements

### Planned Features
- **Dynamic Chunking**: Adaptive chunk size based on content
- **Advanced Quantization**: INT4 and mixed precision support
- **Distributed Processing**: Multi-GPU chunk distribution
- **Cloud Integration**: Hybrid local/cloud processing

### Performance Targets
- **Target Processing**: 120min video in <30 minutes
- **Memory Efficiency**: <70% GPU memory usage
- **Scalability**: Support for 4K+ resolution videos
- **Real-time**: <1 second latency for progress updates

## 📞 Support and Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce chunk size or frame count
2. **Timeout Issues**: Check Cloudflare and Flask settings
3. **WebSocket Failures**: Verify CORS and network configuration
4. **Performance Issues**: Enable torch.compile and quantization

### Debug Mode
```bash
# Enable debug logging
export DEBUG=1
python start_optimized.py

# Check GPU status
nvidia-smi

# Monitor memory usage
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

---

**Note**: This optimization system is designed for production use with 120+ minute videos. Test thoroughly with your specific hardware and video formats before deployment.
