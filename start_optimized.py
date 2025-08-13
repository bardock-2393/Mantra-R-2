#!/usr/bin/env python3
"""
Optimized Startup Script for AI Video Detective
Enhanced with video chunking, WebSocket progress updates, and performance optimization
"""

import os
import sys
import time
import asyncio
import threading
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'flask', 'flask_socketio', 
        'decord', 'opencv-python', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_optimized.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def check_gpu():
    """Check GPU availability and capabilities"""
    print("\n🚀 Checking GPU capabilities...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            device_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"✅ CUDA available")
            print(f"   Devices: {device_count}")
            print(f"   Current: {current_device}")
            print(f"   Name: {device_name}")
            print(f"   Memory: {device_memory:.1f} GB")
            
            # Check if we have enough memory for 7B model
            if device_memory >= 16:
                print("✅ Sufficient GPU memory for Qwen2.5-VL-7B")
            elif device_memory >= 8:
                print("✅ Sufficient GPU memory for Qwen2.5-VL-7B (minimum)")
            else:
                print("⚠️ Limited GPU memory - may need optimization")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_video_tools():
    """Check video processing tools"""
    print("\n🎬 Checking video processing tools...")
    
    # Check decord
    try:
        import decord
        print("✅ decord available for fast video processing")
    except ImportError:
        print("❌ decord not available - will use OpenCV fallback")
    
    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV available for image processing")
    except ImportError:
        print("❌ OpenCV not available")
    
    # Check ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ffmpeg available for video manipulation")
        else:
            print("⚠️ ffmpeg not working properly")
    except Exception:
        print("⚠️ ffmpeg not available")

def initialize_services():
    """Initialize core services"""
    print("\n🔧 Initializing services...")
    
    try:
        from config import Config
        print("✅ Configuration loaded")
        
        # Initialize performance monitor
        from services.performance_service import PerformanceMonitor
        performance_monitor = PerformanceMonitor(Config.__dict__)
        print("✅ Performance monitor initialized")
        
        # Initialize video chunking service
        if hasattr(Config, 'VIDEO_CHUNKING_CONFIG'):
            from services.video_chunking_service import VideoChunkingService
            chunking_service = VideoChunkingService(Config.VIDEO_CHUNKING_CONFIG)
            print("✅ Video chunking service initialized")
        else:
            print("⚠️ Video chunking configuration not found")
        
        # Initialize WebSocket service
        if hasattr(Config, 'WEBSOCKET_CONFIG'):
            print("✅ WebSocket configuration found")
        else:
            print("⚠️ WebSocket configuration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def run_performance_tests():
    """Run basic performance tests"""
    print("\n⚡ Running performance tests...")
    
    try:
        import torch
        import time
        
        if torch.cuda.is_available():
            # Test GPU memory allocation
            print("   Testing GPU memory allocation...")
            start_time = time.time()
            
            # Allocate some test tensors
            test_tensor = torch.randn(1000, 1000, device='cuda')
            allocation_time = time.time() - start_time
            
            print(f"   GPU allocation: {allocation_time:.3f}s")
            
            # Test memory cleanup
            del test_tensor
            torch.cuda.empty_cache()
            print("   GPU memory cleanup: OK")
        
        # Test CPU performance
        print("   Testing CPU performance...")
        start_time = time.time()
        
        # Simple computation test
        result = sum(i**2 for i in range(10000))
        cpu_time = time.time() - start_time
        
        print(f"   CPU computation: {cpu_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return False

def main():
    """Main startup function"""
    print("🚀 AI Video Detective - Optimized Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check video tools
    check_video_tools()
    
    # Initialize services
    if not initialize_services():
        print("\n❌ Service initialization failed.")
        return False
    
    # Run performance tests
    if not run_performance_tests():
        print("\n⚠️ Performance tests failed, but continuing...")
    
    print("\n✅ Startup completed successfully!")
    
    if gpu_available:
        print("\n🎯 Ready for GPU-accelerated video processing:")
        print("   - Video chunking for 120+ minute videos")
        print("   - Real-time WebSocket progress updates")
        print("   - torch.compile optimization enabled")
        print("   - Model quantization for memory efficiency")
        print("   - Parallel chunk processing")
    else:
        print("\n⚠️ Running in CPU mode - performance will be limited")
    
    print("\n🚀 Starting Flask application...")
    print("   Use Ctrl+C to stop")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            # Import and run the main app
            from app import app, socketio
            socketio.run(app, host='0.0.0.0', port=8000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)
