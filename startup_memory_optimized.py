#!/usr/bin/env python3
"""
Memory-Optimized Startup Script for AI Video Detective
This script optimizes GPU memory settings before starting the application
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Set up environment variables for optimal memory management"""
    print("🔧 Setting up environment variables...")
    
    # CUDA memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB cache
    
    # PyTorch memory optimization
    os.environ['PYTORCH_CUDA_MEMORY_FRACTION'] = '0.8'
    
    print("✅ Environment variables set")

def check_gpu_status():
    """Check GPU status and memory"""
    try:
        print("🔍 Checking GPU status...")
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        total_memory = int(parts[1].strip())
                        free_memory = int(parts[2].strip())
                        print(f"🚀 GPU: {name}")
                        print(f"   Total Memory: {total_memory} MB")
                        print(f"   Free Memory: {free_memory} MB")
                        print(f"   Usage: {((total_memory - free_memory) / total_memory * 100):.1f}%")
        else:
            print("⚠️ Could not get GPU status")
            
    except Exception as e:
        print(f"⚠️ GPU status check failed: {e}")

def cleanup_gpu_memory():
    """Clean up GPU memory before starting"""
    try:
        print("🧹 Cleaning up GPU memory...")
        
        # Kill any existing GPU processes (except current)
        current_pid = os.getpid()
        
        # Get GPU processes
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = []
            for line in result.stdout.strip().split('\n'):
                if line.strip() and line.strip().isdigit():
                    pid = int(line.strip())
                    if pid != current_pid:
                        pids.append(pid)
            
            if pids:
                print(f"🔄 Found {len(pids)} GPU processes, killing them...")
                for pid in pids:
                    try:
                        if os.name == 'nt':  # Windows
                            subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)
                        else:  # Linux/Mac
                            subprocess.run(f'kill -9 {pid}', shell=True, capture_output=True)
                        print(f"✅ Killed process {pid}")
                    except Exception as e:
                        print(f"⚠️ Could not kill process {pid}: {e}")
                
                # Wait for processes to be killed
                time.sleep(3)
            else:
                print("ℹ️ No GPU processes to kill")
        
        print("✅ GPU memory cleanup completed")
        
    except Exception as e:
        print(f"⚠️ GPU memory cleanup failed: {e}")

def install_dependencies():
    """Install required dependencies"""
    try:
        print("📦 Installing dependencies...")
        
        # Check if requirements.txt exists
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        # Install dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        print("✅ Dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Dependency installation failed: {e}")
        return False

def start_application():
    """Start the main application"""
    try:
        print("🚀 Starting AI Video Detective...")
        
        # Check if main.py exists
        main_file = Path("main.py")
        if not main_file.exists():
            print("❌ main.py not found")
            return False
        
        # Start the application
        subprocess.run([sys.executable, 'main.py'], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Application failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Application error: {e}")
        return False

def main():
    """Main startup function"""
    print("="*60)
    print("🚀 AI Video Detective - Memory Optimized Startup")
    print("="*60)
    
    try:
        # Setup environment
        setup_environment()
        
        # Check GPU status
        check_gpu_status()
        
        # Cleanup GPU memory
        cleanup_gpu_memory()
        
        # Install dependencies if needed
        if not install_dependencies():
            print("⚠️ Continuing with existing dependencies...")
        
        # Start application
        start_application()
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 