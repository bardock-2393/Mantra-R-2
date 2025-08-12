#!/usr/bin/env python3
"""
Dependency Installation Script for AI Video Detective
Installs all required packages for Qwen2.5-VL-32B model
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing core dependencies...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install PyTorch with CUDA support
    if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "Installing PyTorch with CUDA"):
        return False
    
    # Install transformers and related packages
    if not run_command("pip install transformers>=4.40.0 accelerate>=0.20.0", "Installing transformers"):
        return False
    
    # Install other required packages
    packages = [
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "sentence-transformers",
        "qwen-vl-utils[decord]",
        "bitsandbytes",
        "sentencepiece",
        "protobuf"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Warning: Failed to install {package}, continuing...")
    
    return True

def install_optional_dependencies():
    """Install optional performance packages"""
    print("\n🚀 Installing optional performance packages...")
    
    optional_packages = [
        "flash-attn",
        "xformers",
        "av",
        "decord",
        "ffmpeg-python"
    ]
    
    for package in optional_packages:
        try:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️ Warning: {package} installation failed, continuing...")
        except Exception as e:
            print(f"⚠️ Warning: Could not install {package}: {e}")
    
    return True

def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        'torch',
        'transformers',
        'PIL',
        'numpy',
        'cv2'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} (PIL) available")
            elif package == 'cv2':
                import cv2
                print(f"✅ {package} (opencv) available")
            else:
                __import__(package)
                print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            all_good = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("❌ CUDA not available")
            all_good = False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        all_good = False
    
    return all_good

def main():
    """Main installation function"""
    print("🎯 AI Video Detective - Dependency Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("❌ Python version check failed")
        return False
    
    # Install core dependencies
    if not install_core_dependencies():
        print("❌ Core dependency installation failed")
        return False
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return False
    
    print("\n🎉 Dependency installation completed successfully!")
    print("\nNext steps:")
    print("1. Run: python load_model.py")
    print("2. Or run: python start_with_model.py")
    print("3. The model will be loaded and the application will start")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Installation completed successfully!")
        else:
            print("\n❌ Installation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Installation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

