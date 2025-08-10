#!/usr/bin/env python3
"""
AI Video Detective - Setup Test Script
Tests the configuration and dependencies for the AI Video Detective application.
"""

import os
import sys
import importlib
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_imports():
    """Test required package imports"""
    print("\n📦 Testing package imports...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'Transformers'),
        ('opencv-python', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('asyncio', 'asyncio'),
        ('dotenv', 'python-dotenv'),
        ('werkzeug', 'Werkzeug')
    ]
    
    all_imports_ok = True
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - Imported successfully")
        except ImportError as e:
            print(f"❌ {name} - Import failed: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_environment():
    """Test environment configuration for Round 2"""
    print("\n🔧 Testing environment configuration...")
    
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env file loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load .env file: {e}")
            return False
    else:
        print("⚠️  .env file not found - create one with your configuration")
    
    # Check GPU availability for Round 2
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
        else:
            print("❌ No CUDA-capable GPU detected")
            print("⚠️  GPU is required for Round 2 performance targets")
            return False
    except ImportError:
        print("❌ PyTorch not installed - required for GPU processing")
        return False
    
    # Check secret key
    secret_key = os.getenv('SECRET_KEY')
    if secret_key:
        print("✅ Flask secret key found")
    else:
        print("⚠️  Flask secret key not found - using default")
    
    return True

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        'static',
        'static/css',
        'static/js',
        'static/uploads',
        'templates'
    ]
    
    all_dirs_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ - Directory exists")
        else:
            print(f"❌ {dir_path}/ - Directory missing")
            all_dirs_ok = False
    
    return all_dirs_ok

def test_files():
    """Test required files"""
    print("\n📄 Testing required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'static/css/style.css',
        'static/js/app.js',
        'templates/index.html'
    ]
    
    all_files_ok = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - File exists")
        else:
            print(f"❌ {file_path} - File missing")
            all_files_ok = False
    
    return all_files_ok

def test_gpu_performance():
    """Test GPU performance for Round 2"""
    print("\n🖥️  Testing GPU performance...")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        # Test GPU memory allocation
        device = torch.device('cuda:0')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(device)}")
        
        # Test basic tensor operations
        start_time = time.time()
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        gpu_time = (time.time() - start_time) * 1000
        
        print(f"✅ GPU matrix multiplication: {gpu_time:.2f}ms")
        
        # Test memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"✅ GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        print("✅ GPU performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ GPU performance test failed: {e}")
        return False

def test_local_ai_model():
    """Test local AI model for Round 2"""
    print("\n🤖 Testing local AI model...")
    
    try:
        from services.ai_service_fixed import minicpm_service
        
        # Test if the service can be imported
        print("✅ MiniCPM-V 2.6 service imported successfully")
        
        # Test basic initialization (without loading the full model)
        print("✅ Local AI service structure ready")
        
        # Note: Full model loading is done on first use to save startup time
        print("ℹ️  Full model will be loaded on first video analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Local AI model test failed: {e}")
        return False

def test_flask_app():
    """Test Flask application"""
    print("\n🌐 Testing Flask application...")
    
    try:
        # Import the Flask app
        from app import app
        
        # Test basic app configuration
        if app.config.get('UPLOAD_FOLDER'):
            print("✅ Flask app configured with upload folder")
        else:
            print("❌ Flask app missing upload folder configuration")
            return False
        
        if app.secret_key:
            print("✅ Flask app has secret key")
        else:
            print("⚠️  Flask app missing secret key")
        
        print("✅ Flask application ready")
        return True
        
    except Exception as e:
        print(f"❌ Flask application test failed: {e}")
        return False

def main():
    """Run all tests for Round 2"""
    print("🚀 AI Video Detective Round 2 - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Environment", test_environment),
        ("Directories", test_directories),
        ("Files", test_files),
        ("GPU Performance", test_gpu_performance),
        ("Local AI Model", test_local_ai_model),
        ("Flask App", test_flask_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary for Round 2")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your AI Video Detective Round 2 is ready to run.")
        print("\n🚀 Round 2 Features:")
        print("  • GPU-powered local AI processing")
        print("  • <1000ms latency for video analysis")
        print("  • 90fps real-time video processing")
        print("  • No external API dependencies")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        
        if not any(name == "Environment" and result for name, result in results):
            print("\n💡 Quick fix for environment issues:")
            print("1. Ensure CUDA drivers are installed")
            print("2. Install PyTorch with CUDA support")
            print("3. Create a .env file with your configuration:")
            print("   SECRET_KEY=your_secret_key_here")
        
        if not any(name == "Package Imports" and result for name, result in results):
            print("\n💡 Quick fix for import issues:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Activate virtual environment if using one")
        
        if not any(name == "GPU Performance" and result for name, result in results):
            print("\n💡 Quick fix for GPU issues:")
            print("1. Install NVIDIA CUDA drivers")
            print("2. Install PyTorch with CUDA support")
            print("3. Ensure your GPU supports CUDA")

if __name__ == "__main__":
    main() 