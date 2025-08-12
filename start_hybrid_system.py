#!/usr/bin/env python3
"""
Start Hybrid System - DeepStream + 7B Model + Vector Search
Simple startup script to test and run the hybrid analysis system
"""

import asyncio
import sys
import os

def print_banner():
    """Print the system banner"""
    print("=" * 70)
    print("🚀 AI Video Detective - Hybrid System")
    print("🔍 DeepStream + 7B Model + Vector Search")
    print("=" * 70)
    print()

async def check_dependencies():
    """Check if all required dependencies are available"""
    print("📦 Checking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch for GPU acceleration',
        'cv2': 'OpenCV for video processing',
        'numpy': 'NumPy for numerical operations',
        'transformers': 'Hugging Face transformers for 7B model',
        'sentence_transformers': 'Sentence transformers for vector search',
        'faiss': 'Faiss for vector similarity search'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
        except ImportError:
            print(f"❌ {dep}: {description} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing packages:")
        for dep in missing_deps:
            if dep == 'faiss':
                print(f"   pip install faiss-gpu")
            elif dep == 'cv2':
                print(f"   pip install opencv-python")
            else:
                print(f"   pip install {dep}")
        return False
    
    print("✅ All dependencies available!")
    return True

async def test_system():
    """Test the hybrid system"""
    print("\n🧪 Testing hybrid system...")
    
    try:
        # Import and test the hybrid service
        from services.hybrid_analysis_service import HybridAnalysisService
        
        print("🔗 Initializing hybrid analysis service...")
        hybrid_service = HybridAnalysisService()
        await hybrid_service.initialize()
        
        if hybrid_service.is_initialized:
            print("✅ Hybrid system initialized successfully!")
            return True
        else:
            print("❌ Hybrid system failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    deps_ok = await check_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Test system
    system_ok = await test_system()
    if not system_ok:
        print("\n❌ System test failed. Please check the configuration.")
        return
    
    print("\n🎉 Hybrid system is ready!")
    print("\nNext steps:")
    print("1. Run the main application: python main.py")
    print("2. Or test with a video: python test_hybrid_system.py")
    print("3. Access the web interface at: http://localhost:8000")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
