#!/usr/bin/env python3
"""
Test script to verify GPU fixes work correctly
Tests for:
1. GPU info errors (decode and format issues)
2. Coroutine never awaited issues
3. Model ID validation
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_gpu_service():
    """Test GPU service initialization and status methods"""
    print("🧪 Testing GPU Service...")
    
    try:
        from services.gpu_service import GPUService
        
        gpu_service = GPUService()
        
        # Test initialization
        print("  → Testing initialization...")
        await gpu_service.initialize()
        print("  ✅ GPU service initialized successfully")
        
        # Test GPU status message
        print("  → Testing GPU status message...")
        status_msg = await gpu_service.get_gpu_status_message()
        print(f"  ✅ GPU Status: {status_msg}")
        
        # Test performance metrics
        print("  → Testing performance metrics...")
        metrics = await gpu_service.get_performance_metrics()
        print(f"  ✅ Performance metrics: {metrics}")
        
        # Test memory status
        print("  → Testing memory status...")
        memory_status = await gpu_service.get_memory_status()
        print(f"  ✅ Memory status: {memory_status}")
        
        # Cleanup
        await gpu_service.cleanup()
        print("  ✅ GPU service cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU service test failed: {e}")
        return False

async def test_ai_service():
    """Test AI service initialization"""
    print("🧪 Testing AI Service...")
    
    try:
        from services.ai_service import MiniCPMV26Service
        
        ai_service = MiniCPMV26Service()
        
        # Test initialization
        print("  → Testing initialization...")
        await ai_service.initialize()
        print("  ✅ AI service initialized successfully")
        
        # Test chat response generation (should not have coroutine issues)
        print("  → Testing chat response generation...")
        response = await ai_service.generate_chat_response(
            analysis_result="Test analysis",
            analysis_type="test",
            user_focus="test",
            message="Hello",
            chat_history=[]
        )
        print(f"  ✅ Chat response generated: {response[:100]}...")
        
        # Cleanup
        ai_service.cleanup()
        print("  ✅ AI service cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI service test failed: {e}")
        return False

async def test_config():
    """Test configuration values"""
    print("🧪 Testing Configuration...")
    
    try:
        from config import Config
        
        # Test model path
        print(f"  → Model path: {Config.MINICPM_MODEL_PATH}")
        if "MiniCPM-V-2_6" in Config.MINICPM_MODEL_PATH:
            print("  ✅ Model ID is correct (contains underscore)")
        else:
            print("  ❌ Model ID is incorrect")
            return False
        
        # Test GPU config
        print(f"  → GPU device: {Config.GPU_CONFIG['device']}")
        print(f"  → GPU precision: {Config.GPU_CONFIG['precision']}")
        print("  ✅ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting GPU Fixes Test Suite...")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("GPU Service", test_gpu_service),
        ("AI Service", test_ai_service),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! GPU fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1) 