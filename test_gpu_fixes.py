#!/usr/bin/env python3
"""
Test script to verify GPU fixes and model initialization
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.gpu_service import GPUService
from services.ai_service import MiniCPMV26Service
from config import Config

async def test_gpu_service():
    """Test GPU service fixes"""
    print("🧪 Testing GPU Service fixes...")
    
    try:
        gpu_service = GPUService()
        await gpu_service.initialize()
        
        # Test GPU info retrieval
        gpu_info = await gpu_service.get_gpu_info()
        print(f"✅ GPU Info: {gpu_info}")
        
        # Test performance metrics
        metrics = await gpu_service.get_performance_metrics()
        print(f"✅ Performance Metrics: {metrics}")
        
        # Test GPU status message
        status_msg = await gpu_service.get_gpu_status_message()
        print(f"✅ GPU Status: {status_msg}")
        
        # Test memory status
        mem_status = await gpu_service.get_memory_status()
        print(f"✅ Memory Status: {mem_status}")
        
        print("✅ GPU Service tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ GPU Service test failed: {e}")
        return False

async def test_ai_service_initialization():
    """Test AI service initialization without unsupported parameters"""
    print("🧪 Testing AI Service initialization...")
    
    try:
        ai_service = MiniCPMV26Service()
        
        # Test initialization
        await ai_service.initialize()
        
        if ai_service.is_initialized:
            print("✅ AI Service initialized successfully!")
            print(f"✅ Model device: {ai_service.device}")
            print(f"✅ Tokenizer loaded: {ai_service.tokenizer is not None}")
            print(f"✅ Model loaded: {ai_service.model is not None}")
            
            # Cleanup
            ai_service.cleanup()
            return True
        else:
            print("❌ AI Service failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ AI Service test failed: {e}")
        return False

def test_config_values():
    """Test configuration values"""
    print("🧪 Testing configuration values...")
    
    try:
        # Check MiniCPM model path
        print(f"✅ MINICPM_MODEL_PATH: {Config.MINICPM_MODEL_PATH}")
        
        # Check GPU config
        print(f"✅ GPU device: {Config.GPU_CONFIG['device']}")
        print(f"✅ GPU precision: {Config.GPU_CONFIG['precision']}")
        
        # Check MiniCPM config (should not have unsupported params)
        print(f"✅ MiniCPM config keys: {list(Config.MINICPM_CONFIG.keys())}")
        
        # Verify no unsupported parameters
        unsupported_params = ['use_flash_attention', 'quantization']
        for param in unsupported_params:
            if param not in Config.MINICPM_CONFIG:
                print(f"✅ {param} parameter removed (as expected)")
            else:
                print(f"❌ {param} parameter still present (should be removed)")
                return False
        
        print("✅ Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting GPU fixes verification tests...\n")
    
    tests = [
        ("Configuration Values", test_config_values),
        ("GPU Service", test_gpu_service),
        ("AI Service Initialization", test_ai_service_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        
        results.append((test_name, result))
        print()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 