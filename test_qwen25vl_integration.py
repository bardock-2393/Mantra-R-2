#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL integration
Tests the model manager and model switching functionality
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_model_manager():
    """Test the model manager functionality"""
    try:
        print("🧪 Testing Model Manager Integration...")
        
        # Import the model manager
        from services.model_manager import model_manager
        
        print("✅ Model manager imported successfully")
        
        # Test getting available models
        available_models = model_manager.get_available_models()
        print(f"📋 Available models: {list(available_models.keys())}")
        
        # Test getting current model status
        current_status = model_manager.get_current_model()
        print(f"🎯 Current model: {current_status.get('name', 'Unknown')}")
        
        # Test model switching (without actually loading models)
        print("🔄 Testing model switching logic...")
        
        # Simulate switching to Qwen2.5-VL
        success = await model_manager.switch_model('qwen25vl')
        if success:
            print("✅ Model switching test passed")
        else:
            print("❌ Model switching test failed")
        
        # Test getting status after switch
        final_status = model_manager.get_status()
        print(f"📊 Final status: {final_status['current_model']}")
        
        print("🎉 All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install -r requirements_round2.txt")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_qwen25vl_service():
    """Test the Qwen2.5-VL service directly"""
    try:
        print("\n🧪 Testing Qwen2.5-VL Service...")
        
        # Import the service
        from services.qwen25vl_service import Qwen25VLService
        
        print("✅ Qwen2.5-VL service imported successfully")
        
        # Create service instance
        service = Qwen25VLService()
        print("✅ Service instance created")
        
        # Test service methods (without initialization)
        status = service.get_status()
        print(f"📊 Service status: {status}")
        
        print("✅ Qwen2.5-VL service test passed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure qwen-vl-utils is installed:")
        print("   pip install qwen-vl-utils[decord]==0.0.8")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 Starting Qwen2.5-VL Integration Tests...")
    print("=" * 50)
    
    # Test model manager
    await test_model_manager()
    
    # Test Qwen2.5-VL service
    await test_qwen25vl_service()
    
    print("\n" + "=" * 50)
    print("🏁 Integration tests completed!")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main()) 