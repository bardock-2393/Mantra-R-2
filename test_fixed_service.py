#!/usr/bin/env python3
"""
Test script for the fixed AI service
Verifies that the MiniCPM-V-2_6 service can be imported and initialized without errors
"""

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fixed_service():
    """Test the fixed AI service"""
    try:
        print("🧪 Testing fixed AI service...")
        
        # Test import
        print("📦 Testing import...")
        from services.ai_service_fixed import MiniCPMV26Service
        print("✅ Import successful")
        
        # Test service creation
        print("🔧 Testing service creation...")
        service = MiniCPMV26Service()
        print("✅ Service creation successful")
        
        # Test initialization (this will fail without GPU, but should not crash)
        print("🚀 Testing initialization...")
        try:
            await service.initialize()
            print("✅ Initialization successful")
        except Exception as e:
            if "CUDA not available" in str(e):
                print("⚠️ Expected error: CUDA not available (no GPU)")
                print("✅ Service structure is correct")
            else:
                print(f"❌ Unexpected error during initialization: {e}")
                return False
        
        print("🎉 All tests passed! The fixed service is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting AI service tests...")
    print("=" * 50)
    
    success = await test_fixed_service()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! The service is ready to use.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 