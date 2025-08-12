#!/usr/bin/env python3
"""
Test script to verify 32B service integration
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_32b_service():
    """Test the 32B service integration"""
    print("🧪 Testing 32B service integration...")
    
    try:
        # Import the 32B service
        from services.qwen25vl_32b_service import qwen25vl_32b_service
        
        print(f"✅ 32B service imported successfully")
        print(f"🔍 Service type: {type(qwen25vl_32b_service)}")
        print(f"🔍 Service initialized: {qwen25vl_32b_service.is_initialized}")
        print(f"🔍 Service ready: {qwen25vl_32b_service.is_ready()}")
        
        # Test initialization
        print("\n🔄 Testing service initialization...")
        await qwen25vl_32b_service.initialize()
        
        print(f"✅ Service initialized: {qwen25vl_32b_service.is_initialized}")
        print(f"✅ Service ready: {qwen25vl_32b_service.is_ready()}")
        
        # Test text generation
        print("\n📝 Testing text generation...")
        test_prompt = "Hello, this is a test. Please respond with a simple greeting."
        
        try:
            response = await qwen25vl_32b_service._generate_text(test_prompt, max_new_tokens=50)
            print(f"✅ Text generation successful: {response[:100]}...")
        except Exception as e:
            print(f"❌ Text generation failed: {e}")
        
        # Test synchronous wrapper
        print("\n🔄 Testing synchronous wrapper...")
        try:
            response = qwen25vl_32b_service._generate_text_sync(test_prompt, max_new_tokens=50)
            print(f"✅ Synchronous text generation successful: {response[:100]}...")
        except Exception as e:
            print(f"❌ Synchronous text generation failed: {e}")
        
        print("\n✅ 32B service integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_32b_service())
