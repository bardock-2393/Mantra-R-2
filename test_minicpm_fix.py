#!/usr/bin/env python3
"""
Test script for MiniCPM-V-2_6 fixes
Tests the chat interface and error handling
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.ai_service import MiniCPMV26Service

async def test_minicpm_fix():
    """Test the fixed MiniCPM-V-2_6 service"""
    print("🧪 Testing MiniCPM-V-2_6 fixes...")
    
    try:
        # Initialize the service
        service = MiniCPMV26Service()
        print("✅ Service created successfully")
        
        # Test initialization
        print("🔄 Initializing service...")
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Test text generation
        print("🔄 Testing text generation...")
        test_prompt = "Hello, this is a test message. Please respond with a simple greeting."
        response = service._generate_analysis(test_prompt)
        print(f"✅ Text generation successful: {response[:100]}...")
        
        # Test chat response
        print("🔄 Testing chat response...")
        chat_response = await service.generate_chat_response(
            analysis_result="Test analysis result",
            analysis_type="general",
            user_focus="testing",
            message="Hello, how are you?",
            chat_history=[]
        )
        print(f"✅ Chat response successful: {chat_response[:100]}...")
        
        print("🎉 All tests passed! MiniCPM-V-2_6 is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'service' in locals():
            service.cleanup()
    
    return True

if __name__ == "__main__":
    print("🚀 Starting MiniCPM-V-2_6 fix test...")
    success = asyncio.run(test_minicpm_fix())
    
    if success:
        print("✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 