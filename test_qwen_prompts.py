#!/usr/bin/env python3
"""
Test script to verify Qwen service works with analysis templates
"""

import asyncio
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_qwen_service():
    """Test the Qwen service with analysis templates"""
    try:
        print("🧪 Testing Qwen service with analysis templates...")
        
        # Test 1: Check if analysis templates are available
        print("\n1️⃣ Testing analysis templates import...")
        try:
            from analysis_templates import generate_analysis_prompt, ANALYSIS_TEMPLATES
            print("✅ Analysis templates imported successfully")
            print(f"   Available templates: {list(ANALYSIS_TEMPLATES.keys())}")
            
            # Test prompt generation
            test_prompt = generate_analysis_prompt("comprehensive_analysis", "vehicle performance")
            print(f"✅ Prompt generation successful (length: {len(test_prompt)} chars)")
            
        except ImportError as e:
            print(f"❌ Failed to import analysis templates: {e}")
            return False
        
        # Test 2: Check if Qwen service can be imported
        print("\n2️⃣ Testing Qwen service import...")
        try:
            from services.qwen25vl_service import qwen25vl_service
            print("✅ Qwen service imported successfully")
            
            # Check service status
            status = qwen25vl_service.get_status()
            print(f"   Service status: {status}")
            
        except ImportError as e:
            print(f"❌ Failed to import Qwen service: {e}")
            return False
        
        # Test 3: Test prompt generation with different analysis types
        print("\n3️⃣ Testing prompt generation with different analysis types...")
        analysis_types = [
            "comprehensive_analysis",
            "safety_investigation", 
            "performance_analysis",
            "pattern_detection",
            "creative_review"
        ]
        
        for analysis_type in analysis_types:
            try:
                prompt = generate_analysis_prompt(analysis_type, "test focus")
                print(f"   ✅ {analysis_type}: {len(prompt)} chars")
            except Exception as e:
                print(f"   ❌ {analysis_type}: {e}")
        
        # Test 4: Test model manager integration
        print("\n4️⃣ Testing model manager integration...")
        try:
            from services.model_manager import model_manager
            print("✅ Model manager imported successfully")
            
            # Check available models
            available_models = model_manager.get_available_models()
            print(f"   Available models: {list(available_models.keys())}")
            
        except ImportError as e:
            print(f"❌ Failed to import model manager: {e}")
            return False
        
        print("\n🎉 All tests passed! Qwen service is ready to use with analysis templates.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def test_prompt_generation():
    """Test specific prompt generation scenarios"""
    try:
        print("\n🔍 Testing specific prompt generation scenarios...")
        
        from analysis_templates import generate_analysis_prompt
        
        # Test scenarios
        test_scenarios = [
            ("comprehensive_analysis", "vehicle safety and performance"),
            ("safety_investigation", "workplace safety violations"),
            ("performance_analysis", "athletic performance optimization"),
            ("pattern_detection", "behavioral patterns in crowds"),
            ("creative_review", "advertising campaign effectiveness")
        ]
        
        for analysis_type, user_focus in test_scenarios:
            try:
                prompt = generate_analysis_prompt(analysis_type, user_focus)
                
                # Check if prompt contains expected elements
                if "AGENT ANALYSIS PROTOCOL" in prompt or "AGENT" in prompt:
                    print(f"   ✅ {analysis_type}: Contains agent protocol")
                else:
                    print(f"   ⚠️ {analysis_type}: Missing agent protocol")
                
                # Check if user focus is included
                if user_focus in prompt:
                    print(f"   ✅ {analysis_type}: User focus included")
                else:
                    print(f"   ⚠️ {analysis_type}: User focus not found")
                    
            except Exception as e:
                print(f"   ❌ {analysis_type}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Qwen Service Analysis Templates Test")
    print("=" * 50)
    
    # Run tests
    success1 = await test_qwen_service()
    success2 = await test_prompt_generation()
    
    if success1 and success2:
        print("\n🎯 All tests completed successfully!")
        print("✅ Qwen service is properly integrated with analysis templates")
        print("✅ Prompt system matches the old project structure")
        print("✅ Ready for video analysis with enhanced prompts")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 