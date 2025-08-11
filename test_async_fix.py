#!/usr/bin/env python3
"""
Test script to verify async method signatures are correct
"""

import inspect
import asyncio

def test_async_methods():
    """Test that async methods are properly defined"""
    
    # Test ModelManager methods
    print("🔍 Testing ModelManager async methods...")
    
    # Import without executing the full initialization
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        # Import the module without executing it
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_manager", "services/model_manager.py")
        model_manager_module = importlib.util.module_from_spec(spec)
        
        # Check if the class exists and has the right methods
        if hasattr(model_manager_module, 'ModelManager'):
            ModelManager = model_manager_module.ModelManager
            
            # Check analyze_video method
            if hasattr(ModelManager, 'analyze_video'):
                method = getattr(ModelManager, 'analyze_video')
                if inspect.iscoroutinefunction(method):
                    print("✅ ModelManager.analyze_video is async")
                else:
                    print("❌ ModelManager.analyze_video is NOT async")
            else:
                print("❌ ModelManager.analyze_video method not found")
            
            # Check generate_chat_response method
            if hasattr(ModelManager, 'generate_chat_response'):
                method = getattr(ModelManager, 'generate_chat_response')
                if inspect.iscoroutinefunction(method):
                    print("✅ ModelManager.generate_chat_response is async")
                else:
                    print("❌ ModelManager.generate_chat_response is NOT async")
            else:
                print("❌ ModelManager.generate_chat_response method not found")
            
            # Check switch_model method
            if hasattr(ModelManager, 'switch_model'):
                method = getattr(ModelManager, 'switch_model')
                if inspect.iscoroutinefunction(method):
                    print("✅ ModelManager.switch_model is async")
                else:
                    print("❌ ModelManager.switch_model is NOT async")
            else:
                print("❌ ModelManager.switch_model method not found")
                
        else:
            print("❌ ModelManager class not found")
            
    except Exception as e:
        print(f"⚠️ Could not test ModelManager: {e}")
    
    print("\n🔍 Testing service async methods...")
    
    try:
        # Test MiniCPM service
        spec = importlib.util.spec_from_file_location("ai_service_fixed", "services/ai_service_fixed.py")
        minicpm_module = importlib.util.module_from_spec(spec)
        
        if hasattr(minicpm_module, 'MiniCPMV26Service'):
            MiniCPMService = minicpm_module.MiniCPMV26Service
            
            # Check analyze_video method
            if hasattr(MiniCPMService, 'analyze_video'):
                method = getattr(MiniCPMService, 'analyze_video')
                if inspect.iscoroutinefunction(method):
                    print("✅ MiniCPMService.analyze_video is async")
                else:
                    print("❌ MiniCPMService.analyze_video is NOT async")
            else:
                print("❌ MiniCPMService.analyze_video method not found")
            
            # Check generate_chat_response method
            if hasattr(MiniCPMService, 'generate_chat_response'):
                method = getattr(MiniCPMService, 'generate_chat_response')
                if inspect.iscoroutinefunction(method):
                    print("✅ MiniCPMService.generate_chat_response is async")
                else:
                    print("❌ MiniCPMService.generate_chat_response is NOT async")
            else:
                print("❌ MiniCPMService.generate_chat_response method not found")
            
            # Check initialize method
            if hasattr(MiniCPMService, 'initialize'):
                method = getattr(MiniCPMService, 'initialize')
                if inspect.iscoroutinefunction(method):
                    print("✅ MiniCPMService.initialize is async")
                else:
                    print("❌ MiniCPMService.initialize is NOT async")
            else:
                print("❌ MiniCPMService.initialize method not found")
                
        else:
            print("❌ MiniCPMService class not found")
            
    except Exception as e:
        print(f"⚠️ Could not test MiniCPM service: {e}")
    
    try:
        # Test Qwen service
        spec = importlib.util.spec_from_file_location("qwen25vl_service", "services/qwen25vl_service.py")
        qwen_module = importlib.util.module_from_spec(spec)
        
        if hasattr(qwen_module, 'Qwen25VLService'):
            QwenService = qwen_module.Qwen25VLService
            
            # Check analyze_video method
            if hasattr(QwenService, 'analyze_video'):
                method = getattr(QwenService, 'analyze_video')
                if inspect.iscoroutinefunction(method):
                    print("✅ QwenService.analyze_video is async")
                else:
                    print("❌ QwenService.analyze_video is NOT async")
            else:
                print("❌ QwenService.analyze_video method not found")
            
            # Check generate_chat_response method
            if hasattr(QwenService, 'generate_chat_response'):
                method = getattr(QwenService, 'generate_chat_response')
                if inspect.iscoroutinefunction(method):
                    print("✅ QwenService.generate_chat_response is async")
                else:
                    print("❌ QwenService.generate_chat_response is NOT async")
            else:
                print("❌ QwenService.generate_chat_response method not found")
            
            # Check initialize method
            if hasattr(QwenService, 'initialize'):
                method = getattr(QwenService, 'initialize')
                if inspect.iscoroutinefunction(method):
                    print("✅ QwenService.initialize is async")
                else:
                    print("❌ QwenService.initialize is NOT async")
            else:
                print("❌ QwenService.initialize method not found")
                
        else:
            print("❌ QwenService class not found")
            
    except Exception as e:
        print(f"⚠️ Could not test Qwen service: {e}")

if __name__ == "__main__":
    test_async_methods() 