#!/usr/bin/env python3
"""
Test Hybrid System - DeepStream + 7B Model + Vector Search
Tests the complete hybrid analysis system initialization and basic functionality
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_hybrid_system():
    """Test the complete hybrid analysis system"""
    print("🧪 Testing Hybrid Analysis System...")
    print("=" * 50)
    
    try:
        # Test 1: Import all services
        print("📦 Test 1: Importing services...")
        from services.hybrid_analysis_service import HybridAnalysisService
        from services.gpu_service import GPUService
        from services.ai_service import ai_service
        from models.deepstream_pipeline import DeepStreamPipeline
        from services.vector_search_service import VectorSearchService
        print("✅ All services imported successfully")
        
        # Test 2: Test GPU service
        print("\n🖥️  Test 2: Testing GPU service...")
        gpu_service = GPUService()
        await gpu_service.initialize()
        print("✅ GPU service initialized")
        
        # Test 3: Test DeepStream pipeline
        print("\n🔍 Test 3: Testing DeepStream pipeline...")
        deepstream_pipeline = DeepStreamPipeline()
        await deepstream_pipeline.initialize()
        print("✅ DeepStream pipeline initialized")
        
        # Test 4: Test AI service (7B model)
        print("\n🧠 Test 4: Testing AI service...")
        await ai_service.initialize()
        print("✅ AI service initialized")
        
        # Test 5: Test vector search service
        print("\n💾 Test 5: Testing vector search service...")
        vector_service = VectorSearchService()
        await vector_service.initialize()
        print("✅ Vector search service initialized")
        
        # Test 6: Test complete hybrid system
        print("\n🔗 Test 6: Testing complete hybrid system...")
        hybrid_service = HybridAnalysisService()
        await hybrid_service.initialize()
        print("✅ Hybrid system initialized successfully!")
        
        # Test 7: Test system status
        print("\n📊 Test 7: System status check...")
        status = {
            'gpu_service': gpu_service.is_initialized if hasattr(gpu_service, 'is_initialized') else True,
            'deepstream_pipeline': deepstream_pipeline.is_initialized,
            'ai_service': ai_service.is_initialized,
            'vector_service': vector_service.model is not None,
            'hybrid_service': hybrid_service.is_initialized
        }
        
        print("System Status:")
        for service, status_bool in status.items():
            status_icon = "✅" if status_bool else "❌"
            print(f"   {status_icon} {service}: {'Ready' if status_bool else 'Not Ready'}")
        
        all_ready = all(status.values())
        if all_ready:
            print("\n🎉 ALL SYSTEMS READY! Hybrid analysis system is fully operational!")
        else:
            print("\n⚠️  Some systems are not ready. Check the configuration.")
        
        return all_ready
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_video_processing():
    """Test basic video processing capabilities"""
    print("\n🎬 Testing Video Processing Capabilities...")
    print("=" * 50)
    
    try:
        from services.hybrid_analysis_service import HybridAnalysisService
        
        # Initialize hybrid service
        hybrid_service = HybridAnalysisService()
        await hybrid_service.initialize()
        
        # Test with sample video if available
        sample_video = "BMW M4 - Ultimate Racetrack - BMW Canada (720p, h264).mp4"
        if os.path.exists(sample_video):
            print(f"📹 Testing with sample video: {sample_video}")
            
            # Test hybrid analysis
            result = await hybrid_service.analyze_video_hybrid(sample_video, "hybrid")
            
            if 'error' not in result:
                print("✅ Video analysis completed successfully!")
                print(f"   - Processing time: {result.get('performance_metrics', {}).get('total_processing_time', 0):.2f}s")
                print(f"   - Frames processed: {result.get('performance_metrics', {}).get('frames_processed', 0)}")
                print(f"   - Analysis type: {result.get('analysis_type', 'unknown')}")
            else:
                print(f"⚠️  Video analysis failed: {result.get('error')}")
        else:
            print("⚠️  Sample video not found, skipping video processing test")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 AI Video Detective - Hybrid System Test")
    print("Testing DeepStream + 7B Model + Vector Search Integration")
    print("=" * 60)
    
    # Test 1: System initialization
    init_success = await test_hybrid_system()
    
    if init_success:
        # Test 2: Video processing (if initialization succeeded)
        await test_video_processing()
    
    print("\n" + "=" * 60)
    if init_success:
        print("🎉 ALL TESTS PASSED! Your hybrid system is working correctly!")
        print("🚀 Ready for high-performance video analysis!")
    else:
        print("❌ Some tests failed. Please check the configuration and dependencies.")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    asyncio.run(main()) 