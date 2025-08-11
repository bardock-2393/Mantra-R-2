#!/usr/bin/env python3
"""
Test Script for Hybrid Analysis System
Tests DeepStream + 7B Model + Vector Search integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

async def test_hybrid_system():
    """Test the hybrid analysis system components"""
    print("🧪 Testing Hybrid Analysis System...")
    print("=" * 50)
    
    try:
        # Test 1: Import components
        print("📦 Test 1: Importing components...")
        from services.hybrid_analysis_service import hybrid_analysis_service
        from models.deepstream_pipeline import deepstream_pipeline
        from services.vector_search_service import VectorSearchService
        print("✅ All components imported successfully")
        
        # Test 2: Initialize services
        print("\n🚀 Test 2: Initializing services...")
        await hybrid_analysis_service.initialize()
        print("✅ Hybrid service initialized")
        
        # Test 3: Check GPU service
        print("\n🖥️ Test 3: Checking GPU service...")
        gpu_status = await hybrid_analysis_service.gpu_service.get_gpu_status()
        print(f"✅ GPU Status: {gpu_status}")
        
        # Test 4: Check DeepStream pipeline
        print("\n🔍 Test 4: Checking DeepStream pipeline...")
        deepstream_status = await deepstream_pipeline.get_processing_status()
        print(f"✅ DeepStream Status: {deepstream_status}")
        
        # Test 5: Check Vector Search
        print("\n💾 Test 5: Checking Vector Search...")
        vector_service = VectorSearchService()
        print("✅ Vector Search service ready")
        
        # Test 6: Test with sample video
        print("\n🎬 Test 6: Testing with sample video...")
        sample_video = "BMW M4 - Ultimate Racetrack - BMW Canada (720p, h264).mp4"
        
        if os.path.exists(sample_video):
            print(f"📹 Found sample video: {sample_video}")
            
            # Start hybrid analysis
            print("🔍 Starting hybrid analysis...")
            results = await hybrid_analysis_service.analyze_video_hybrid(sample_video, "test")
            
            if 'error' not in results:
                print("✅ Hybrid analysis completed successfully!")
                print(f"📊 Session ID: {results.get('session_id')}")
                print(f"⏱️ Processing time: {results.get('performance_metrics', {}).get('total_processing_time', 0):.2f}s")
                
                # Test search
                print("\n🔍 Testing vector search...")
                search_results = await hybrid_analysis_service.search_analysis_results(
                    results['session_id'], 
                    "car racing", 
                    5
                )
                print(f"✅ Search completed: {len(search_results)} results found")
                
            else:
                print(f"❌ Analysis failed: {results['error']}")
        else:
            print(f"⚠️ Sample video not found: {sample_video}")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            await hybrid_analysis_service.cleanup()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    print("🚀 AI Video Detective - Hybrid System Test")
    print("Testing DeepStream + 7B Model + Vector Search integration")
    print()
    
    # Run the test
    asyncio.run(test_hybrid_system()) 