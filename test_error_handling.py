#!/usr/bin/env python3
"""
Test script for Error Handling
This script tests that all endpoints return proper JSON instead of HTML error pages
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_endpoint_returns_json(url, method='GET', data=None, description=""):
    """Test that an endpoint returns valid JSON"""
    try:
        print(f"🔍 Testing {description}...")
        
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        # Check if response is JSON
        try:
            response_json = response.json()
            print(f"✅ {description} returned valid JSON: {response.status_code}")
            print(f"   Response keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'Not a dict'}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ {description} returned invalid JSON: {response.status_code}")
            print(f"   Content type: {response.headers.get('content-type', 'Unknown')}")
            print(f"   Response preview: {response.text[:200]}...")
            print(f"   JSON decode error: {e}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {description} request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} test failed: {e}")
        return False

def test_error_handling():
    """Test error handling for various endpoints"""
    print("🧪 TESTING ERROR HANDLING")
    print("="*50)
    
    # Base URL - adjust this to match your server
    base_url = "http://localhost:5000"
    
    tests = [
        # Test valid endpoints
        {
            'url': f"{base_url}/api/health",
            'method': 'GET',
            'description': 'Health Check Endpoint'
        },
        {
            'url': f"{base_url}/api/agent-info",
            'method': 'GET',
            'description': 'Agent Info Endpoint'
        },
        {
            'url': f"{base_url}/api/model-status",
            'method': 'GET',
            'description': 'Model Status Endpoint'
        },
        {
            'url': f"{base_url}/api/session/status",
            'method': 'GET',
            'description': 'Session Status Endpoint'
        },
        
        # Test invalid requests (should return JSON errors)
        {
            'url': f"{base_url}/api/session/invalid-session-id",
            'method': 'GET',
            'description': 'Invalid Session ID (should return 404 JSON)'
        },
        {
            'url': f"{base_url}/api/switch-model",
            'method': 'POST',
            'data': {},
            'description': 'Switch Model with Invalid Data (should return 400 JSON)'
        },
        {
            'url': f"{base_url}/api/switch-model",
            'method': 'POST',
            'data': None,
            'description': 'Switch Model with No Data (should return 400 JSON)'
        },
        
        # Test main routes
        {
            'url': f"{base_url}/upload",
            'method': 'POST',
            'data': {},
            'description': 'Upload with No File (should return 400 JSON)'
        },
        {
            'url': f"{base_url}/analyze",
            'method': 'POST',
            'data': {},
            'description': 'Analyze with No Session (should return 400 JSON)'
        },
        
        # Test chat routes
        {
            'url': f"{base_url}/chat",
            'method': 'POST',
            'data': {},
            'description': 'Chat with No Message (should return 400 JSON)'
        },
        {
            'url': f"{base_url}/chat",
            'method': 'POST',
            'data': None,
            'description': 'Chat with No Data (should return 400 JSON)'
        }
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        success = test_endpoint_returns_json(
            test['url'], 
            test['method'], 
            test.get('data'), 
            test['description']
        )
        if success:
            passed += 1
        print()
    
    # Summary
    print("="*50)
    print("📊 ERROR HANDLING TEST RESULTS")
    print("="*50)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Error handling is working correctly.")
        print("✅ All endpoints return proper JSON instead of HTML error pages.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        print("❌ Some endpoints may still return HTML error pages.")
        return False

if __name__ == "__main__":
    try:
        success = test_error_handling()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1) 