#!/usr/bin/env python3
"""
Simple API endpoint test script
"""

import requests
import json

# Base URL - change this to match your server
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            headers = {"Content-Type": "application/json"} if data else {}
            response = requests.post(url, json=data, headers=headers)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
            
        print(f"🔍 Testing {method} {endpoint}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   Response: {json.dumps(result, indent=2)}")
                return True
            except json.JSONDecodeError:
                print(f"   Response: {response.text[:200]}...")
                return True
        else:
            print(f"   Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed to {url}")
        return False
    except Exception as e:
        print(f"❌ Error testing {endpoint}: {e}")
        return False

def main():
    """Test all API endpoints"""
    print("🚀 Testing AI Video Detective API Endpoints")
    print("=" * 50)
    
    # Test basic endpoints
    test_endpoint("/")
    test_endpoint("/api/health")
    test_endpoint("/api/session/status")
    
    # Test model switching
    test_endpoint("/api/switch-model", "POST", {"model": "qwen25vl_32b"})
    
    # Test cleanup endpoints
    test_endpoint("/api/cleanup-uploads", "POST")
    test_endpoint("/api/session/cleanup", "POST")
    
    print("\n" + "=" * 50)
    print("✅ API endpoint testing completed!")

if __name__ == "__main__":
    main()




