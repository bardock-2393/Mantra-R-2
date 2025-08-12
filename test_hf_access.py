#!/usr/bin/env python3
"""
Test script to check HuggingFace model access
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

print("🔍 Testing HuggingFace access...")

# Load .env file
load_dotenv()

# Get HF token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("❌ No HF_TOKEN found in environment")
    exit(1)

print(f"🔑 HF Token found: {hf_token[:10]}...{hf_token[-4:] if len(hf_token) > 14 else ''}")

# Test login
try:
    login(token=hf_token)
    print("✅ Login successful")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# Test API access
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"✅ API access successful - User: {user_info.get('name', 'Unknown')}")
except Exception as e:
    print(f"❌ API access failed: {e}")
    exit(1)

# Test model access
model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
try:
    model_info = api.model_info(model_name)
    print(f"✅ Model access successful: {model_name}")
    print(f"   Model ID: {model_info.modelId}")
    print(f"   Last Modified: {model_info.lastModified}")
    print(f"   Tags: {model_info.tags}")
except Exception as e:
    print(f"❌ Model access failed: {e}")
    print("   This model might require special access or the token doesn't have permission")
    
    # Try to get more info about the error
    if "401" in str(e):
        print("   Error 401: Unauthorized - Check if your token has access to this model")
    elif "404" in str(e):
        print("   Error 404: Model not found - Check the model name")
    elif "403" in str(e):
        print("   Error 403: Forbidden - Token doesn't have permission for this model")

print("✅ HuggingFace access test completed")
