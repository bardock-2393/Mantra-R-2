#!/usr/bin/env python3
"""
Test script to check available Qwen models
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

print("🔍 Checking available Qwen models...")

# Load .env file
load_dotenv()

# Get HF token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("❌ No HF_TOKEN found in environment")
    exit(1)

# Login
try:
    login(token=hf_token)
    api = HfApi()
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# Search for Qwen models
print("🔍 Searching for Qwen models...")
try:
    models = api.list_models(
        search="Qwen",
        limit=20,
        sort="downloads",
        direction=-1
    )
    
    print("📋 Available Qwen models (by popularity):")
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model.modelId}")
        print(f"    Downloads: {model.downloads:,}")
        print(f"    Likes: {model.likes:,}")
        print(f"    Tags: {', '.join(model.tags[:5]) if model.tags else 'None'}")
        print()
        
        # Check specific models we're interested in
        if "Qwen2.5-VL-32B-Instruct" in model.modelId:
            print("   ⭐ This is the model we want!")
        elif "Qwen2.5-VL" in model.modelId and "32B" in model.modelId:
            print("   🔍 Alternative 32B model found!")
        elif "Qwen2.5-VL" in model.modelId:
            print("   🔍 Alternative Qwen2.5-VL model found!")
            
except Exception as e:
    print(f"❌ Search failed: {e}")

# Check specific models
specific_models = [
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-32B",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B"
]

print("\n🔍 Checking specific model access...")
for model_name in specific_models:
    try:
        model_info = api.model_info(model_name)
        print(f"✅ {model_name}: Accessible")
        print(f"   Tags: {', '.join(model_info.tags[:5]) if model_info.tags else 'None'}")
    except Exception as e:
        print(f"❌ {model_name}: {e}")

print("\n✅ Model availability check completed")
