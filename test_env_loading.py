#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""

import os
from dotenv import load_dotenv

print("🔍 Testing environment variable loading...")

# Load .env file
load_dotenv()

# Check HF_TOKEN
hf_token = os.getenv('HF_TOKEN')
print(f"HF_TOKEN: {'Found' if hf_token else 'Not found'}")
if hf_token:
    print(f"Token preview: {hf_token[:10]}...{hf_token[-4:] if len(hf_token) > 14 else ''}")

# Check other variables
print(f"SECRET_KEY: {'Found' if os.getenv('SECRET_KEY') else 'Not found'}")
print(f"GOOGLE_API_KEY: {'Found' if os.getenv('GOOGLE_API_KEY') else 'Not found'}")

# Test config loading
try:
    from config import Config
    print(f"Config loaded successfully")
    print(f"QWEN25VL_32B_MODEL_PATH: {Config.QWEN25VL_32B_MODEL_PATH}")
    print(f"HF_TOKEN in config: {'Found' if Config.QWEN25VL_32B_CONFIG.get('hf_token') else 'Not found'}")
except Exception as e:
    print(f"Config loading failed: {e}")

print("✅ Environment test completed")
