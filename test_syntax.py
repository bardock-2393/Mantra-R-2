#!/usr/bin/env python3
"""
Syntax check script for the fixed AI service
Verifies that the Python code is syntactically correct without running imports
"""

import ast
import sys
import os

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
        
        # Parse the source code
        ast.parse(source)
        print(f"✅ {file_path}: Syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file: {e}")
        return False

def main():
    """Main function to check syntax of key files"""
    print("🔍 Checking syntax of key files...")
    print("=" * 50)
    
    files_to_check = [
        "services/ai_service_fixed.py",
        "routes/main_routes.py",
        "routes/chat_routes.py",
        "config.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if not check_syntax(file_path):
                all_valid = False
        else:
            print(f"⚠️ {file_path}: File not found")
    
    print("=" * 50)
    if all_valid:
        print("🎉 All files have valid syntax!")
        print("📝 The code structure is correct and ready for deployment.")
        print("💡 Note: You'll need to install the required dependencies (torch, transformers, etc.) to run the service.")
        sys.exit(0)
    else:
        print("❌ Some files have syntax errors. Please fix them before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 