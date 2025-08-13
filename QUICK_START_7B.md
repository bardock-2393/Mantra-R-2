# 🚀 Quick Start Guide - Qwen2.5-VL-7B Model

## Overview
This guide will help you get started with the AI Video Detective using the **Qwen2.5-VL-7B-Instruct** model, which is now the **primary model** for the application.

## 🎯 What You'll Get
- **Fast Video Analysis**: Quick processing with the 7B model
- **High Quality Results**: Excellent video understanding capabilities
- **Lower Resource Usage**: 8GB+ GPU requirement (vs 40GB+ for 32B)
- **Balanced Performance**: Speed and quality optimization

## ⚡ Quick Start (5 minutes)

### 1. **Check Requirements**
```bash
# Check Python version
python --version  # Should be 3.8+

# Check GPU
nvidia-smi  # Should show 8GB+ VRAM
```

### 2. **Install Dependencies**
```bash
# Install required packages
pip install -r requirements_optimized.txt

# Or install core requirements
pip install -r requirements.txt
```

### 3. **Start the Application**
```bash
# Start with 7B model (recommended)
python start_with_model.py

# Or start with optimized settings
python start_optimized.py
```

### 4. **Access the Application**
Open your browser and go to: `http://localhost:5000`

## 🔧 Detailed Setup

### **System Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 8GB+ system RAM (16GB+ recommended)
- **Storage**: 5GB+ free space
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+

### **Environment Configuration**
```bash
# Copy environment template
cp env_example.txt .env

# Edit configuration (optional)
nano .env
```

**Basic .env configuration:**
```env
# Flask configuration
SECRET_KEY=your-secret-key-here

# Hugging Face token (optional)
HF_TOKEN=your-huggingface-token-here

# Model paths (will use defaults if not set)
QWEN25VL_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
```

## 🧪 Testing the Setup

### **Run Integration Test**
```bash
python test_7b_integration.py
```

**Expected Output:**
```
🧪 Testing Qwen2.5-VL-7B integration...
✅ Model manager imported successfully
📋 Available models: ['qwen25vl_7b', 'qwen25vl_32b']
🎯 Current model: qwen25vl_7b
✅ 7B model found in model manager
🚀 Initializing 7B model...
✅ 7B model initialized successfully
✅ 7B model integration test completed successfully!
```

### **Test Video Analysis**
1. Upload a video file (MP4, AVI, MOV, etc.)
2. Wait for the 7B model to analyze
3. Ask questions about the video content
4. Enjoy high-quality AI-powered analysis!

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### **1. CUDA Not Available**
```
❌ CUDA not available. GPU is required for Qwen2.5-VL-7B
```
**Solution:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. Insufficient GPU Memory**
```
❌ Error: Insufficient GPU memory for 7B model
   Required: 8GB+, Recommended: 16GB+
```
**Solution:**
- Close other GPU applications
- Reduce batch size in config.py
- Use smaller video files for testing

#### **3. Model Download Failed**
```
❌ Failed to load Qwen2.5-VL-7B model
```
**Solution:**
- Check internet connection
- Verify Hugging Face access
- Try again (model is ~14GB)

#### **4. Import Errors**
```
❌ ModuleNotFoundError: No module named 'transformers'
```
**Solution:**
```bash
# Install transformers from source (required for Qwen2.5-VL)
pip install git+https://github.com/huggingface/transformers accelerate
```

## 📊 Performance Tips

### **Optimal Settings**
- **GPU Memory**: 16GB+ for best performance
- **Video Length**: Keep under 10 minutes for optimal analysis
- **Frame Resolution**: Automatically optimized to 336x336
- **Batch Processing**: 2 batches enabled by default

### **Memory Optimization**
```python
# In config.py - adjust for your GPU
GPU_CONFIG = {
    'memory_limit': 16 * 1024 * 1024 * 1024,  # 16GB
    'batch_size': 2,  # Optimal for 7B model
    'use_flash_attention': True,  # Enabled for 7B
}
```

## 🔄 Model Switching

### **Switch to 32B Model (if available)**
```python
# In your code
await model_manager.switch_model('qwen25vl_32b')
```

### **Check Available Models**
```python
# List all available models
models = model_manager.get_available_models()
print(f"Available: {list(models.keys())}")
```

## 📁 File Structure

```
ai_video_detective/
├── services/
│   ├── qwen25vl_7b_service.py      # 7B model service
│   ├── qwen25vl_32b_service.py     # 32B model service
│   └── model_manager.py             # Model management
├── start_with_model.py              # Start with 7B model
├── start_optimized.py               # Optimized startup
├── test_7b_integration.py           # 7B model test
└── config.py                        # Configuration
```

## 🎉 Success Indicators

### **When Everything Works:**
✅ Model loads in under 2 minutes  
✅ GPU memory usage: 6-8GB  
✅ Video analysis completes successfully  
✅ Chat responses are generated quickly  
✅ No CUDA errors in console  

### **Performance Metrics:**
- **Model Loading**: 1-2 minutes (vs 3-5 minutes for 32B)
- **Video Analysis**: 2-5 minutes for 1-minute videos
- **Memory Usage**: 6-8GB GPU memory
- **Response Time**: 1-3 seconds for chat responses

## 🆘 Getting Help

### **Check Logs**
```bash
# Look for error messages in console output
# Common indicators:
# ✅ = Success
# ⚠️ = Warning
# ❌ = Error
```

### **Verify Installation**
```bash
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')"
```

### **Common Commands**
```bash
# Start application
python start_with_model.py

# Test model
python test_7b_integration.py

# Check GPU
nvidia-smi

# Monitor memory
watch -n 1 nvidia-smi
```

## 🚀 Next Steps

1. **Test with a short video** (1-2 minutes)
2. **Try different analysis types** (brief, comprehensive, technical)
3. **Experiment with chat functionality**
4. **Monitor performance metrics**
5. **Explore advanced features**

## 📚 Additional Resources

- **Full Documentation**: README.md
- **Integration Summary**: QWEN25VL_7B_INTEGRATION_SUMMARY.md
- **Configuration Guide**: config.py
- **API Reference**: routes/ directory

---

**🎯 You're all set!** The 7B model provides an excellent balance of performance and quality for most video analysis needs. Enjoy your AI-powered video detective experience!

