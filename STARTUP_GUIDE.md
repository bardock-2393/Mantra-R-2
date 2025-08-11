# AI Video Detective - OPTIMIZED FOR 7B MODEL ONLY 🚀

## Overview
This application has been **OPTIMIZED FOR MAXIMUM PERFORMANCE** using the Qwen2.5-VL-7B-Instruct model. All other models have been commented out to focus resources on achieving the highest possible speed, accuracy, and efficiency with the 7B model.

## 🎯 PERFORMANCE OPTIMIZATIONS IMPLEMENTED

### Model Optimizations
- **FP16 Precision**: Maximum speed with half-precision floating point
- **Flash Attention 2**: Advanced attention mechanism for speed
- **xformers**: Memory-efficient attention optimization
- **Optimized Pixel Limits**: 1024 tokens for 7B model efficiency
- **Enhanced Generation Parameters**: Optimized temperature, top-p, top-k for accuracy

### GPU Optimizations
- **Automatic Device Mapping**: Smart GPU distribution
- **Low CPU Memory Usage**: Reduced system overhead
- **Optimized Batch Processing**: Enhanced throughput
- **Memory Management**: Efficient GPU memory utilization

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Configuration
```bash
# Set environment variables
export QWEN25VL_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export HF_TOKEN="your_huggingface_token_here"  # If required
```

### 3. Run the Application
```bash
# Start the optimized 7B model application
python app.py
```

## 🔧 Configuration

### Model Settings (OPTIMIZED)
- **Model**: Qwen2.5-VL-7B-Instruct
- **Precision**: FP16 (Maximum Speed)
- **Max Length**: 8192 tokens (Optimized for 7B)
- **Temperature**: 0.1 (High Accuracy)
- **Top-p**: 0.95 (Quality)
- **Top-k**: 50 (Diversity)

### GPU Requirements
- **CUDA**: Required
- **Memory**: 8GB+ VRAM recommended
- **Precision**: FP16 for maximum speed

## 📊 Performance Metrics

### Expected Performance
- **Latency**: <1000ms target
- **Throughput**: 90+ FPS target
- **Memory**: Optimized for 7B model
- **Accuracy**: Enhanced with optimized parameters

## 🚫 Disabled Models

The following models have been **COMMENTED OUT** for 7B optimization:
- ❌ MiniCPM-V-2_6
- ❌ Qwen2.5-VL-32B-Instruct

## 🔍 Troubleshooting

### Common Issues
1. **CUDA Not Available**: Ensure NVIDIA GPU with CUDA support
2. **Memory Issues**: Reduce batch size or use FP16 precision
3. **Model Loading**: Verify HF_TOKEN if required

### Performance Tips
1. **Use FP16**: Maximum speed with acceptable accuracy
2. **Optimize Batch Size**: Balance memory and throughput
3. **Monitor GPU**: Use nvidia-smi for monitoring

## 📈 Monitoring

### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi

# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Application Monitoring
- Check `/api/health` endpoint
- Monitor `/api/model-status` for model health
- Use performance service for metrics

## 🎯 Why 7B Model Only?

1. **Speed**: 7B parameters = faster inference
2. **Efficiency**: Optimized memory usage
3. **Accuracy**: Enhanced parameters for quality
4. **Resource Focus**: All optimizations dedicated to one model
5. **Production Ready**: Stable and reliable performance

## 🚀 Next Steps

1. **Test Performance**: Run benchmark tests
2. **Monitor Metrics**: Track latency and throughput
3. **Fine-tune**: Adjust parameters if needed
4. **Scale**: Deploy with optimized settings

---

**Note**: This application is now optimized exclusively for the Qwen2.5-VL-7B-Instruct model to achieve maximum performance, accuracy, and speed. All other model endpoints have been disabled to focus resources on the 7B model optimization. 