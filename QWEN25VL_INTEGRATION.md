# Qwen2.5-VL-7B-Instruct Integration

This document describes the integration of the Qwen2.5-VL-7B-Instruct model into the AI Video Detective application.

## Overview

The Qwen2.5-VL-7B-Instruct model is a state-of-the-art vision-language model that provides enhanced video understanding capabilities compared to the existing MiniCPM-V-2_6 model.

## Key Features

- **Enhanced Video Understanding**: Better comprehension of long videos (over 1 hour)
- **Event Capture**: Ability to pinpoint relevant video segments
- **Visual Localization**: Accurate object localization with bounding boxes
- **Structured Outputs**: Better handling of forms, tables, and structured data
- **Dynamic Resolution**: Optimized for various video resolutions and frame rates

## Model Architecture

- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Parameters**: 7 billion
- **Vision Encoder**: Enhanced ViT with SwiGLU and RMSNorm
- **Temporal Understanding**: Dynamic FPS sampling with mRoPE
- **Context Length**: Up to 32,768 tokens (extensible to 64k for videos)

## Installation

### 1. Install Dependencies

```bash
# Install from source (required for Qwen2.5-VL support)
pip install git+https://github.com/huggingface/transformers accelerate

# Install Qwen2.5-VL utilities
pip install qwen-vl-utils[decord]==0.0.8

# Install other requirements
pip install -r requirements_round2.txt
```

### 2. Environment Variables

Add to your `.env` file:

```bash
# Qwen2.5-VL Model Configuration
QWEN25VL_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Model Selection

The application now includes a model selection dropdown in the upload interface:

1. **MiniCPM-V-2_6**: Fast and efficient for quick analysis
2. **Qwen2.5-VL-7B-Instruct**: Advanced analysis with enhanced capabilities

### Switching Models

Users can switch between models at any time using the dropdown. The system will:

1. Clean up the current model
2. Initialize the new model
3. Maintain the same analysis interface

### API Endpoints

New endpoints for model management:

- `POST /api/switch-model`: Switch between models
- `GET /api/model-status`: Get current model status

## Implementation Details

### Service Architecture

```
services/
├── model_manager.py          # Manages model switching
├── qwen25vl_service.py      # Qwen2.5-VL specific service
├── ai_service_fixed.py      # MiniCPM service (existing)
└── ...
```

### Model Manager

The `ModelManager` class provides:

- Unified interface for all models
- Automatic model switching
- Resource cleanup
- Status monitoring

### Qwen2.5-VL Service

The `Qwen25VLService` class handles:

- Model initialization and GPU optimization
- Video analysis with enhanced prompts
- Chat responses with context awareness
- Memory management and cleanup

## Configuration

### Performance Optimization

```python
# In config.py
QWEN25VL_CONFIG = {
    'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'max_length': 32768,
    'temperature': 0.2,
    'top_p': 0.9,
    'top_k': 40,
    'chat_temperature': 0.3
}
```

### Resolution Control

```python
# Optimize for performance vs. quality
min_pixels = 256 * 28 * 28    # 256 tokens
max_pixels = 1280 * 28 * 28   # 1280 tokens
```

## Testing

Run the integration test:

```bash
python test_qwen25vl_integration.py
```

## Performance Considerations

### Memory Usage

- **Model Size**: ~14GB VRAM for full precision
- **Optimization**: Uses FP16 by default
- **Cleanup**: Automatic memory cleanup on model switch

### Speed vs. Quality

- **MiniCPM-V-2_6**: Faster, good for quick analysis
- **Qwen2.5-VL**: Slower but higher quality, better for detailed analysis

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure transformers is installed from source
2. **Memory Error**: Reduce batch size or use model offloading
3. **Model Loading**: Check internet connection and HF token

### Error Messages

- `KeyError: 'qwen2_5_vl'`: Install transformers from source
- `CUDA out of memory`: Reduce model precision or batch size
- `Model not found`: Verify model path and HF token

## Future Enhancements

- **Model Quantization**: 4-bit and 8-bit support
- **Batch Processing**: Multiple video analysis
- **Streaming**: Real-time video analysis
- **Custom Training**: Fine-tuning for specific domains

## References

- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Official Documentation](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Qwen-VL-Utils](https://github.com/QwenLM/qwen-vl-utils)

## Support

For issues related to:

- **Model Integration**: Check this document and test scripts
- **Performance**: Review GPU configuration and memory settings
- **Dependencies**: Ensure all packages are up to date
- **API Usage**: Refer to the main application documentation 