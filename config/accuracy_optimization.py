"""
Accuracy Optimization Configuration for Qwen2.5-VL-32B Model
Optimized settings for maximum video analysis accuracy
"""

# Model Configuration for Maximum Accuracy
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
    "device_map": "auto",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "use_cache": False,  # Disable cache for better accuracy
    "low_cpu_mem_usage": True,
    "max_memory": {0: "80GB"},  # Optimize for 80GB GPU
}

# Generation Configuration for High Accuracy
GENERATION_CONFIG = {
    "max_new_tokens": 4096,        # Increased for more detailed analysis
    "do_sample": True,
    "temperature": 0.3,            # Lower temperature for more focused/accurate output
    "top_p": 0.85,                # Nucleus sampling for quality
    "top_k": 40,                  # Top-k sampling
    "repetition_penalty": 1.15,   # Prevent repetition
    "length_penalty": 1.2,        # Encourage detailed responses
    "early_stopping": True,
    "pad_token_id": None,
    "eos_token_id": None,
    "use_cache": False,
    "num_beams": 1,               # Single beam for faster processing
    "no_repeat_ngram_size": 3,    # Prevent repetitive phrases
}

# Vision Processing Configuration
VISION_CONFIG = {
    "frame_extraction": {
        "method": "adaptive",      # Adaptive frame selection
        "min_frames": 30,         # Minimum frames to analyze
        "max_frames": 150,        # Maximum frames to analyze
        "quality_threshold": 0.8, # Minimum frame quality
        "extraction_rate": 2,     # Extract every Nth frame
    },
    "image_processing": {
        "resize_method": "bicubic",
        "target_size": (1024, 1024),  # Optimal size for 32B model
        "normalization": "imagenet",
        "augmentation": False,     # Disable augmentation for accuracy
    },
    "multi_scale_analysis": {
        "enabled": True,
        "scales": [0.5, 1.0, 1.5],  # Multiple scales for better detection
        "weight_method": "confidence_weighted",
    }
}

# Analysis Prompt Templates for Maximum Accuracy
ANALYSIS_PROMPTS = {
    "ultra_accurate": """Analyze this video with ULTRA-HIGH ACCURACY. Your goal is to be 100% precise in every observation.

CRITICAL REQUIREMENTS:
- Be absolutely certain of every detail you mention
- If you're unsure about ANYTHING, say "I'm uncertain about [specific detail]"
- Provide exact, measurable descriptions
- Avoid vague terms like "some", "maybe", "appears to be"
- Use specific colors, sizes, positions, and timings

ANALYSIS STRUCTURE:

1. VISUAL ELEMENTS (Be Extremely Precise):
   - Exact object identification (no guessing)
   - Precise color descriptions (RGB values if possible)
   - Exact spatial positioning (left, right, center, etc.)
   - Specific sizes and proportions
   - Exact motion patterns and speeds

2. TEMPORAL ANALYSIS (Be Precise):
   - Exact timing of events (seconds, frames)
   - Duration of specific actions
   - Sequence of movements
   - Changes over time with specific details

3. SPATIAL RELATIONSHIPS (Be Exact):
   - Relative positions of objects
   - Distances between elements
   - Spatial layout and arrangement
   - Background vs foreground positioning

4. QUALITY ASSESSMENT:
   - Video resolution and clarity
   - Lighting conditions (bright, dim, shadows)
   - Camera angles and perspectives
   - Any technical limitations affecting analysis

5. CONFIDENCE LEVELS:
   - Mark each observation with confidence level
   - High confidence: "I can clearly see..."
   - Medium confidence: "I believe I can see..."
   - Low confidence: "I think I might see..."

Remember: ACCURACY OVER COMPLETENESS. It's better to be certain about fewer details than uncertain about many.""",

    "comprehensive_accurate": """Provide a COMPREHENSIVE and ACCURATE analysis of this video. Focus on being thorough while maintaining high precision.

ANALYSIS APPROACH:

1. SYSTEMATIC SCANNING:
   - Start from top-left, move systematically
   - Cover every quadrant of the video
   - Don't skip any visible area
   - Note even minor details

2. DETAILED OBSERVATIONS:
   - Object identification with confidence levels
   - Color analysis with specific descriptions
   - Size relationships and proportions
   - Motion analysis with timing
   - Environmental factors and context

3. ACCURACY CHECKS:
   - Cross-reference observations across frames
   - Verify consistency of details
   - Note any discrepancies or changes
   - Acknowledge limitations clearly

4. COMPREHENSIVE COVERAGE:
   - Main subjects and focal points
   - Background elements and context
   - Temporal progression and changes
   - Spatial relationships and layout
   - Technical aspects and quality

Provide detailed analysis with specific, measurable observations.""",

    "focused_precision": """Analyze this video with FOCUSED PRECISION. Concentrate on the most important elements while maintaining extremely high accuracy.

FOCUS AREAS:

1. PRIMARY SUBJECTS:
   - Main objects, people, or actions
   - Exact identification and characteristics
   - Precise positioning and movement
   - Specific attributes and features

2. KEY EVENTS:
   - Important moments and actions
   - Exact timing and duration
   - Sequence and progression
   - Significance and context

3. CRITICAL DETAILS:
   - Essential visual information
   - Important spatial relationships
   - Key temporal patterns
   - Relevant environmental factors

4. ACCURACY VERIFICATION:
   - Double-check all observations
   - Verify consistency across frames
   - Confirm spatial relationships
   - Validate temporal sequences

Focus on being 100% accurate in your analysis of the key elements."""
}

# Quality Enhancement Configuration
QUALITY_ENHANCEMENT = {
    "confidence_scoring": {
        "enabled": True,
        "thresholds": {
            "high": 0.9,      # 90%+ confidence
            "medium": 0.7,    # 70-89% confidence
            "low": 0.5,       # 50-69% confidence
            "uncertain": 0.0  # Below 50% confidence
        }
    },
    "error_correction": {
        "enabled": True,
        "methods": [
            "cross_frame_validation",
            "spatial_consistency_check",
            "temporal_sequence_verification",
            "object_relationship_validation"
        ]
    },
    "fallback_strategies": {
        "vision_failure": "enhanced_text_analysis",
        "model_error": "simplified_analysis",
        "memory_issue": "chunked_analysis",
        "timeout": "fast_analysis",
        "quality_issue": "multi_scale_analysis"
    }
}

# Performance Optimization for Accuracy
PERFORMANCE_CONFIG = {
    "gpu_optimization": {
        "mixed_precision": True,
        "gradient_checkpointing": False,  # Disable for inference
        "attention_implementation": "flash_attention_2",
        "memory_efficient_attention": True,
        "compile_model": True,           # Enable torch.compile
    },
    "batch_processing": {
        "batch_size": 1,                # Single batch for accuracy
        "max_workers": 2,               # Limited workers for stability
        "prefetch_factor": 2,
    },
    "caching": {
        "enable_cache": False,           # Disable for fresh analysis
        "clear_cache_interval": 1,      # Clear after each analysis
    }
}

# Error Handling and Recovery
ERROR_HANDLING = {
    "retry_attempts": 3,
    "backoff_strategy": "exponential",
    "fallback_graceful": True,
    "log_errors": True,
    "user_friendly_errors": True,
    "recovery_methods": [
        "restart_model_service",
        "clear_gpu_memory",
        "reduce_batch_size",
        "switch_to_cpu_fallback"
    ]
}

# Validation and Quality Control
VALIDATION_CONFIG = {
    "output_validation": {
        "check_consistency": True,
        "validate_timestamps": True,
        "verify_spatial_relationships": True,
        "cross_reference_observations": True
    },
    "quality_metrics": {
        "min_analysis_length": 1000,    # Minimum characters
        "max_analysis_length": 10000,   # Maximum characters
        "required_sections": [
            "visual_elements",
            "temporal_analysis",
            "spatial_relationships",
            "confidence_levels"
        ]
    }
}

# Model-Specific Optimizations for Qwen2.5-VL-32B
QWEN_SPECIFIC_CONFIG = {
    "model_loading": {
        "load_in_8bit": False,          # Disable for accuracy
        "load_in_4bit": False,          # Disable for accuracy
        "use_safetensors": True,
        "trust_remote_code": True,
    },
    "inference_settings": {
        "use_flash_attention": True,
        "use_xformers": False,          # Disable for stability
        "use_deepspeed": False,         # Disable for simplicity
        "use_accelerate": True,
    },
    "memory_management": {
        "max_memory": {0: "80GB"},
        "offload_folder": "offload",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
}

# Export all configurations
ACCURACY_CONFIG = {
    "model": MODEL_CONFIG,
    "generation": GENERATION_CONFIG,
    "vision": VISION_CONFIG,
    "prompts": ANALYSIS_PROMPTS,
    "quality": QUALITY_ENHANCEMENT,
    "performance": PERFORMANCE_CONFIG,
    "error_handling": ERROR_HANDLING,
    "validation": VALIDATION_CONFIG,
    "qwen_specific": QWEN_SPECIFIC_CONFIG
}
