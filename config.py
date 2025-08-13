import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration for Round 2 - GPU-powered local AI"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB for longer videos
    
    # File Upload Configuration
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'm4v'}
    
    # GPU Configuration - Optimized for 16GB+ + 7B model + video processing
    GPU_CONFIG = {
        'enabled': True,
        'device': 'cuda:0',  # Primary GPU
        'memory_limit': 16 * 1024 * 1024 * 1024,  # 16GB (7B model requirement)
        'batch_size': 2,  # 7B model can handle 2 batches
        'precision': 'bfloat16',  # Better for 7B models than float16
        'num_workers': 4,  # Increased for 7B model efficiency
        'gradient_checkpointing': False,  # Disable for inference
        'use_flash_attention': True,  # Enable for 7B model
        'compile_mode': 'reduce-overhead'  # Speed optimization
    }
    
    # MiniCPM-V Model Configuration
    MINICPM_MODEL_PATH = os.getenv('MINICPM_MODEL_PATH', 'openbmb/MiniCPM-V-2_6')
    MINICPM_CONFIG = {
        'model_name': 'openbmb/MiniCPM-V-2_6',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 32768,
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 40
    }
    
    # Qwen2.5-VL Model Configuration
    QWEN25VL_MODEL_PATH = os.getenv('QWEN25VL_MODEL_PATH', 'Qwen/Qwen2.5-VL-7B-Instruct')
    QWEN25VL_CONFIG = {
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 32768,
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 40,
        'chat_temperature': 0.3
    }
    
    # Qwen2.5-VL-32B Model Configuration - ULTRA-FAST for short videos
    QWEN25VL_32B_MODEL_PATH = os.getenv('QWEN25VL_32B_MODEL_PATH', 'Qwen/Qwen2.5-VL-32B-Instruct')
    QWEN25VL_32B_CONFIG = {
        'model_name': 'Qwen/Qwen2.5-VL-32B-Instruct',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 1024,  # ULTRA-FAST: Reduced from 4096 to 1024 for speed
        'temperature': 0.0,   # ULTRA-FAST: Deterministic (greedy) generation
        'top_p': 1.0,        # ULTRA-FAST: No nucleus sampling
        'top_k': 1,          # ULTRA-FAST: Greedy selection only
        'chat_temperature': 0.0,  # ULTRA-FAST: Deterministic chat
        'min_pixels': 256 * 28 * 28,  # 256 tokens
        'max_pixels': 640 * 28 * 28,  # ULTRA-FAST: Reduced from 1280 to 640
        # ULTRA-FAST speed optimization settings
        'use_cache': True,
        'do_sample': False,   # ULTRA-FAST: Disable sampling
        'num_beams': 1,       # ULTRA-FAST: Single beam
        'early_stopping': True,
        'repetition_penalty': 1.0,  # ULTRA-FAST: No repetition penalty
        'length_penalty': 1.0,      # ULTRA-FAST: No length penalty
        # Memory optimization for 80GB GPU
        'batch_size': 1,      # Single batch for memory efficiency
        'gradient_checkpointing': False,  # Disable for inference
        'use_flash_attention': False,  # Disable for compatibility
        'compile_mode': 'max-autotune',  # Enhanced speed optimization
        # Timeout settings to prevent Cloudflare 524
        'vision_timeout': 1800,  # 30 minutes for vision processing
        'generation_timeout': 1800,  # 30 minutes for text generation
        'total_timeout': 3600,  # 60 minutes total analysis time
        # ULTRA-FAST frame optimization for short videos
        'max_frames_large': 2,   # ULTRA-FAST: Large videos (>500MB) - only 2 frames
        'max_frames_medium': 3,  # ULTRA-FAST: Medium videos (100-500MB) - only 3 frames
        'max_frames_small': 4,   # ULTRA-FAST: Small videos (<100MB) - only 4 frames
        'frame_resolution': 224,  # Resize frames to 224x224 for memory efficiency
        'use_half_precision': True,  # Use FP16 for memory efficiency
        # ULTRA-FAST generation settings
        'max_new_tokens': 512,   # ULTRA-FAST: Limit output to 512 tokens max
        'min_new_tokens': 100,   # ULTRA-FAST: Minimum 100 tokens for quality
        'no_repeat_ngram_size': 0,  # ULTRA-FAST: Disable n-gram blocking
        'bad_words_ids': None,   # ULTRA-FAST: No bad word filtering
    }
    
    # DeepStream Configuration - Optimized for 120min 720p 90fps
    DEEPSTREAM_CONFIG = {
        'enabled': True,
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # 120 minutes
        'yolo_model': 'yolov8n.engine',  # TensorRT optimized
        'tracking': 'nvdcf',  # NVIDIA DeepStream tracker
        'gpu_memory': 8 * 1024 * 1024 * 1024,  # 8GB for DeepStream (increased)
        'chunk_size': 30,  # Process 30-second chunks
        'overlap': 5,  # 5-second overlap between chunks
        'memory_efficient': True  # Enable memory optimization
    }
    
    # Video Chunking Configuration - For 120+ minute videos
    VIDEO_CHUNKING_CONFIG = {
        'enabled': True,
        'chunk_duration': 300,  # 5 minutes per chunk
        'max_workers': 2,  # Parallel processing workers (GPU memory constrained)
        'frame_rate': 1,  # 1 fps for analysis (reduced from 8)
        'resolution': (720, 480),  # 720p resolution for speed
        'use_decord': True,  # Use decord instead of OpenCV
        'memory_cleanup': True,  # Clean GPU memory between chunks
        'overlap_frames': 2,  # Overlap frames between chunks for continuity
        'min_chunk_size': 60,  # Minimum chunk size in seconds
        'max_chunk_size': 600,  # Maximum chunk size in seconds (10 minutes)
    }
    
    # WebSocket Configuration - For real-time progress updates
    WEBSOCKET_CONFIG = {
        'enabled': True,
        'ping_timeout': 60,
        'ping_interval': 25,
        'max_http_buffer_size': 1e8,
        'cors_allowed_origins': '*',
        'async_mode': 'threading'
    }
    
    # Performance Optimization Configuration
    PERFORMANCE_OPTIMIZATION = {
        'torch_compile_mode': 'max-autotune',  # Enable torch.compile optimization
        'quantization_enabled': True,  # Enable model quantization (INT8/FP16)
        'memory_cleanup_threshold': 0.8,  # 80% GPU memory threshold
        'aggressive_cleanup': False,  # Enable aggressive memory cleanup
        'frame_processing_optimization': True,  # Optimize frame processing
        'parallel_chunk_processing': True,  # Process chunks in parallel
    }
    
    # Performance Targets - Adjusted for 80GB constraint
    PERFORMANCE_TARGETS = {
        'latency_target': 2000,  # Increased to 2s for 32B model
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # 120 minutes
        'concurrent_sessions': 2,  # Reduced for 32B model
        'memory_buffer': 10 * 1024 * 1024 * 1024  # 10GB buffer
    }
    
    # Redis Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://default:nswO0Z95wT9aeXIIOZMMphnDhsPY3slG@redis-10404.c232.us-east-1-2.ec2.redns.redis-cloud.com:10404')
    
    # Session Configuration
    SESSION_EXPIRY = 3600  # 1 hour in seconds
    UPLOAD_CLEANUP_TIME = 2 * 3600  # 2 hours in seconds
    SESSION_STORAGE_PATH = 'sessions'  # Local file storage (fallback)
    
    # Analysis Configuration
    MAX_OUTPUT_TOKENS = 32768
    CHAT_MAX_TOKENS = 8192
    TEMPERATURE = 0.2
    CHAT_TEMPERATURE = 0.3
    TOP_P = 0.9
    TOP_K = 40
    
    # Default Video Configuration
    DEFAULT_VIDEO_PATH = 'BMW M4 - Ultimate Racetrack - BMW Canada (720p, h264).mp4'

# Enhanced Agent Capabilities for Round 2
AGENT_CAPABILITIES = {
    "autonomous_analysis": True,
    "multi_modal_understanding": True,
    "context_aware_responses": True,
    "proactive_insights": True,
    "comprehensive_reporting": True,
    "adaptive_focus": True,
    "real_time_processing": True,  # New for Round 2
    "gpu_optimized": True,         # New for Round 2
    "local_ai": True               # New for Round 2
}

# Agent Tools and Capabilities for Round 2
AGENT_TOOLS = {
    "video_analysis": {
        "description": "GPU-accelerated video content analysis with <1000ms latency",
        "capabilities": ["visual_analysis", "audio_analysis", "temporal_analysis", "spatial_analysis", "real_time_processing"]
    },
    "context_awareness": {
        "description": "Advanced context understanding and adaptive responses",
        "capabilities": ["session_memory", "conversation_history", "user_preferences", "analysis_context"]
    },
    "autonomous_workflow": {
        "description": "Self-directed analysis and proactive insights generation",
        "capabilities": ["autonomous_analysis", "proactive_insights", "adaptive_focus", "comprehensive_reporting"]
    },
    "gpu_processing": {
        "description": "High-performance GPU-optimized video processing pipeline",
        "capabilities": ["90fps_processing", "120min_video_support", "real_time_analysis", "gpu_optimization"]
    }
} 