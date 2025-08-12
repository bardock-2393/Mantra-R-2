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
    
    # GPU Configuration - Optimized for 80GB + 32B model + video processing
    GPU_CONFIG = {
        'enabled': True,
        'device': 'cuda:0',  # Primary GPU
        'memory_limit': 80 * 1024 * 1024 * 1024,  # 80GB
        'batch_size': 1,  # Reduced for 32B model + video processing
        'precision': 'bfloat16',  # Better for 32B models than float16
        'num_workers': 2,  # Reduced for memory efficiency
        'gradient_checkpointing': False,  # Disable for inference
        'use_flash_attention': False,  # Disable for compatibility
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
    
    # Qwen2.5-VL-32B Model Configuration - Optimized for speed
    QWEN25VL_32B_MODEL_PATH = os.getenv('QWEN25VL_32B_MODEL_PATH', 'Qwen/Qwen2.5-VL-32B-Instruct')
    QWEN25VL_32B_CONFIG = {
        'model_name': 'Qwen/Qwen2.5-VL-32B-Instruct',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 8192,  # Reduced for faster generation
        'temperature': 0.1,   # More deterministic = faster
        'top_p': 0.8,        # Reduced for faster sampling
        'top_k': 20,          # Reduced for faster sampling
        'chat_temperature': 0.2,
        'min_pixels': 256 * 28 * 28,  # 256 tokens
        'max_pixels': 1280 * 28 * 28,  # 1280 tokens
        # Speed optimization settings
        'use_cache': True,
        'do_sample': False,   # Disable sampling for speed
        'num_beams': 1,       # Single beam for speed
        'early_stopping': True
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