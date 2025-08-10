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
    
    # GPU Configuration
    GPU_CONFIG = {
        'enabled': True,
        'device': 'cuda:0',  # Primary GPU
        'memory_limit': 80 * 1024 * 1024 * 1024,  # 80GB
        'batch_size': 32,
        'precision': 'float16',  # Use FP16 for speed
        'num_workers': 4
    }
    
    # MiniCPM-V Model Configuration
    MINICPM_MODEL_PATH = os.getenv('MINICPM_MODEL_PATH', 'openbmb/MiniCPM-V')
    MINICPM_CONFIG = {
        'model_name': 'openbmb/MiniCPM-V',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 32768,
        'temperature': 0.2,
        'top_p': 0.9,
        'top_k': 40,
        'use_flash_attention': True,
        'quantization': 'int8'  # Use INT8 for speed
    }
    
    # DeepStream Configuration
    DEEPSTREAM_CONFIG = {
        'enabled': True,
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # 120 minutes
        'yolo_model': 'yolov8n.engine',  # TensorRT optimized
        'tracking': 'nvdcf',  # NVIDIA DeepStream tracker
        'gpu_memory': 4 * 1024 * 1024 * 1024  # 4GB for DeepStream
    }
    
    # Performance Targets
    PERFORMANCE_TARGETS = {
        'latency_target': 1000,  # ms
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # seconds
        'concurrent_sessions': 10
    }
    
    # Session Configuration (Local storage)
    SESSION_EXPIRY = 3600  # 1 hour in seconds
    UPLOAD_CLEANUP_TIME = 2 * 3600  # 2 hours in seconds
    SESSION_STORAGE_PATH = 'sessions'  # Local file storage
    
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