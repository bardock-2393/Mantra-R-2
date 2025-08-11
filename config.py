import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration for Round 2 - OPTIMIZED FOR 7B MODEL ONLY"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB for longer videos
    
    # File Upload Configuration
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'm4v'}
    
    # GPU Configuration - OPTIMIZED FOR 7B MODEL
    GPU_CONFIG = {
        'enabled': True,
        'device': 'cuda:0',  # Primary GPU
        'memory_limit': 80 * 1024 * 1024 * 1024,  # 80GB
        'batch_size': 32,
        'precision': 'float16',  # Use FP16 for MAXIMUM speed
        'num_workers': 4
    }
    
    # Qwen2.5-VL-7B Model Configuration - OPTIMIZED FOR MAXIMUM PERFORMANCE
    QWEN25VL_MODEL_PATH = os.getenv('QWEN25VL_MODEL_PATH', 'Qwen/Qwen2.5-VL-7B-Instruct')
    QWEN25VL_CONFIG = {
        'model_name': 'Qwen/Qwen2.5-VL-7B-Instruct (OPTIMIZED)',
        'hf_token': os.getenv('HF_TOKEN', ''),
        'max_length': 8192,  # Optimized for 7B model speed
        'temperature': 0.1,  # Lower temperature for accuracy
        'top_p': 0.95,      # Optimized top-p for quality
        'top_k': 50,        # Enhanced top-k for diversity
        'chat_temperature': 0.1,  # Consistent temperature for chat
        'chat_max_length': 4096,  # Shorter for real-time chat
        # PERFORMANCE OPTIMIZATIONS
        'min_pixels': 256 * 28 * 28,   # 256 tokens for efficiency
        'max_pixels': 1024 * 28 * 28,  # 1024 tokens optimized for 7B
        'precision': 'float16',         # Force FP16 for speed
        'use_flash_attention': True,    # Enable Flash Attention 2
        'use_xformers': True,           # Enable xformers optimization
    }
    
    # DeepStream Configuration - OPTIMIZED FOR 80GB GPU
    DEEPSTREAM_CONFIG = {
        'enabled': True,
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # 120 minutes
        'yolo_model': 'models/yolov8n.engine',  # TensorRT optimized
        'tracking': 'nvdcf',  # NVIDIA DeepStream tracker
        'gpu_memory_limit': 80 * 1024 * 1024 * 1024,  # 80GB total GPU memory
        'deepstream_memory': 60 * 1024 * 1024 * 1024,  # 60GB for DeepStream
        'batch_size': 32,
        'precision': 'fp16'
    }
    
    # Streaming Configuration - NEW FOR REAL-TIME ANALYSIS
    STREAMING_CONFIG = {
        'enabled': True,
        'fps_target': 30,  # Real-time processing target
        'max_latency_ms': 100,  # Maximum latency for real-time
        'event_detection_enabled': True,
        'continuous_processing': True,
        'event_thresholds': {
            'motion': 0.3,
            'object': 0.5,
            'scene_change': 0.4,
            'anomaly': 0.6
        },
        'frame_buffer_size': 10,  # Number of frames to buffer
        'analysis_interval': 1,   # Analyze every Nth frame
        'real_time_7b_analysis': True  # Enable 7B model on stream frames
    }
    
    # Performance Targets
    PERFORMANCE_TARGETS = {
        'latency_target': 1000,  # ms
        'fps_target': 90,
        'max_video_duration': 120 * 60,  # seconds
        'concurrent_sessions': 10,
        'streaming_latency_target': 100,  # ms for real-time
        'streaming_fps_target': 30        # fps for real-time
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
    
    # Vector Search Configuration - OPTIMIZED FOR HYBRID SYSTEM
    VECTOR_CONFIG = {
        'embedding_dim': 768,  # Higher dimension for better accuracy
        'index_type': 'faiss.IndexFlatL2',
        'search_algorithm': 'approximate_nearest_neighbor',
        'max_results': 100,
        'similarity_threshold': 0.8,
        'chunk_size': 200,
        'overlap': 50
    }

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
    "local_ai": True,              # New for Round 2
    "streaming_analysis": True,    # NEW: Real-time streaming
    "live_event_detection": True   # NEW: Live event detection
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
    },
    "streaming_analysis": {
        "description": "Real-time video streaming with live 7B model analysis",
        "capabilities": ["live_processing", "event_detection", "real_time_insights", "continuous_analysis"]
    }
} 