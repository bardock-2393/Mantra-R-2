"""
AI Video Detective - Main Application for Round 2
Advanced AI video analysis application with GPU-powered local AI processing
"""

import os
import threading
import time
from flask import Flask
from config import Config
from routes.main_routes import main_bp
from routes.chat_routes import chat_bp
from routes.api_routes import api_bp
from services.session_service import cleanup_expired_sessions, cleanup_old_uploads
from services.gpu_service import GPUService
from services.performance_service import PerformanceMonitor

def create_app():
    """Create and configure the Flask application for Round 2"""
    app = Flask(__name__)
    
    # Configure Flask
    app.secret_key = Config.SECRET_KEY
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

def start_cleanup_thread():
    """Start background cleanup thread"""
    def periodic_cleanup():
        while True:
            time.sleep(1800)  # 30 minutes
            print("🧹 Running periodic cleanup...")
            cleanup_expired_sessions()
            cleanup_old_uploads()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    return cleanup_thread

# Create the application instance
app = create_app()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.SESSION_STORAGE_PATH, exist_ok=True)
    
    # Clean up expired sessions on startup
    print("🧹 Cleaning up expired sessions...")
    cleanup_expired_sessions()
    
    # Start background cleanup thread
    cleanup_thread = start_cleanup_thread()
    
    print("🚀 AI Video Detective Round 2 Starting...")
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"📁 Session storage: {Config.SESSION_STORAGE_PATH}")
    print(f"🤖 GPU Processing: {'Enabled' if Config.GPU_CONFIG['enabled'] else 'Disabled'}")
    print(f"🎯 Performance targets: <{Config.PERFORMANCE_TARGETS['latency_target']}ms latency, {Config.PERFORMANCE_TARGETS['fps_target']}fps")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=8000, debug=True) 