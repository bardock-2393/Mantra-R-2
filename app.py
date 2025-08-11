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
    
    # Global error handlers to ensure JSON responses
    @app.errorhandler(400)
    def bad_request(error):
        return {'success': False, 'error': 'Bad Request', 'details': str(error)}, 400, {'Content-Type': 'application/json'}
    
    @app.errorhandler(404)
    def not_found(error):
        return {'success': False, 'error': 'Not Found', 'details': str(error)}, 404, {'Content-Type': 'application/json'}
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'success': False, 'error': 'Internal Server Error', 'details': str(error)}, 500, {'Content-Type': 'application/json'}
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        # Log the error
        import traceback
        print(f"❌ Unhandled exception: {error}")
        traceback.print_exc()
        
        return {
            'success': False, 
            'error': 'Internal Server Error', 
            'details': str(error),
            'message': 'An unexpected error occurred'
        }, 500, {'Content-Type': 'application/json'}
    
    return app

async def initialize_gpu_services():
    """Initialize GPU services for Round 2"""
    try:
        print("🚀 Initializing GPU services...")
        
        # Initialize GPU service
        gpu_service = GPUService()
        await gpu_service.initialize()
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        performance_monitor.start()
        
        print("✅ GPU services initialized successfully")
        return gpu_service, performance_monitor
        
    except Exception as e:
        print(f"❌ Failed to initialize GPU services: {e}")
        return None, None

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
async def main():
    """Main async function to initialize and run the application"""
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.SESSION_STORAGE_PATH, exist_ok=True)
    
    # Clean up expired sessions on startup
    print("🧹 Cleaning up expired sessions...")
    cleanup_expired_sessions()
    
    # Initialize GPU services for Round 2
    gpu_service, performance_monitor = await initialize_gpu_services()
    
    # Start background cleanup thread
    cleanup_thread = start_cleanup_thread()
    
    print("🚀 AI Video Detective Round 2 Starting...")
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"📁 Session storage: {Config.SESSION_STORAGE_PATH}")
    print(f"🤖 GPU Processing: {'Enabled' if Config.GPU_CONFIG['enabled'] else 'Disabled'}")
    print(f"🎯 Performance targets: <{Config.PERFORMANCE_TARGETS['latency_target']}ms latency, {Config.PERFORMANCE_TARGETS['fps_target']}fps")
    
    if gpu_service:
        print(f"🖥️  GPU Device: {Config.GPU_CONFIG['device']}")
        print(f"💾 GPU Memory: {Config.GPU_CONFIG['memory_limit'] // (1024**3)}GB")
    
    return app

if __name__ == '__main__':
    import asyncio
    app_instance = asyncio.run(main())
    app_instance.run(host='0.0.0.0', port=8000, debug=True) 