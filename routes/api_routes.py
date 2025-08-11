"""
API Routes Module
Handles various API endpoints for session management, health checks, and utility functions
"""

import os
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from config import Config, AGENT_CAPABILITIES, AGENT_TOOLS
from services.session_service import (
    get_session_data, cleanup_session_data, cleanup_expired_sessions, 
    cleanup_old_uploads, get_all_session_keys
)
from utils.video_utils import capture_screenshot, extract_video_clip
from utils.text_utils import extract_timestamps_from_text
import time

# Create Blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/session/<session_id>')
def get_session(session_id):
    """Get session data"""
    try:
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        session_data = get_session_data(session_id)
        if not session_data:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'session_data': session_data
        })
        
    except Exception as e:
        print(f"❌ Get session error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get session: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'success': True,
            'status': 'healthy',
            'service': 'AI Video Detective API',
            'version': '2.0.0',
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        }), 500

@api_bp.route('/model-health')
def model_health_check():
    """Health check endpoint for AI models"""
    try:
        from services.model_manager import model_manager
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async health check
        health_status = loop.run_until_complete(model_manager.health_check())
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'service': 'AI Model Health Check',
            'timestamp': time.time(),
            'models': health_status
        })
        
    except Exception as e:
        print(f"❌ Model health check error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'status': 'error',
            'service': 'AI Model Health Check',
            'error': str(e),
            'timestamp': time.time(),
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/switch-model', methods=['POST'])
def switch_model():
    """Switch between different AI models"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON data received'}), 400
        
        model_name = data.get('model')
        if not model_name:
            return jsonify({'success': False, 'error': 'Model name required'}), 400
        
        # Import model manager
        from services.model_manager import model_manager
        
        # Switch model using the model manager (run in event loop)
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            success = loop.run_until_complete(model_manager.switch_model(model_name))
            
            if success:
                # Get updated model status
                model_status = model_manager.get_current_model()
                
                return jsonify({
                    'success': True,
                    'message': f'Successfully switched to {model_name}',
                    'current_model': model_status
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to switch to {model_name}',
                    'details': 'Model initialization failed'
                }), 500
                
        except Exception as e:
            print(f"❌ Model switch error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Model switch failed: {str(e)}',
                'details': 'Check server logs for more information'
            }), 500
            
    except Exception as e:
        print(f"❌ Switch model endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Switch model failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/model-status')
def get_model_status():
    """Get current model status"""
    try:
        from services.model_manager import model_manager
        
        model_status = model_manager.get_status()
        
        return jsonify({
            'success': True,
            'model_status': model_status
        })
        
    except Exception as e:
        print(f"❌ Model status error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get model status: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/agent-info')
def get_agent_info():
    """Get agent capabilities and tools information"""
    try:
        return jsonify({
            'success': True,
            'capabilities': AGENT_CAPABILITIES,
            'tools': AGENT_TOOLS,
            'service': 'AI Video Detective Agent',
            'version': '2.0.0'
        })
        
    except Exception as e:
        print(f"❌ Agent info error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get agent info: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/capture-screenshots', methods=['POST'])
def capture_screenshots():
    """Capture screenshots at specific timestamps from the analyzed video"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        timestamps = data.get('timestamps', [])
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'})
        
        # Get session data
        session_data = get_session_data(session_id)
        if not session_data or 'filepath' not in session_data:
            return jsonify({'success': False, 'error': 'No video found for session'})
        
        video_path = session_data['filepath']
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'})
        
        # Capture screenshots for each timestamp
        screenshots = []
        for timestamp in timestamps:
            screenshot_data = capture_screenshot(video_path, timestamp, session_id, Config.UPLOAD_FOLDER)
            if screenshot_data:
                screenshots.append(screenshot_data)
        
        return jsonify({
            'success': True,
            'screenshots': screenshots,
            'count': len(screenshots)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/auto-capture-screenshots', methods=['POST'])
def auto_capture_screenshots():
    """Automatically capture screenshots based on timestamps found in analysis text"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        analysis_text = data.get('analysis_text', '')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'})
        
        # Extract timestamps from analysis text
        timestamps = extract_timestamps_from_text(analysis_text)
        
        if not timestamps:
            return jsonify({'success': False, 'error': 'No timestamps found in analysis'})
        
        # Get session data
        session_data = get_session_data(session_id)
        if not session_data or 'filepath' not in session_data:
            return jsonify({'success': False, 'error': 'No video found for session'})
        
        video_path = session_data['filepath']
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'})
        
        # Capture screenshots for extracted timestamps
        screenshots = []
        for timestamp in timestamps:
            screenshot_data = capture_screenshot(video_path, timestamp, session_id, Config.UPLOAD_FOLDER)
            if screenshot_data:
                screenshots.append(screenshot_data)
        
        return jsonify({
            'success': True,
            'screenshots': screenshots,
            'timestamps': timestamps,
            'count': len(screenshots)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/session/cleanup', methods=['POST'])
def cleanup_current_session():
    """Clean up current session data"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'No active session to cleanup'
            }), 400
        
        # Clean up session data
        if cleanup_session_data(session_id):
            # Clear session
            session.pop('session_id', None)
            
            return jsonify({
                'success': True,
                'message': 'Session cleaned up successfully',
                'session_id': session_id
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to cleanup session'
            }), 500
            
    except Exception as e:
        print(f"❌ Session cleanup error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Session cleanup failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/session/status')
def get_session_status():
    """Get current session status and file information"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'active': False,
                'session_id': None,
                'message': 'No active session'
            })
        
        # Get session data
        session_data = get_session_data(session_id)
        
        if not session_data:
            return jsonify({
                'success': False,
                'active': False,
                'session_id': session_id,
                'message': 'Session data not found'
            })
        
        # Check if video file exists
        video_exists = False
        video_size = 0
        if 'filepath' in session_data:
            video_path = session_data['filepath']
            if os.path.exists(video_path):
                video_exists = True
                video_size = os.path.getsize(video_path)
        
        # Count evidence files
        evidence_count = 0
        upload_folder = Config.UPLOAD_FOLDER
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                if filename.startswith(f"screenshot_{session_id}_") or filename.startswith(f"clip_{session_id}_"):
                    evidence_count += 1
        
        return jsonify({
            'success': True,
            'active': True,
            'session_id': session_id,
            'video_uploaded': video_exists,
            'video_size': video_size,
            'evidence_count': evidence_count,
            'analysis_complete': 'analysis_result' in session_data,
            'upload_time': session_data.get('upload_time'),
            'analysis_time': session_data.get('analysis_time')
        })
        
    except Exception as e:
        print(f"❌ Session status error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Session status failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/session/cleanup-all', methods=['POST'])
def cleanup_all_sessions():
    """Clean up all sessions (admin function)"""
    try:
        # Get all session keys from local storage
        session_keys = get_all_session_keys()
        cleaned_sessions = 0
        
        for session_id in session_keys:
            if cleanup_session_data(session_id):
                cleaned_sessions += 1
        
        # Also clean up old uploads
        cleanup_old_uploads()
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {cleaned_sessions} sessions and old uploads',
            'cleaned_count': cleaned_sessions
        })
        
    except Exception as e:
        print(f"❌ Cleanup all sessions error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Cleanup all sessions failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@api_bp.route('/cleanup-uploads', methods=['POST'])
def cleanup_uploads():
    """Clean up old upload files"""
    try:
        cleanup_old_uploads()
        return jsonify({
            'success': True,
            'message': 'Uploads cleanup completed'
        })
        
    except Exception as e:
        print(f"❌ Cleanup uploads error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Uploads cleanup failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500 