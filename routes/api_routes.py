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
        session_data = get_session_data(session_id)
        return jsonify(session_data)
    except Exception as e:
        return jsonify({'error': f'Failed to get session: {str(e)}'}), 500

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'AI Video Detective API'
    })

@api_bp.route('/session/status', methods=['GET'])
def get_session_status():
    """Get current session status"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'No active session',
                'has_session': False
            })
        
        # Get session data
        session_data = get_session_data(session_id)
        
        if session_data:
            return jsonify({
                'success': True,
                'has_session': True,
                'session_id': session_id,
                'analysis_available': 'analysis_result' in session_data,
                'chat_history_count': len(session_data.get('chat_history', [])),
                'evidence_count': len(session_data.get('evidence', [])),
                'timestamps_count': len(session_data.get('timestamps_found', []))
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found',
                'has_session': False
            })
            
    except Exception as e:
        print(f"Session status error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/session/cleanup', methods=['POST'])
def cleanup_current_session():
    """Clean up current session data"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'No active session'})
        
        # Clean up session data
        success = cleanup_session_data(session_id)
        
        if success:
            # Clear session
            session.clear()
            return jsonify({
                'success': True,
                'message': 'Session cleaned up successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to cleanup session'
            })
            
    except Exception as e:
        print(f"Session cleanup error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/session/cleanup-all', methods=['POST'])
def cleanup_all_sessions():
    """Clean up all expired sessions"""
    try:
        # Get all session keys
        session_keys = get_all_session_keys()
        
        cleaned_count = 0
        for session_id in session_keys:
            if cleanup_session_data(session_id):
                cleaned_count += 1
        
        # Also clean up old uploads
        cleanup_old_uploads()
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {cleaned_count} sessions',
            'sessions_cleaned': cleaned_count
        })
        
    except Exception as e:
        print(f"Cleanup all sessions error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/cleanup-uploads', methods=['POST'])
def cleanup_uploads():
    """Clean up old uploaded files"""
    try:
        cleanup_old_uploads()
        return jsonify({
            'success': True,
            'message': 'Uploads cleanup completed'
        })
    except Exception as e:
        print(f"Uploads cleanup error: {e}")
        return jsonify({
            'success': False,
            'error': f'Uploads cleanup failed: {str(e)}'
        }) 