"""
API Routes Module - 7B Model Only
Handles various API endpoints for session management, health checks, utility functions, and streaming
"""

import os
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from config import Config, AGENT_CAPABILITIES, AGENT_TOOLS
from services.session_service import (
    get_session_data, cleanup_session_data, cleanup_expired_sessions, 
    cleanup_old_uploads, get_all_session_keys
)
from services.hybrid_analysis_service import hybrid_analysis_service
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
        'service': 'AI Video Detective API (7B Model)',
        'version': '2.0.0',
        'timestamp': time.time()
    })

@api_bp.route('/hybrid-analysis', methods=['POST'])
def start_hybrid_analysis():
    """Start hybrid analysis using DeepStream + 7B Model + Vector Search"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        analysis_type = data.get('analysis_type', 'hybrid')
        
        if not video_path:
            return jsonify({'error': 'Video path required'}), 400
        
        # Check if video file exists
        if not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_path}'}), 404
        
        # Start hybrid analysis asynchronously
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async hybrid analysis
            analysis_results = loop.run_until_complete(
                hybrid_analysis_service.analyze_video_hybrid(video_path, analysis_type)
            )
            
            if 'error' in analysis_results:
                return jsonify({'error': analysis_results['error']}), 500
            
            return jsonify({
                'success': True,
                'session_id': analysis_results.get('session_id'),
                'status': 'completed',
                'performance_metrics': analysis_results.get('performance_metrics', {}),
                'message': 'Hybrid analysis completed successfully'
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

@api_bp.route('/hybrid-search/<session_id>', methods=['POST'])
def search_hybrid_results(session_id):
    """Search analysis results using vector search"""
    try:
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        # Perform vector search
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async search
            search_results = loop.run_until_complete(
                hybrid_analysis_service.search_analysis_results(session_id, query, top_k)
            )
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'query': query,
                'results': search_results,
                'total_results': len(search_results)
            })
            
        except Exception as e:
            return jsonify({'error': f'Search failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to perform search: {str(e)}'}), 500

@api_bp.route('/hybrid-summary/<session_id>')
def get_hybrid_summary(session_id):
    """Get summary of hybrid analysis results"""
    try:
        # Get analysis summary
        import asyncio
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async summary retrieval
            summary = loop.run_until_complete(
                hybrid_analysis_service.get_analysis_summary(session_id)
            )
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'summary': summary
            })
            
        except Exception as e:
            return jsonify({'error': f'Summary retrieval failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Failed to get summary: {str(e)}'}), 500

@api_bp.route('/model-health')
def model_health_check():
    """Health check endpoint for AI models (7B Model Only)"""
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
            'status': 'healthy',
            'service': 'AI Model Health Check (7B Model Only)',
            'timestamp': time.time(),
            'models': health_status
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'service': 'AI Model Health Check (7B Model Only)',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@api_bp.route('/switch-model', methods=['POST'])
def switch_model():
    """Switch between different AI models (Only 7B Model Available)"""
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({'success': False, 'error': 'Model name required'})
        
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
                    'model': model_status
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to switch to {model_name}'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Model switch failed: {str(e)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'API error: {str(e)}'
        }), 500

@api_bp.route('/model-status')
def get_model_status():
    """Get current model status (7B Model Only)"""
    try:
        from services.model_manager import model_manager
        return jsonify(model_manager.get_current_model())
    except Exception as e:
        return jsonify({'error': f'Failed to get model status: {str(e)}'}), 500

@api_bp.route('/agent-info')
def get_agent_info():
    """Get agent capabilities and tools information"""
    return jsonify({
        'capabilities': AGENT_CAPABILITIES,
        'tools': AGENT_TOOLS,
        'model': 'Qwen2.5-VL-7B (Local GPU)',
        'version': '2.0.0'
    })

# =============================================================================
# STREAMING API ENDPOINTS - NEW FOR REAL-TIME ANALYSIS
# =============================================================================

@api_bp.route('/stream/start', methods=['POST'])
def start_video_stream():
    """Start real-time video streaming analysis with 7B model"""
    try:
        data = request.get_json()
        stream_id = data.get('stream_id')
        video_source = data.get('video_source')  # URL, file path, or camera index
        stream_config = data.get('config', {})
        
        if not stream_id or not video_source:
            return jsonify({
                'success': False,
                'error': 'Stream ID and video source are required'
            }), 400
        
        # Import streaming service
        from services.streaming_service import streaming_service
        import asyncio
        
        # Start the stream
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        success = loop.run_until_complete(streaming_service.start_stream(stream_id, video_source, stream_config))
        
        return jsonify({
            'success': success,
            'stream_id': stream_id,
            'message': 'Stream started successfully' if success else 'Failed to start stream',
            'config': {
                'fps_target': Config.STREAMING_CONFIG['fps_target'],
                'max_latency_ms': Config.STREAMING_CONFIG['max_latency_ms'],
                'ai_analysis_enabled': Config.STREAMING_CONFIG['real_time_7b_analysis'],
                'analysis_interval': Config.STREAMING_CONFIG['analysis_interval']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to start stream: {str(e)}'
        }), 500

@api_bp.route('/stream/stop/<stream_id>', methods=['POST'])
def stop_video_stream(stream_id):
    """Stop a video stream"""
    try:
        from services.streaming_service import streaming_service
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        success = loop.run_until_complete(streaming_service.stop_stream(stream_id))
        
        return jsonify({
            'success': success,
            'stream_id': stream_id,
            'message': 'Stream stopped successfully' if success else 'Failed to stop stream'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to stop stream: {str(e)}'
        }), 500

@api_bp.route('/stream/status/<stream_id>')
def get_stream_status(stream_id):
    """Get real-time stream status and metrics"""
    try:
        from services.streaming_service import streaming_service
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        status = loop.run_until_complete(streaming_service.get_stream_status(stream_id))
        
        return jsonify({
            'success': True,
            'stream_id': stream_id,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get stream status: {str(e)}'
        }), 500

@api_bp.route('/stream/events/<stream_id>')
def get_stream_events(stream_id):
    """Get detected events from a stream including 7B model analysis"""
    try:
        from services.streaming_service import streaming_service
        
        # Get events for the stream
        events = streaming_service.stream_events.get(stream_id, [])
        
        return jsonify({
            'success': True,
            'stream_id': stream_id,
            'events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp,
                    'confidence': event.confidence,
                    'metadata': event.metadata,
                    'ai_analysis': event.ai_analysis  # 7B model analysis results
                }
                for event in events
            ],
            'total_events': len(events),
            'ai_analysis_events': len([e for e in events if e.event_type == 'ai_analysis'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get stream events: {str(e)}'
        }), 500

@api_bp.route('/stream/all')
def get_all_streams():
    """Get status of all active streams"""
    try:
        from services.streaming_service import streaming_service
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        status = loop.run_until_complete(streaming_service.get_all_streams_status())
        
        return jsonify({
            'success': True,
            'streams': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get streams status: {str(e)}'
        }), 500

@api_bp.route('/stream/performance')
def get_streaming_performance():
    """Get overall streaming performance statistics"""
    try:
        from services.streaming_service import streaming_service
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        stats = loop.run_until_complete(streaming_service.get_performance_stats())
        
        return jsonify({
            'success': True,
            'performance': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get performance stats: {str(e)}'
        }), 500

@api_bp.route('/stream/config')
def get_streaming_config():
    """Get current streaming configuration"""
    try:
        return jsonify({
            'success': True,
            'config': {
                'enabled': Config.STREAMING_CONFIG['enabled'],
                'fps_target': Config.STREAMING_CONFIG['fps_target'],
                'max_latency_ms': Config.STREAMING_CONFIG['max_latency_ms'],
                'event_detection_enabled': Config.STREAMING_CONFIG['event_detection_enabled'],
                'continuous_processing': Config.STREAMING_CONFIG['continuous_processing'],
                'real_time_7b_analysis': Config.STREAMING_CONFIG['real_time_7b_analysis'],
                'frame_buffer_size': Config.STREAMING_CONFIG['frame_buffer_size'],
                'analysis_interval': Config.STREAMING_CONFIG['analysis_interval'],
                'event_thresholds': Config.STREAMING_CONFIG['event_thresholds']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get streaming config: {str(e)}'
        }), 500

# =============================================================================
# EXISTING API ENDPOINTS
# =============================================================================

@api_bp.route('/capture-screenshots', methods=['POST'])
def capture_screenshots():
    """Capture screenshots from video at specified timestamps"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        timestamps = data.get('timestamps', [])
        
        if not video_path or not timestamps:
            return jsonify({
                'success': False,
                'error': 'Video path and timestamps are required'
            })
        
        # Validate video file exists
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': 'Video file not found'
            })
        
        # Capture screenshots
        screenshots = []
        for timestamp in timestamps:
            try:
                screenshot_path = capture_screenshot(video_path, timestamp)
                if screenshot_path:
                    screenshots.append({
                        'timestamp': timestamp,
                        'path': screenshot_path
                    })
            except Exception as e:
                print(f"Failed to capture screenshot at {timestamp}: {e}")
        
        return jsonify({
            'success': True,
            'screenshots': screenshots,
            'total_captured': len(screenshots)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to capture screenshots: {str(e)}'
        }), 500

@api_bp.route('/auto-capture-screenshots', methods=['POST'])
def auto_capture_screenshots():
    """Automatically capture screenshots at regular intervals"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        interval_seconds = data.get('interval_seconds', 10)
        max_screenshots = data.get('max_screenshots', 20)
        
        if not video_path:
            return jsonify({
                'success': False,
                'error': 'Video path is required'
            })
        
        # Validate video file exists
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': 'Video file not found'
            })
        
        # Get video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'error': 'Failed to open video file'
            })
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        if duration <= 0:
            return jsonify({
                'success': False,
                'error': 'Invalid video duration'
            })
        
        # Calculate timestamps
        timestamps = []
        current_time = 0
        while current_time < duration and len(timestamps) < max_screenshots:
            timestamps.append(current_time)
            current_time += interval_seconds
        
        # Capture screenshots
        screenshots = []
        for timestamp in timestamps:
            try:
                screenshot_path = capture_screenshot(video_path, timestamp)
                if screenshot_path:
                    screenshots.append({
                        'timestamp': timestamp,
                        'path': screenshot_path
                    })
            except Exception as e:
                print(f"Failed to capture screenshot at {timestamp}: {e}")
        
        return jsonify({
            'success': True,
            'screenshots': screenshots,
            'total_captured': len(screenshots),
            'video_duration': duration,
            'interval_seconds': interval_seconds
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to auto-capture screenshots: {str(e)}'
        }), 500

@api_bp.route('/session/cleanup', methods=['POST'])
def cleanup_current_session():
    """Clean up current session data"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'No active session'
            })
        
        # Clean up session data
        cleanup_session_data(session_id)
        
        # Clear session
        session.clear()
        
        return jsonify({
            'success': True,
            'message': 'Session cleaned up successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to cleanup session: {str(e)}'
        }), 500

@api_bp.route('/session/status')
def get_session_status():
    """Get current session status"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'active': False,
                'message': 'No active session'
            })
        
        # Get session data
        session_data = get_session_data(session_id)
        
        if not session_data:
            return jsonify({
                'active': False,
                'message': 'Session not found'
            })
        
        return jsonify({
            'active': True,
            'session_id': session_id,
            'created_at': session_data.get('created_at'),
            'last_activity': session_data.get('last_activity'),
            'analysis_count': session_data.get('analysis_count', 0),
            'chat_count': session_data.get('chat_count', 0)
        })
        
    except Exception as e:
        return jsonify({
            'active': False,
            'error': f'Failed to get session status: {str(e)}'
        }), 500

@api_bp.route('/session/cleanup-all', methods=['POST'])
def cleanup_all_sessions():
    """Clean up all expired sessions"""
    try:
        # Clean up expired sessions
        expired_count = cleanup_expired_sessions()
        
        # Clean up old uploads
        upload_count = cleanup_old_uploads()
        
        return jsonify({
            'success': True,
            'message': 'Cleanup completed successfully',
            'expired_sessions_removed': expired_count,
            'old_uploads_removed': upload_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to cleanup sessions: {str(e)}'
        }), 500

@api_bp.route('/cleanup-uploads', methods=['POST'])
def cleanup_uploads():
    """Clean up old upload files"""
    try:
        # Clean up old uploads
        removed_count = cleanup_old_uploads()
        
        return jsonify({
            'success': True,
            'message': 'Upload cleanup completed',
            'files_removed': removed_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to cleanup uploads: {str(e)}'
        }), 500 