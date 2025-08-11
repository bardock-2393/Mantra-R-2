"""
Main Routes Module
Handles core application routes like index, upload, and analysis
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
from config import Config
from services.session_service import generate_session_id, store_session_data, get_session_data, cleanup_old_uploads
from services.ai_service_fixed import minicpm_service
from services.vector_search_service import vector_search_service
from utils.video_utils import extract_video_metadata, create_evidence_for_timestamps
from utils.text_utils import extract_timestamps_from_text, extract_timestamp_ranges_from_text
from analysis_templates import ANALYSIS_TEMPLATES

# Create Blueprint
main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@main_bp.route('/')
def index():
    """Main page"""
    # Clean up old files when user visits the page
    cleanup_old_uploads()
    
    if 'session_id' not in session:
        session['session_id'] = generate_session_id()
    return render_template('index.html')

@main_bp.route('/api/analysis-types')
def get_analysis_types():
    """Get available analysis types"""
    return jsonify({
        'analysis_types': ANALYSIS_TEMPLATES,
        'success': True
    })

@main_bp.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        session_id = session.get('session_id', generate_session_id())
        print(f"Debug: Upload - Session ID: {session_id}")
        
        # Check if video file was uploaded
        if 'video' in request.files and request.files['video'].filename != '':
            # User uploaded a file
            file = request.files['video']
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, WebM, or MKV'}), 400
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            
            # Save file
            file.save(filepath)
            
            # Extract video metadata
            video_metadata = extract_video_metadata(filepath)
            
            file_info = {
                'filename': unique_filename,
                'original_name': filename,
                'filepath': filepath,
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded',
                'metadata': json.dumps(video_metadata),
                'is_default_video': False
            }
            
        else:
            # No file uploaded, use default video
            default_video_path = Config.DEFAULT_VIDEO_PATH
            
            if not os.path.exists(default_video_path):
                return jsonify({'error': 'Default video file not found'}), 500
            
            # Copy default video to uploads folder with unique name
            filename = os.path.basename(default_video_path)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            
            # Copy the default video file
            shutil.copy2(default_video_path, filepath)
            
            # Extract video metadata
            video_metadata = extract_video_metadata(filepath)
            
            file_info = {
                'filename': unique_filename,
                'original_name': filename,
                'filepath': filepath,
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded',
                'metadata': json.dumps(video_metadata),
                'is_default_video': True
            }
        
        # Store file info in session
        store_session_data(session_id, file_info)
        print(f"✅ Session data stored locally: {session_id}")
        print(f"Debug: Upload - Stored data keys: {list(file_info.keys())}")
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': file_info['filename'],
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@main_bp.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON data received'}), 400
        
        analysis_type = data.get('analysis_type', 'comprehensive_analysis')
        user_focus = data.get('user_focus', 'Analyze this video comprehensively')
        
        session_id = session.get('session_id')
        print(f"Debug: Session ID: {session_id}")
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session found'}), 400
        
        # Get session data
        session_data = get_session_data(session_id)
        print(f"Debug: Session data keys: {list(session_data.keys()) if session_data else 'None'}")
        print(f"Debug: Session data: {session_data}")
        
        if not session_data or 'filepath' not in session_data:
            return jsonify({'success': False, 'error': 'No video uploaded'}), 400
        
        video_path = session_data['filepath']
        print(f"Debug: Video path: {video_path}")
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
        
        # Perform analysis using model manager
        from services.model_manager import model_manager
        import asyncio
        
        analysis_result = None
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            analysis_result = loop.run_until_complete(model_manager.analyze_video(video_path, analysis_type, user_focus))
            
            # Validate the analysis result
            if not analysis_result or analysis_result.startswith("Error:"):
                print(f"❌ Analysis returned error: {analysis_result}")
                return jsonify({
                    'success': False, 
                    'error': f'Analysis failed: {analysis_result}',
                    'analysis': analysis_result
                }), 500
                
        except Exception as e:
            print(f"❌ Error in video analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'Analysis failed: {str(e)}',
                'details': 'Check server logs for more information'
            }), 500
        
        # Extract video metadata to validate timestamps
        video_metadata = extract_video_metadata(video_path)
        video_duration = video_metadata.get('duration', 0) if video_metadata else 0
        
        print(f"🎯 Video duration: {video_duration:.2f} seconds")
        
        # Extract timestamps and capture screenshots automatically
        timestamps = extract_timestamps_from_text(analysis_result)
        print(f"🎯 Raw extracted timestamps: {timestamps}")
        
        # Clean and deduplicate timestamps
        from utils.text_utils import clean_and_deduplicate_timestamps, aggressive_timestamp_validation
        timestamps = clean_and_deduplicate_timestamps(timestamps)
        print(f"🎯 Cleaned timestamps: {timestamps}")
        
        # Filter timestamps to only include those within video duration
        if video_duration > 0:
            # Use aggressive validation to completely remove out-of-bounds timestamps
            timestamps = aggressive_timestamp_validation(timestamps, video_duration)
            print(f"🎯 Final valid timestamps: {timestamps}")
        else:
            print("⚠️ Warning: Could not determine video duration, using all extracted timestamps")
        
        evidence = []
        
        if timestamps:
            # Create evidence (screenshots or video clips) based on timeframe length
            evidence = create_evidence_for_timestamps(timestamps, video_path, session_id, Config.UPLOAD_FOLDER)
        
        # Store analysis results and evidence in session
        analysis_data = {
            'analysis_result': analysis_result,
            'analysis_type': analysis_type,
            'user_focus': user_focus,
            'timestamps_found': timestamps,
            'evidence': evidence,
            'analysis_time': datetime.now().isoformat(),
            'video_duration': video_duration,
            'video_metadata': video_metadata
        }
        
        # Update session data
        session_data.update(analysis_data)
        store_session_data(session_id, session_data)
        
        # Create vector embeddings for semantic search
        try:
            vector_search_service.create_embeddings(session_id, analysis_data)
            print(f"✅ Vector embeddings created for session {session_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not create vector embeddings: {e}")
        
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'timestamps': timestamps,
            'evidence': evidence,
            'evidence_count': len(evidence),
            'video_duration': video_duration,
            'vector_search_available': True
        })
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': str(e),
            'details': 'Unexpected error occurred during analysis'
        }), 500

@main_bp.route('/screenshot/<filename>')
def get_screenshot(filename):
    """Serve screenshot files"""
    try:
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Screenshot not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/demo-video')
def get_demo_video():
    """Serve the demo video file"""
    try:
        demo_video_path = Config.DEFAULT_VIDEO_PATH
        if os.path.exists(demo_video_path):
            return send_file(demo_video_path, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Demo video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500 