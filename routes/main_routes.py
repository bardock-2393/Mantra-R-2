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
# COMMENTED OUT TO SAVE MEMORY - ONLY USE 32B MODEL
# from services.ai_service_fixed import minicpm_service
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
        print("🔍 Debug: Starting upload process...")
        session_id = session.get('session_id', generate_session_id())
        print(f"Debug: Upload - Session ID: {session_id}")
        
        # Check if video file was uploaded
        if 'video' in request.files and request.files['video'].filename != '':
            print("🔍 Debug: User uploaded a file")
            # User uploaded a file
            file = request.files['video']
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, WebM, or MKV'}), 400
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            print(f"🔍 Debug: File will be saved to: {filepath}")
            
            # Save file
            file.save(filepath)
            print(f"🔍 Debug: File saved successfully")
            
            # Extract video metadata
            print(f"🔍 Debug: Extracting video metadata from: {filepath}")
            video_metadata = extract_video_metadata(filepath)
            print(f"🔍 Debug: Video metadata extracted: {video_metadata}")
            
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
            print("🔍 Debug: No file uploaded, using default video")
            # No file uploaded, use default video
            default_video_path = Config.DEFAULT_VIDEO_PATH
            print(f"🔍 Debug: Default video path: {default_video_path}")
            
            if not os.path.exists(default_video_path):
                print(f"❌ Debug: Default video not found at: {default_video_path}")
                return jsonify({'error': 'Default video file not found'}), 500
            
            # Copy default video to uploads folder with unique name
            filename = os.path.basename(default_video_path)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
            print(f"🔍 Debug: Will copy to: {filepath}")
            
            # Copy the default video file
            print(f"🔍 Debug: Copying default video...")
            shutil.copy2(default_video_path, filepath)
            print(f"🔍 Debug: Default video copied successfully")
            
            # Extract video metadata
            print(f"🔍 Debug: Extracting metadata from copied video...")
            video_metadata = extract_video_metadata(filepath)
            print(f"🔍 Debug: Metadata extracted: {video_metadata}")
            
            file_info = {
                'filename': unique_filename,
                'original_name': filename,
                'filepath': filepath,
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded',
                'metadata': json.dumps(video_metadata),
                'is_default_video': True
            }
        
        print(f"Debug: Upload - File info: {file_info}")
        print(f"🔍 Debug: Storing session data...")
        store_session_data(session_id, file_info)
        print(f"🔍 Debug: Session data stored")
        
        # Verify storage
        stored_data = get_session_data(session_id)
        print(f"Debug: Upload - Stored data keys: {list(stored_data.keys()) if stored_data else 'None'}")
        
        message = 'Default video loaded successfully' if file_info.get('is_default_video', False) else 'Video uploaded successfully'
        print(f"🔍 Debug: Upload completed successfully")
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'message': message,
            'is_default_video': file_info.get('is_default_video', False)
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Upload error: {e}")
        print(f"❌ Upload error traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@main_bp.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video"""
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type', 'comprehensive_analysis')
        user_focus = data.get('user_focus', 'Analyze this video comprehensively')
        
        session_id = session.get('session_id')
        print(f"Debug: Session ID: {session_id}")
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session found'})
        
        # Get session data
        session_data = get_session_data(session_id)
        print(f"Debug: Session data keys: {list(session_data.keys()) if session_data else 'None'}")
        print(f"Debug: Session data: {session_data}")
        
        if not session_data or 'filepath' not in session_data:
            return jsonify({'success': False, 'error': 'No video uploaded'})
        
        video_path = session_data['filepath']
        print(f"Debug: Video path: {video_path}")
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'})
        
        # Analyze video using 32B AI service
        from services.qwen25vl_32b_service import qwen25vl_32b_service
        import asyncio
        
        # Debug: Check 32B service status
        print(f"🔍 32B Service Status: {qwen25vl_32b_service.is_ready()}")
        print(f"🔍 32B Service Initialized: {qwen25vl_32b_service.is_initialized}")
        print(f"🔍 32B Service Model: {qwen25vl_32b_service.model is not None if hasattr(qwen25vl_32b_service, 'model') else 'No model attribute'}")
        
        try:
            # Check if 32B service is ready
            if not qwen25vl_32b_service.is_ready():
                print("🔄 32B service not ready, initializing...")
                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Initialize the 32B service
                loop.run_until_complete(qwen25vl_32b_service.initialize())
                print("✅ 32B service initialized")
                
                # Check status again
                print(f"🔍 After init - 32B Service Status: {qwen25vl_32b_service.is_ready()}")
                print(f"🔍 After init - 32B Service Initialized: {qwen25vl_32b_service.is_initialized}")
            
            # Get or create event loop for analysis
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async analysis using 32B service
            print(f"🎬 Starting video analysis with 32B model...")
            analysis_result = loop.run_until_complete(qwen25vl_32b_service.analyze(
                video_path, analysis_type, user_focus
            ))
            
            # Validate analysis result
            if analysis_result and len(analysis_result) > 50:
                print(f"✅ Video analysis completed successfully using 32B model")
                print(f"📊 Analysis length: {len(analysis_result)} characters")
            else:
                print(f"⚠️ Analysis result seems incomplete, length: {len(analysis_result) if analysis_result else 0}")
                # Generate a basic analysis as fallback
                analysis_result = f"""**Video Analysis Report**

**Status:** Basic analysis completed

**Video File:** {os.path.basename(video_path)}
**Analysis Type:** {analysis_type}
**User Focus:** {user_focus}

**Note:** The AI model analysis was incomplete. This is a basic analysis based on available information.

**What I can tell you:**
- The video file exists and is accessible
- Basic technical specifications can be extracted
- For detailed content analysis, the 32B AI model needs to be fully loaded

**Next Steps:**
1. Ensure the 32B model is properly loaded
2. Check GPU memory availability (requires ~80GB)
3. Verify HuggingFace authentication token"""
                print(f"✅ Generated fallback analysis")
            
        except Exception as e:
            print(f"❌ 32B AI analysis failed: {e}")
            # Generate a comprehensive fallback analysis
            try:
                print(f"🔄 Attempting fallback analysis...")
                analysis_result = loop.run_until_complete(qwen25vl_32b_service._generate_text_only_analysis(
                    f"Analyze this video with focus on: {user_focus}", video_path
                ))
                print(f"✅ Fallback analysis completed")
            except Exception as fallback_error:
                print(f"❌ Fallback analysis also failed: {fallback_error}")
                analysis_result = f"""**Video Analysis Report**

**Status:** Analysis failed - using basic information

**Video File:** {os.path.basename(video_path)}
**Analysis Type:** {analysis_type}
**User Focus:** {user_focus}

**Error Details:** {str(e)}

**What I can tell you:**
- The video file exists and is accessible
- The system encountered an error during analysis
- For detailed content analysis, the 32B AI model needs to be properly configured

**Next Steps:**
1. Check the error logs for details
2. Ensure the 32B model is properly loaded
3. Verify GPU memory availability (requires ~80GB)
4. Check HuggingFace authentication token

**Your video is ready for analysis once the system issues are resolved.**"""
            
            print(f"⚠️ Using fallback analysis result")
        
        # Use stored video metadata from session instead of re-extracting
        stored_metadata = session_data.get('metadata')
        print(f"🔍 Debug: Stored metadata type: {type(stored_metadata)}")
        print(f"🔍 Debug: Stored metadata content: {stored_metadata}")
        
        if stored_metadata and isinstance(stored_metadata, str):
            try:
                video_metadata = json.loads(stored_metadata)
                print(f"🔍 Debug: Parsed metadata: {video_metadata}")
                video_duration = video_metadata.get('duration', 0)
                print(f"🎯 Using stored video duration: {video_duration:.2f} seconds")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse stored metadata: {e}, re-extracting...")
                video_metadata = extract_video_metadata(video_path)
                video_duration = video_metadata.get('duration', 0) if video_metadata else 0
                print(f"🎯 Re-extracted video duration: {video_duration:.2f} seconds")
        else:
            # Fallback to re-extraction if no stored metadata
            print("⚠️ No stored metadata found, re-extracting...")
            video_metadata = extract_video_metadata(video_path)
            video_duration = video_metadata.get('duration', 0) if video_metadata else 0
            print(f"🎯 Fallback video duration: {video_duration:.2f} seconds")
        
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
        
        # DISABLED: Evidence creation (clips and images)
        evidence = []
        print("🚫 Evidence creation disabled - no clips or images will be generated")
        
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
            'evidence': evidence,  # Empty list - no evidence
            'evidence_count': 0,  # Always 0
            'video_duration': video_duration,
            'vector_search_available': True
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# DISABLED: Screenshot serving route
# @main_bp.route('/screenshot/<filename>')
# def get_screenshot(filename):
#     """Serve screenshot files - DISABLED"""
#     return jsonify({'error': 'Screenshot feature has been disabled'}), 404

@main_bp.route('/demo-video')
def demo_video():
    """Serve demo video file"""
    try:
        demo_path = Config.DEFAULT_VIDEO_PATH
        if os.path.exists(demo_path):
            return send_file(demo_path, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Demo video not found'}), 404
    except Exception as e:
        print(f"Demo video error: {e}")
        return jsonify({'error': f'Demo video error: {str(e)}'}), 500 