"""
Chat Routes Module
Handles chat functionality and AI responses
"""

import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from config import Config
from services.session_service import get_session_data, store_session_data
from services.ai_service_fixed import minicpm_service
from utils.video_utils import create_evidence_for_timestamps
from utils.text_utils import extract_timestamps_from_text, extract_timestamp_ranges_from_text

# Create Blueprint
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with enhanced AI responses"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        session_id = session.get('session_id')
        print(f"Debug: Chat - Session ID: {session_id}")
        
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        # Get session data including video analysis
        session_data = get_session_data(session_id)
        print(f"Debug: Chat - Session data keys: {list(session_data.keys()) if session_data else 'None'}")
        
        # Store chat message
        chat_history = session_data.get('chat_history', [])
        if isinstance(chat_history, str):
            try:
                chat_list = json.loads(chat_history)
            except json.JSONDecodeError:
                chat_list = []
        else:
            chat_list = chat_history if isinstance(chat_history, list) else []
        
        chat_list.append({
            'user': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get video analysis results for context
        analysis_result = session_data.get('analysis_result', '')
        analysis_type = session_data.get('analysis_type', 'comprehensive_analysis')
        user_focus = session_data.get('user_focus', '')
        
        print(f"Debug: Chat - Analysis result length: {len(analysis_result) if analysis_result else 0}")
        
        # Generate contextual AI response based on video analysis
        if analysis_result:
            ai_response = minicpm_service.generate_chat_response(analysis_result, analysis_type, user_focus, message, chat_list)
        else:
            # No analysis available yet
            ai_response = f"I don't have the video analysis results yet. Please first analyze the uploaded video, then I can help you with: {message}. Click 'Start Analysis' to begin the video analysis."
        
        # Capture additional evidence for timestamps mentioned in the response
        additional_evidence = []
        if analysis_result and ai_response:
            try:
                video_path = session_data.get('filepath', '')
                if video_path and os.path.exists(video_path):
                    # Extract video metadata to validate timestamps
                    from utils.video_utils import extract_video_metadata
                    video_metadata = extract_video_metadata(video_path)
                    video_duration = video_metadata.get('duration', 0) if video_metadata else 0
                    
                    # Extract timestamps from response
                    response_timestamps = extract_timestamps_from_text(ai_response)
                    # Extract timestamp ranges from response
                    timestamp_ranges = extract_timestamp_ranges_from_text(ai_response)
                    
                    # Clean and deduplicate timestamps
                    from utils.text_utils import clean_and_deduplicate_timestamps, validate_and_fix_timestamps
                    response_timestamps = clean_and_deduplicate_timestamps(response_timestamps)
                    
                    # Filter timestamps to only include those within video duration
                    if video_duration > 0:
                        # First validate and fix any out-of-bounds timestamps
                        response_timestamps = validate_and_fix_timestamps(response_timestamps, video_duration)
                        
                        # Then filter to ensure all are within bounds
                        valid_timestamps = [ts for ts in response_timestamps if 0 <= ts < video_duration]
                        if len(valid_timestamps) != len(response_timestamps):
                            print(f"⚠️ Chat: Filtered out {len(response_timestamps) - len(valid_timestamps)} invalid timestamps beyond video duration ({video_duration:.2f}s)")
                            response_timestamps = valid_timestamps
                        
                        # Filter timestamp ranges
                        valid_ranges = [(start, end) for start, end in timestamp_ranges if 0 <= start <= end < video_duration]
                        if len(valid_ranges) != len(timestamp_ranges):
                            print(f"⚠️ Chat: Filtered out {len(timestamp_ranges) - len(valid_ranges)} invalid timestamp ranges beyond video duration")
                            timestamp_ranges = valid_ranges
                    
                    # Create evidence for individual timestamps
                    if response_timestamps:
                        additional_evidence.extend(create_evidence_for_timestamps(response_timestamps, video_path, session_id, Config.UPLOAD_FOLDER))
                    
                    # Create video clips for timestamp ranges
                    for start_time, end_time in timestamp_ranges:
                        from utils.video_utils import extract_video_clip
                        clip_data = extract_video_clip(video_path, start_time, end_time, session_id, Config.UPLOAD_FOLDER)
                        if clip_data:
                            additional_evidence.append(clip_data)
                    
                    print(f"Debug: Chat - Captured {len(additional_evidence)} additional evidence items")
            except Exception as e:
                print(f"Debug: Chat - Error capturing additional evidence: {e}")
        
        chat_list.append({
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update session data with new chat history
        try:
            session_data['chat_history'] = chat_list
            store_session_data(session_id, session_data)
            print(f"Debug: Chat - Successfully updated session data")
        except Exception as e:
            print(f"Debug: Chat - Error updating session data: {e}")
            # Continue without failing the request
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'chat_history': chat_list,
            'additional_screenshots': additional_evidence
        })
        
    except Exception as e:
        print(f"Debug: Chat - Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500 