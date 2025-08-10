"""
Chat Routes Module
Handles chat functionality and AI responses with vector search
"""

import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from config import Config
from services.session_service import get_session_data, store_session_data
from services.ai_service_fixed import minicpm_service
from services.vector_search_service import vector_search_service
from utils.video_utils import create_evidence_for_timestamps
from utils.text_utils import extract_timestamps_from_text, extract_timestamp_ranges_from_text

# Create Blueprint
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with enhanced AI responses using vector search"""
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
        
        # Use vector search to find relevant content for the question
        relevant_content = []
        if analysis_result:
            try:
                # Search for content similar to the user's question
                relevant_content = vector_search_service.search_similar_content(session_id, message, top_k=3)
                print(f"🔍 Vector search found {len(relevant_content)} relevant content items")
                
                # Format relevant content for context
                context_info = ""
                if relevant_content:
                    context_info = "\n\n**Relevant Content Found:**\n"
                    for i, content in enumerate(relevant_content, 1):
                        context_info += f"{i}. {content['text']}\n"
                        if content.get('timestamp'):
                            context_info += f"   Timestamp: {content['timestamp']:.2f}s\n"
                        if content.get('start_time') and content.get('end_time'):
                            context_info += f"   Range: {content['start_time']:.2f}s - {content['end_time']:.2f}s\n"
                        context_info += f"   Relevance: {content['similarity_score']:.3f}\n\n"
                
                # Generate contextual AI response based on video analysis and relevant content
                enhanced_message = f"{message}\n\n{context_info}" if context_info else message
                ai_response = minicpm_service.generate_chat_response(
                    analysis_result, analysis_type, user_focus, enhanced_message, chat_list
                )
                
            except Exception as e:
                print(f"⚠️ Vector search failed, falling back to basic response: {e}")
                # Fallback to basic response
                ai_response = minicpm_service.generate_chat_response(
                    analysis_result, analysis_type, user_focus, message, chat_list
                )
        else:
            # No analysis available yet
            ai_response = f"I don't have the video analysis results yet. Please first analyze the uploaded video, then I can help you with: {message}. Click 'Start Analysis' to begin the video analysis."
        
        # Capture additional evidence for timestamps mentioned in the response
        additional_evidence = []
        if analysis_result and ai_response:
            try:
                video_path = session_data.get('filepath', '')
                if video_path and os.path.exists(video_path):
                    # Extract timestamps from response
                    response_timestamps = extract_timestamps_from_text(ai_response)
                    # Extract timestamp ranges from response
                    timestamp_ranges = extract_timestamp_ranges_from_text(ai_response)
                    
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
            'additional_screenshots': additional_evidence,
            'relevant_content': relevant_content,
            'vector_search_used': len(relevant_content) > 0
        })
        
    except Exception as e:
        print(f"Debug: Chat - Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@chat_bp.route('/search', methods=['POST'])
def search_content():
    """Search for specific content in the video analysis"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        # Search for relevant content
        relevant_content = vector_search_service.search_similar_content(session_id, query, top_k=5)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': relevant_content,
            'result_count': len(relevant_content)
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/vector-status', methods=['GET'])
def get_vector_status():
    """Get the status of vector search for the current session"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        summary = vector_search_service.get_session_summary(session_id)
        
        return jsonify({
            'success': True,
            'vector_search_available': summary['available'],
            'summary': summary
        })
        
    except Exception as e:
        print(f"Vector status error: {e}")
        return jsonify({'error': str(e)}), 500 