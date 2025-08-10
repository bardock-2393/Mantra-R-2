"""
Session Service Module for Round 2 - Local file-based session management
Handles local session storage without external dependencies
"""

import json
import uuid
import time
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from config import Config

class LocalSessionService:
    """Local file-based session management service"""
    
    def __init__(self):
        self.sessions_dir = Config.SESSION_STORAGE_PATH
        self._ensure_sessions_directory()
    
    def _ensure_sessions_directory(self):
        """Ensure the sessions directory exists"""
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir, exist_ok=True)
    
    def _get_session_file_path(self, session_id: str) -> str:
        """Get the file path for a session"""
        return os.path.join(self.sessions_dir, f"session_{session_id}.pkl")
    
    def _get_session_metadata_path(self, session_id: str) -> str:
        """Get the metadata file path for a session"""
        return os.path.join(self.sessions_dir, f"metadata_{session_id}.json")

def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def store_session_data(session_id, data):
    """Store session data in local file storage"""
    try:
        session_service = LocalSessionService()
        session_file = session_service._get_session_file_path(session_id)
        metadata_file = session_service._get_session_metadata_path(session_id)
        
        # Store main session data
        with open(session_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Store metadata for quick access
        metadata = {
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'expires_at': datetime.fromtimestamp(datetime.now().timestamp() + Config.SESSION_EXPIRY).isoformat(),
            'data_keys': list(data.keys())
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
            
        print(f"✅ Session data stored locally: {session_id}")
        
    except Exception as e:
        print(f"❌ Error storing session data: {e}")

def get_session_data(session_id):
    """Get session data from local file storage"""
    try:
        session_service = LocalSessionService()
        session_file = session_service._get_session_file_path(session_id)
        metadata_file = session_service._get_session_metadata_path(session_id)
        
        # Check if session exists and is not expired
        if not os.path.exists(session_file) or not os.path.exists(metadata_file):
            return {}
        
        # Check expiration
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        expires_at = datetime.fromisoformat(metadata['expires_at'])
        if datetime.now() > expires_at:
            cleanup_session_data(session_id)
            return {}
        
        # Update last accessed time
        metadata['last_accessed'] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Load session data
        with open(session_file, 'rb') as f:
            data = pickle.load(f)
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading session data: {e}")
        return {}

def cleanup_session_data(session_id):
    """Clean up all session data from local storage and delete uploaded files"""
    try:
        session_service = LocalSessionService()
        session_file = session_service._get_session_file_path(session_id)
        metadata_file = session_service._get_session_metadata_path(session_id)
        
        # Get session data to find uploaded files
        session_data = get_session_data(session_id)
        
        # Delete uploaded video file
        if session_data and 'filepath' in session_data:
            video_path = session_data['filepath']
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"✅ Deleted video file: {video_path}")
                except Exception as e:
                    print(f"❌ Error deleting video file {video_path}: {e}")
        
        # Delete all screenshots and video clips for this session
        upload_folder = Config.UPLOAD_FOLDER
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                if filename.startswith(f"screenshot_{session_id}_") or filename.startswith(f"clip_{session_id}_"):
                    file_path = os.path.join(upload_folder, filename)
                    try:
                        os.remove(file_path)
                        print(f"✅ Deleted evidence file: {filename}")
                    except Exception as e:
                        print(f"❌ Error deleting evidence file {filename}: {e}")
        
        # Delete session files
        try:
            if os.path.exists(session_file):
                os.remove(session_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            print(f"✅ Deleted session files: {session_id}")
        except Exception as e:
            print(f"❌ Error deleting session files: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in cleanup_session_data: {e}")
        return False

def cleanup_expired_sessions():
    """Clean up all expired sessions"""
    try:
        session_service = LocalSessionService()
        sessions_dir = session_service.sessions_dir
        
        if not os.path.exists(sessions_dir):
            return
        
        cleaned_count = 0
        for filename in os.listdir(sessions_dir):
            if filename.startswith("metadata_"):
                session_id = filename.replace("metadata_", "").replace(".json", "")
                
                try:
                    metadata_file = os.path.join(sessions_dir, filename)
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    expires_at = datetime.fromisoformat(metadata['expires_at'])
                    if datetime.now() > expires_at:
                        if cleanup_session_data(session_id):
                            cleaned_count += 1
                            
                except Exception as e:
                    print(f"❌ Error processing metadata file {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} expired sessions")
            
    except Exception as e:
        print(f"❌ Error in cleanup_expired_sessions: {e}")

def cleanup_old_uploads():
    """Clean up old uploaded files"""
    try:
        upload_folder = Config.UPLOAD_FOLDER
        if not os.path.exists(upload_folder):
            return
        
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check file age
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > Config.UPLOAD_CLEANUP_TIME:
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    print(f"🧹 Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"❌ Error deleting old file {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} old files")
            
    except Exception as e:
        print(f"❌ Error in cleanup_old_uploads: {e}")

def get_all_session_keys():
    """Get all active session keys"""
    try:
        session_service = LocalSessionService()
        sessions_dir = session_service.sessions_dir
        
        if not os.path.exists(sessions_dir):
            return []
        
        session_keys = []
        for filename in os.listdir(sessions_dir):
            if filename.startswith("metadata_"):
                session_id = filename.replace("metadata_", "").replace(".json", "")
                
                try:
                    metadata_file = os.path.join(sessions_dir, filename)
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    expires_at = datetime.fromisoformat(metadata['expires_at'])
                    if datetime.now() <= expires_at:
                        session_keys.append(session_id)
                        
                except Exception as e:
                    print(f"❌ Error processing metadata file {filename}: {e}")
        
        return session_keys
        
    except Exception as e:
        print(f"❌ Error getting session keys: {e}")
        return [] 