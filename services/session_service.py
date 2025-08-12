"""
Session Service Module - Redis-based session management
Handles Redis session management and data storage
"""

import json
import uuid
import time
import os
from datetime import datetime
import redis
from config import Config

# Configure Redis
try:
    redis_client = redis.from_url(Config.REDIS_URL)
    print("✅ Redis connection established")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None

def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def store_session_data(session_id, data):
    """Store session data in Redis"""
    try:
        if not redis_client:
            print("⚠️ Redis not available, using fallback storage")
            return
        
        # Convert complex data types to JSON strings for Redis storage
        redis_data = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                redis_data[key] = json.dumps(value)
            else:
                redis_data[key] = str(value)
        
        # Store as Redis Hash
        redis_client.hset(f"session:{session_id}", mapping=redis_data)
        redis_client.expire(f"session:{session_id}", Config.SESSION_EXPIRY)
        
        print(f"✅ Session data stored in Redis: {session_id}")
        
    except Exception as e:
        print(f"❌ Redis error: {e}")

def get_session_data(session_id):
    """Get session data from Redis"""
    try:
        if not redis_client:
            print("⚠️ Redis not available, returning empty data")
            return {}
        
        data = redis_client.hgetall(f"session:{session_id}")
        
        # Convert bytes to strings for Windows compatibility
        decoded_data = {}
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Try to decode JSON strings back to original types
            try:
                if value.startswith('[') or value.startswith('{'):
                    decoded_data[key] = json.loads(value)
                else:
                    decoded_data[key] = value
            except (json.JSONDecodeError, AttributeError):
                decoded_data[key] = value
                
        return decoded_data
        
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return {}

def cleanup_session_data(session_id):
    """Clean up all session data from local storage and delete uploaded files"""
    try:
        if not redis_client:
            print("⚠️ Redis not available, skipping cleanup")
            return False
        
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
    """Clean up expired sessions"""
    try:
        if not redis_client:
            print("⚠️ Redis not available, skipping expired session cleanup")
            return
        
        # Get all session keys from Redis
        session_keys = redis_client.keys("session:*")
        
        for key in session_keys:
            session_id = key.decode('utf-8').replace('session:', '')
            
            # Check if session is expired
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No expiration set, set it now
                redis_client.expire(key, Config.SESSION_EXPIRY)
            elif ttl == -2:  # Key doesn't exist
                continue
            elif ttl == 0:  # Expired, clean it up
                cleanup_session_data(session_id)
        
        print("✅ Expired sessions cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during expired session cleanup: {e}")

def cleanup_old_uploads():
    """Clean up old uploaded files"""
    try:
        upload_folder = Config.UPLOAD_FOLDER
        if not os.path.exists(upload_folder):
            return
        
        current_time = time.time()
        max_age = Config.SESSION_EXPIRY  # Use session expiry as file age limit
        
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            
            # Check file age
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age:
                    try:
                        os.remove(file_path)
                        print(f"✅ Deleted old file: {filename}")
                    except Exception as e:
                        print(f"⚠️ Error deleting old file {filename}: {e}")
        
        print("✅ Old uploads cleanup completed")
        
    except Exception as e:
        print(f"❌ Error during old uploads cleanup: {e}")

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