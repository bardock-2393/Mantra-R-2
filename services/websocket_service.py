"""
WebSocket Service for Real-time Progress Updates
Handles real-time communication for long video processing progress
"""

import json
import time
import threading
from typing import Dict, List, Optional, Callable
from flask_socketio import SocketIO, emit, join_room, leave_room
import queue

class WebSocketService:
    """Service for managing WebSocket connections and progress updates"""
    
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.active_sessions = {}  # session_id -> connection info
        self.progress_trackers = {}  # session_id -> progress info
        self.cleanup_thread = None
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background cleanup thread for expired sessions"""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_expired_sessions()
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def register_session(self, session_id: str, user_id: str = None):
        """Register a new session for progress tracking"""
        try:
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'created_at': time.time(),
                'last_activity': time.time(),
                'status': 'active'
            }
            
            self.progress_trackers[session_id] = {
                'current_chunk': 0,
                'total_chunks': 0,
                'overall_progress': 0.0,
                'chunk_progress': 0.0,
                'start_time': time.time(),
                'estimated_total_time': 0,
                'current_chunk_start': time.time(),
                'current_chunk_estimated_time': 0,
                'status': 'initializing'
            }
            
            print(f"🔗 WebSocket session registered: {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to register WebSocket session: {e}")
            return False
    
    def unregister_session(self, session_id: str):
        """Unregister a session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.progress_trackers:
                del self.progress_trackers[session_id]
            print(f"🔌 WebSocket session unregistered: {session_id}")
        except Exception as e:
            print(f"❌ Failed to unregister WebSocket session: {e}")
    
    def update_progress(self, session_id: str, 
                       chunk_progress: float = None,
                       overall_progress: float = None,
                       current_chunk: int = None,
                       total_chunks: int = None,
                       status: str = None,
                       message: str = None):
        """Update progress for a session and emit to client"""
        try:
            if session_id not in self.progress_trackers:
                print(f"⚠️ Session {session_id} not found in progress trackers")
                return False
            
            tracker = self.progress_trackers[session_id]
            
            # Update progress values
            if chunk_progress is not None:
                tracker['chunk_progress'] = chunk_progress
            if overall_progress is not None:
                tracker['overall_progress'] = overall_progress
            if current_chunk is not None:
                tracker['current_chunk'] = current_chunk
            if total_chunks is not None:
                tracker['total_chunks'] = total_chunks
            if status is not None:
                tracker['status'] = status
            
            # Update timestamps and estimates
            current_time = time.time()
            if current_chunk != tracker['current_chunk']:
                tracker['current_chunk_start'] = current_time
                tracker['current_chunk_estimated_time'] = 0
            
            # Calculate time estimates
            if tracker['overall_progress'] > 0:
                elapsed_time = current_time - tracker['start_time']
                estimated_total_time = elapsed_time / tracker['overall_progress']
                tracker['estimated_total_time'] = estimated_total_time
            
            if tracker['chunk_progress'] > 0:
                chunk_elapsed = current_time - tracker['current_chunk_start']
                chunk_estimated = chunk_elapsed / tracker['chunk_progress']
                tracker['current_chunk_estimated_time'] = chunk_estimated
            
            # Emit progress update to client
            self._emit_progress_update(session_id, tracker, message)
            
            # Update last activity
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['last_activity'] = current_time
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update progress: {e}")
            return False
    
    def _emit_progress_update(self, session_id: str, tracker: Dict, message: str = None):
        """Emit progress update to the client"""
        try:
            # Calculate time remaining
            current_time = time.time()
            overall_remaining = max(0, tracker['estimated_total_time'] - (current_time - tracker['start_time']))
            chunk_remaining = max(0, tracker['current_chunk_estimated_time'] - (current_time - tracker['current_chunk_start']))
            
            # Prepare progress data
            progress_data = {
                'session_id': session_id,
                'timestamp': current_time,
                'current_chunk': tracker['current_chunk'],
                'total_chunks': tracker['total_chunks'],
                'chunk_progress': round(tracker['chunk_progress'], 2),
                'overall_progress': round(tracker['overall_progress'], 2),
                'status': tracker['status'],
                'time_remaining': {
                    'overall_seconds': round(overall_remaining),
                    'chunk_seconds': round(chunk_remaining),
                    'overall_formatted': self._format_time(overall_remaining),
                    'chunk_formatted': self._format_time(chunk_remaining)
                },
                'message': message or self._get_status_message(tracker)
            }
            
            # Emit to specific session room
            self.socketio.emit('progress_update', progress_data, room=session_id)
            
            # Also emit to general progress room for monitoring
            self.socketio.emit('general_progress', progress_data, room='progress_monitors')
            
            print(f"📊 Progress update sent: Chunk {tracker['current_chunk']}/{tracker['total_chunks']} - {tracker['overall_progress']:.1f}%")
            
        except Exception as e:
            print(f"❌ Failed to emit progress update: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        try:
            if seconds < 60:
                return f"{int(seconds)}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        except:
            return "Unknown"
    
    def _get_status_message(self, tracker: Dict) -> str:
        """Generate status message based on progress"""
        try:
            if tracker['status'] == 'initializing':
                return "Initializing video processing..."
            elif tracker['status'] == 'chunking':
                return f"Preparing {tracker['total_chunks']} video chunks..."
            elif tracker['status'] == 'processing':
                if tracker['total_chunks'] > 0:
                    return f"Processing chunk {tracker['current_chunk']} of {tracker['total_chunks']} ({tracker['overall_progress']:.1f}%)"
                else:
                    return "Processing video..."
            elif tracker['status'] == 'merging':
                return "Merging chunk results..."
            elif tracker['status'] == 'completed':
                return "Video analysis completed successfully!"
            elif tracker['status'] == 'error':
                return "An error occurred during processing"
            else:
                return "Processing video..."
        except:
            return "Processing video..."
    
    def start_chunk_processing(self, session_id: str, total_chunks: int):
        """Start chunk processing progress tracking"""
        try:
            if session_id in self.progress_trackers:
                tracker = self.progress_trackers[session_id]
                tracker['total_chunks'] = total_chunks
                tracker['current_chunk'] = 0
                tracker['overall_progress'] = 0.0
                tracker['chunk_progress'] = 0.0
                tracker['status'] = 'chunking'
                tracker['start_time'] = time.time()
                
                self.update_progress(session_id, status='chunking', 
                                   message=f"Starting processing of {total_chunks} video chunks")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to start chunk processing: {e}")
            return False
    
    def update_chunk_progress(self, session_id: str, chunk_id: int, chunk_progress: float):
        """Update progress for a specific chunk"""
        try:
            if session_id in self.progress_trackers:
                tracker = self.progress_trackers[session_id]
                
                # Update chunk progress
                tracker['chunk_progress'] = chunk_progress
                
                # Calculate overall progress
                if tracker['total_chunks'] > 0:
                    chunk_weight = 1.0 / tracker['total_chunks']
                    completed_chunks = tracker['current_chunk'] - 1
                    overall_progress = (completed_chunks + chunk_progress) * chunk_weight
                    tracker['overall_progress'] = min(overall_progress, 1.0)
                
                self.update_progress(session_id, chunk_progress=chunk_progress, 
                                   overall_progress=tracker['overall_progress'])
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to update chunk progress: {e}")
            return False
    
    def complete_chunk(self, session_id: str, chunk_id: int):
        """Mark a chunk as completed"""
        try:
            if session_id in self.progress_trackers:
                tracker = self.progress_trackers[session_id]
                
                # Move to next chunk
                tracker['current_chunk'] = chunk_id + 1
                tracker['chunk_progress'] = 0.0
                tracker['current_chunk_start'] = time.time()
                
                # Update overall progress
                if tracker['total_chunks'] > 0:
                    overall_progress = chunk_id / tracker['total_chunks']
                    tracker['overall_progress'] = min(overall_progress, 1.0)
                
                self.update_progress(session_id, 
                                   current_chunk=tracker['current_chunk'],
                                   overall_progress=tracker['overall_progress'],
                                   message=f"Completed chunk {chunk_id}")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to complete chunk: {e}")
            return False
    
    def complete_processing(self, session_id: str, success: bool = True):
        """Mark processing as completed"""
        try:
            if session_id in self.progress_trackers:
                tracker = self.progress_trackers[session_id]
                tracker['overall_progress'] = 1.0
                tracker['chunk_progress'] = 1.0
                tracker['status'] = 'completed' if success else 'error'
                
                message = "Video analysis completed successfully!" if success else "Video analysis failed"
                self.update_progress(session_id, status=tracker['status'], message=message)
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to complete processing: {e}")
            return False
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions (inactive for more than 1 hour)"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session_info in self.active_sessions.items():
                if current_time - session_info['last_activity'] > 3600:  # 1 hour
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.unregister_session(session_id)
                print(f"🧹 Cleaned up expired session: {session_id}")
                
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a session"""
        try:
            if session_id in self.progress_trackers:
                return self.progress_trackers[session_id].copy()
            return None
        except Exception as e:
            print(f"❌ Failed to get session status: {e}")
            return None
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        try:
            return len(self.active_sessions)
        except Exception as e:
            print(f"❌ Failed to get active sessions count: {e}")
            return 0
