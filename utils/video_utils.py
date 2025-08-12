"""
Video Utilities Module
Handles video processing, metadata extraction, and screenshot capture
"""

import os
import cv2
import numpy as np
from PIL import Image
import io
import subprocess
from datetime import datetime

def extract_video_metadata(video_path):
    """Extract basic metadata from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
    except Exception as e:
        print(f"Error extracting video metadata: {e}")
        return None

def capture_screenshot(video_path, timestamp, session_id, upload_folder):
    """Capture a screenshot from video at specific timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return None
        
        # Calculate frame number from timestamp
        frame_number = int(timestamp * fps)
        
        # Set position to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large (max 800px width)
        if pil_image.width > 800:
            ratio = 800 / pil_image.width
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((800, new_height), Image.Resampling.LANCZOS)
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        # Generate filename
        timestamp_str = f"{timestamp:.2f}".replace('.', '_')
        filename = f"screenshot_{session_id}_{timestamp_str}.jpg"
        filepath = os.path.join(upload_folder, filename)
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(buffer.getvalue())
        
        return {
            'filename': filename,
            'url': f'/static/uploads/{filename}',
            'timestamp': timestamp,
            'width': pil_image.width,
            'height': pil_image.height
        }
        
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None

def extract_video_clip(video_path, start_time, end_time, session_id, upload_folder):
    """Extract a video clip from the video file"""
    try:
        # Generate output filename
        start_str = f"{start_time:.2f}".replace('.', '_')
        end_str = f"{end_time:.2f}".replace('.', '_')
        filename = f"clip_{session_id}_{start_str}_to_{end_str}.mp4"
        filepath = os.path.join(upload_folder, filename)
        
        # Use ffmpeg to extract the clip
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-y',  # Overwrite output file
            filepath
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(filepath):
            return {
                'filename': filename,
                'url': f'/static/uploads/{filename}',
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'type': 'video_clip'
            }
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error extracting video clip: {e}")
        return None

def create_evidence_for_timestamps(timestamps, video_path, session_id, upload_folder, video_metadata=None):
    """Create evidence (screenshots or video clips) based on timeframe length"""
    evidence = []
    
    # Use provided metadata or extract if not provided
    if video_metadata is None:
        video_metadata = extract_video_metadata(video_path)
    
    if not video_metadata:
        print("❌ Warning: Could not extract video metadata, skipping evidence creation")
        return evidence
    
    video_duration = video_metadata['duration']
    print(f"📹 Video duration: {video_duration:.2f} seconds")
    
    # Filter timestamps to only include those within video duration
    valid_timestamps = [ts for ts in timestamps if 0 <= ts < video_duration]
    
    if not valid_timestamps:
        print("❌ No valid timestamps found within video duration")
        return evidence
    
    print(f"✅ Valid timestamps: {valid_timestamps}")
    
    # Create evidence for each valid timestamp
    for i, timestamp in enumerate(valid_timestamps):
        try:
            # Determine if this is a single timestamp or a range
            if i < len(valid_timestamps) - 1:
                next_timestamp = valid_timestamps[i + 1]
                timeframe_length = next_timestamp - timestamp
            else:
                # For the last timestamp, use a short duration
                timeframe_length = 2.0  # 2 seconds for single timestamp
            
            # Ensure end_time doesn't exceed video duration
            end_time = min(timestamp + timeframe_length, video_duration)
            
            print(f"🔍 Processing timestamp {timestamp:.2f}s -> {end_time:.2f}s (duration: {timeframe_length:.2f}s)")
            
            # If timeframe is longer than 3 seconds, create a video clip
            if timeframe_length > 3.0:
                print(f"🎬 Creating video clip for {timestamp:.2f}s - {end_time:.2f}s")
                clip_data = extract_video_clip(video_path, timestamp, end_time, session_id, upload_folder)
                if clip_data:
                    evidence.append(clip_data)
                    print(f"✅ Video clip created: {clip_data['filename']}")
                else:
                    print(f"❌ Failed to create video clip for {timestamp:.2f}s - {end_time:.2f}s")
            else:
                # For shorter timeframes, create a screenshot
                print(f"📸 Creating screenshot for {timestamp:.2f}s")
                screenshot_data = capture_screenshot(video_path, timestamp, session_id, upload_folder)
                if screenshot_data:
                    screenshot_data['type'] = 'screenshot'
                    evidence.append(screenshot_data)
                    print(f"✅ Screenshot created: {screenshot_data['filename']}")
                else:
                    print(f"❌ Failed to create screenshot for {timestamp:.2f}s")
                    
        except Exception as e:
            print(f"❌ Error processing timestamp {timestamp:.2f}s: {e}")
            continue
    
    print(f"🎯 Evidence creation complete: {len(evidence)} items created")
    return evidence 