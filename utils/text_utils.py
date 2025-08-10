"""
Text Utilities Module
Handles timestamp extraction and text processing functionality
"""

import re

def aggressive_timestamp_validation(timestamps, video_duration):
    """Aggressively validate timestamps and remove any that are out of bounds"""
    if not timestamps or video_duration <= 0:
        return []
    
    # Convert to seconds if needed and filter
    valid_timestamps = []
    for ts in timestamps:
        # Ensure timestamp is within bounds - use actual video duration
        if 0 <= ts < video_duration:
            valid_timestamps.append(ts)
        else:
            print(f"🚫 Removed out-of-bounds timestamp: {ts:.2f}s (video duration: {video_duration:.2f}s)")
    
    # Remove duplicates and sort
    unique_timestamps = sorted(list(set(valid_timestamps)))
    
    # Ensure minimum spacing between timestamps (at least 2 seconds)
    spaced_timestamps = []
    for ts in unique_timestamps:
        if not spaced_timestamps or abs(ts - spaced_timestamps[-1]) >= 2.0:
            spaced_timestamps.append(ts)
        else:
            print(f"🚫 Removed too-close timestamp: {ts:.2f}s (too close to {spaced_timestamps[-1]:.2f}s)")
    
    print(f"✅ Timestamp validation: {len(timestamps)} -> {len(spaced_timestamps)} valid timestamps")
    return spaced_timestamps

def clean_and_deduplicate_timestamps(timestamps):
    """Clean and deduplicate timestamps, removing duplicates and sorting"""
    if not timestamps:
        return []
    
    # Remove duplicates and sort
    unique_timestamps = sorted(list(set(timestamps)))
    
    # Remove any invalid timestamps (negative) - don't limit upper bound here
    valid_timestamps = [ts for ts in unique_timestamps if ts >= 0]
    
    # Remove timestamps that are too close together (within 1 second)
    cleaned_timestamps = []
    for ts in valid_timestamps:
        if not cleaned_timestamps or abs(ts - cleaned_timestamps[-1]) >= 1.0:
            cleaned_timestamps.append(ts)
    
    return cleaned_timestamps

def extract_timestamps_from_text(text):
    """Extract timestamps from text using regex patterns"""
    # Pattern for timestamps like 00:15, 1:30, 00:15-00:17, etc.
    timestamp_patterns = [
        r'(\d{1,2}):(\d{2})(?::(\d{2}))?',  # MM:SS or HH:MM:SS
        r'(\d{1,2})\.(\d{2})',  # MM.SS format
        r'(\d+)s',  # seconds format
        r'(\d+)\.(\d+)s'  # decimal seconds
    ]
    
    timestamps = []
    
    for pattern in timestamp_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if ':' in match.group():
                # Handle MM:SS or HH:MM:SS format
                parts = match.group().split(':')
                if len(parts) == 2:
                    minutes, seconds = int(parts[0]), int(parts[1])
                    timestamp = minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                    timestamp = hours * 3600 + minutes * 60 + seconds
            elif '.' in match.group() and 's' not in match.group():
                # Handle MM.SS format
                parts = match.group().split('.')
                minutes, seconds = int(parts[0]), int(parts[1])
                timestamp = minutes * 60 + seconds
            elif 's' in match.group():
                # Handle seconds format
                timestamp = float(match.group().replace('s', ''))
            else:
                continue
                
            # Allow any positive timestamp - validation will be done later with actual video duration
            if timestamp >= 0:
                timestamps.append(timestamp)
    
    # Remove duplicates and sort
    timestamps = sorted(list(set(timestamps)))
    
    # Prioritize timestamps that are mentioned in key events
    # Look for specific patterns that indicate important moments
    key_events = []
    for match in re.finditer(r'(\d{1,2}):(\d{2})', text):
        minutes, seconds = int(match.group(1)), int(match.group(2))
        timestamp = minutes * 60 + seconds
        if timestamp >= 0:  # Allow any positive timestamp
            key_events.append(timestamp)
    
    # Combine and prioritize key events
    all_timestamps = list(set(timestamps + key_events))
    
    # Clean and deduplicate
    cleaned_timestamps = clean_and_deduplicate_timestamps(all_timestamps)
    
    # Return up to 12 most relevant timestamps (increased from 8)
    return cleaned_timestamps[:12]

def extract_timestamp_ranges_from_text(text):
    """Extract timestamp ranges from text (e.g., '00:15-00:17')"""
    ranges = []
    # Pattern for timestamp ranges like 00:15-00:17, 1:30-1:45, etc.
    range_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?'
    
    for match in re.finditer(range_pattern, text):
        # Parse start time
        start_hours = int(match.group(3)) if match.group(3) else 0
        start_minutes = int(match.group(1))
        start_seconds = int(match.group(2))
        start_time = start_hours * 3600 + start_minutes * 60 + start_seconds
        
        # Parse end time
        end_hours = int(match.group(6)) if match.group(6) else 0
        end_minutes = int(match.group(4))
        end_seconds = int(match.group(5))
        end_time = end_hours * 3600 + end_minutes * 60 + end_seconds
        
        # Validate logical order and reasonable range - allow longer ranges for longer videos
        if start_time >= 0 and end_time > start_time and (end_time - start_time) <= 600:  # Max 10 minute range
            ranges.append((start_time, end_time))
    
    return ranges 