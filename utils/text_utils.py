"""
Text Utilities Module
Handles timestamp extraction and text processing functionality
"""

import re

def validate_and_fix_timestamps(timestamps, video_duration):
    """Validate and fix timestamps to ensure they're within video duration"""
    if not timestamps or video_duration <= 0:
        return []
    
    fixed_timestamps = []
    for ts in timestamps:
        if 0 <= ts < video_duration:
            fixed_timestamps.append(ts)
        else:
            # If timestamp is out of bounds, try to fix it
            if ts >= video_duration:
                # Scale down the timestamp proportionally
                scaled_ts = (ts / (ts + 1)) * video_duration * 0.9  # Use 90% of video duration
                if 0 <= scaled_ts < video_duration:
                    print(f"⚠️ Fixed out-of-bounds timestamp {ts:.2f}s -> {scaled_ts:.2f}s")
                    fixed_timestamps.append(scaled_ts)
            elif ts < 0:
                # If negative, use a small positive value
                fixed_ts = min(1.0, video_duration * 0.1)
                print(f"⚠️ Fixed negative timestamp {ts:.2f}s -> {fixed_ts:.2f}s")
                fixed_timestamps.append(fixed_ts)
    
    return sorted(list(set(fixed_timestamps)))

def clean_and_deduplicate_timestamps(timestamps):
    """Clean and deduplicate timestamps, removing duplicates and sorting"""
    if not timestamps:
        return []
    
    # Remove duplicates and sort
    unique_timestamps = sorted(list(set(timestamps)))
    
    # Remove any invalid timestamps (negative or too large)
    valid_timestamps = [ts for ts in unique_timestamps if 0 <= ts <= 1800]
    
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
                
            # More restrictive range: 0 to 30 minutes (1800 seconds) for typical videos
            if 0 <= timestamp <= 1800:
                timestamps.append(timestamp)
    
    # Remove duplicates and sort
    timestamps = sorted(list(set(timestamps)))
    
    # Prioritize timestamps that are mentioned in key events
    # Look for specific patterns that indicate important moments
    key_events = []
    for match in re.finditer(r'(\d{1,2}):(\d{2})', text):
        minutes, seconds = int(match.group(1)), int(match.group(2))
        timestamp = minutes * 60 + seconds
        if 0 <= timestamp <= 1800:  # Same range restriction
            key_events.append(timestamp)
    
    # Combine and prioritize key events
    all_timestamps = list(set(timestamps + key_events))
    
    # Clean and deduplicate
    cleaned_timestamps = clean_and_deduplicate_timestamps(all_timestamps)
    
    # Return up to 8 most relevant timestamps
    return cleaned_timestamps[:8]

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
        
        # More restrictive validation: 0 to 30 minutes (1800 seconds) and ensure logical order
        if 0 <= start_time <= end_time <= 1800 and (end_time - start_time) <= 300:  # Max 5 minute range
            ranges.append((start_time, end_time))
    
    return ranges 