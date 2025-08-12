"""
Performance Monitoring Service
Optimized for speed - only essential functions kept
"""

import time
from collections import deque
from typing import Dict, List, Deque
from config import Config

class PerformanceMonitor:
    """Simplified performance monitor - only essential functions kept for speed"""
    
    def __init__(self):
        # Only keep what's actually used
        self.analysis_latencies: Deque[float] = deque(maxlen=100)
        self.start_time = time.time()
        
    def record_analysis_latency(self, latency_ms: float):
        """Record analysis latency - only essential function kept"""
        self.analysis_latencies.append(latency_ms)
    
    def get_performance_summary(self) -> Dict:
        """Get basic performance summary - simplified for speed"""
        if not self.analysis_latencies:
            return {
                'total_analyses': 0,
                'average_latency': 0,
                'uptime': time.time() - self.start_time
            }
        
        avg_latency = sum(self.analysis_latencies) / len(self.analysis_latencies)
        return {
            'total_analyses': len(self.analysis_latencies),
            'average_latency': round(avg_latency, 2),
            'uptime': round(time.time() - self.start_time, 2)
        }
    
    def start(self, interval: float = 2.0):
        """Start monitoring - simplified"""
        print(f"🚀 Performance monitoring started (simplified mode)")
    
    def start_monitoring(self, *args, **kwargs):
        """Start monitoring - compatibility method"""
        self.start()

# Create global instance
performance_monitor = PerformanceMonitor() 