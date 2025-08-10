"""
Performance Monitoring Service for Round 2
Tracks latency, throughput, and GPU performance metrics
"""

import time
import statistics
from typing import Dict, List, Optional, Tuple
from collections import deque
from config import Config

class PerformanceMonitor:
    """Performance monitoring and optimization service"""
    
    def __init__(self):
        # Latency tracking
        self.analysis_latencies = deque(maxlen=100)
        self.chat_latencies = deque(maxlen=100)
        self.frame_processing_latencies = deque(maxlen=100)
        
        # Throughput tracking
        self.fps_measurements = deque(maxlen=100)
        self.video_durations = deque(maxlen=50)
        
        # Performance targets
        self.latency_target = Config.PERFORMANCE_TARGETS['latency_target']
        self.fps_target = Config.PERFORMANCE_TARGETS['fps_target']
        
        # Performance history
        self.performance_history = []
        self.optimization_suggestions = []
        
    def record_analysis_latency(self, latency_ms: float):
        """Record video analysis latency"""
        self.analysis_latencies.append(latency_ms)
        self._check_performance_targets('analysis', latency_ms)
        
    def record_chat_latency(self, latency_ms: float):
        """Record chat response latency"""
        self.chat_latencies.append(latency_ms)
        self._check_performance_targets('chat', latency_ms)
        
    def record_frame_processing_latency(self, latency_ms: float):
        """Record frame processing latency"""
        self.frame_processing_latencies.append(latency_ms)
        self._check_performance_targets('frame_processing', latency_ms)
        
    def record_fps(self, fps: float):
        """Record video processing FPS"""
        self.fps_measurements.append(fps)
        
    def record_video_duration(self, duration_seconds: float):
        """Record video duration for analysis"""
        self.video_durations.append(duration_seconds)
        
    def _check_performance_targets(self, metric_type: str, value: float):
        """Check if performance meets targets and suggest optimizations"""
        if metric_type == 'analysis' and value > self.latency_target:
            self._add_optimization_suggestion(
                'analysis_latency',
                f"Analysis latency ({value:.2f}ms) exceeds target ({self.latency_target}ms). "
                "Consider: GPU memory optimization, batch size tuning, model quantization."
            )
        elif metric_type == 'chat' and value > 500:  # Chat target: 500ms
            self._add_optimization_suggestion(
                'chat_latency',
                f"Chat latency ({value:.2f}ms) exceeds target (500ms). "
                "Consider: Context length reduction, model optimization."
            )
        elif metric_type == 'frame_processing' and value > 11:  # 90fps = 11ms per frame
            self._add_optimization_suggestion(
                'frame_processing',
                f"Frame processing latency ({value:.2f}ms) exceeds target (11ms for 90fps). "
                "Consider: DeepStream optimization, GPU memory allocation, batch processing."
            )
    
    def _add_optimization_suggestion(self, category: str, suggestion: str):
        """Add optimization suggestion"""
        self.optimization_suggestions.append({
            'timestamp': time.time(),
            'category': category,
            'suggestion': suggestion
        })
        
        # Keep only recent suggestions
        if len(self.optimization_suggestions) > 20:
            self.optimization_suggestions = self.optimization_suggestions[-20:]
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        summary = {
            'latency_metrics': {
                'analysis': self._get_latency_stats(self.analysis_latencies, 'analysis'),
                'chat': self._get_latency_stats(self.chat_latencies, 'chat'),
                'frame_processing': self._get_latency_stats(self.frame_processing_latencies, 'frame_processing')
            },
            'throughput_metrics': {
                'fps': self._get_fps_stats(),
                'video_duration': self._get_duration_stats()
            },
            'performance_targets': {
                'latency_target_ms': self.latency_target,
                'fps_target': self.fps_target,
                'max_video_duration_minutes': Config.PERFORMANCE_TARGETS['max_video_duration'] / 60
            },
            'optimization_suggestions': self.optimization_suggestions[-5:],  # Last 5 suggestions
            'overall_performance_score': self._calculate_performance_score()
        }
        
        return summary
    
    def _get_latency_stats(self, latencies: deque, metric_name: str) -> Dict:
        """Get latency statistics for a specific metric"""
        if not latencies:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'target_met_percent': 0,
                'status': 'no_data'
            }
        
        values = list(latencies)
        mean = statistics.mean(values)
        median = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        
        # Calculate target compliance
        if metric_name == 'analysis':
            target = self.latency_target
        elif metric_name == 'chat':
            target = 500
        else:  # frame_processing
            target = 11
            
        target_met = sum(1 for v in values if v <= target)
        target_met_percent = (target_met / len(values)) * 100
        
        # Determine status
        if target_met_percent >= 95:
            status = 'excellent'
        elif target_met_percent >= 80:
            status = 'good'
        elif target_met_percent >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'count': len(values),
            'mean': round(mean, 2),
            'median': round(median, 2),
            'min': round(min_val, 2),
            'max': round(max_val, 2),
            'target_met_percent': round(target_met_percent, 1),
            'status': status
        }
    
    def _get_fps_stats(self) -> Dict:
        """Get FPS statistics"""
        if not self.fps_measurements:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'target_met_percent': 0
            }
        
        values = list(self.fps_measurements)
        mean = statistics.mean(values)
        median = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        
        # Calculate target compliance (90fps)
        target_met = sum(1 for v in values if v >= self.fps_target)
        target_met_percent = (target_met / len(values)) * 100
        
        return {
            'count': len(values),
            'mean': round(mean, 1),
            'median': round(median, 1),
            'min': round(min_val, 1),
            'max': round(max_val, 1),
            'target_met_percent': round(target_met_percent, 1)
        }
    
    def _get_duration_stats(self) -> Dict:
        """Get video duration statistics"""
        if not self.video_durations:
            return {
                'count': 0,
                'mean_minutes': 0,
                'max_minutes': 0,
                'target_support_percent': 0
            }
        
        values = list(self.video_durations)
        mean_minutes = statistics.mean(values) / 60
        max_minutes = max(values) / 60
        
        # Calculate target support (120 minutes)
        target_support = sum(1 for v in values if v <= Config.PERFORMANCE_TARGETS['max_video_duration'])
        target_support_percent = (target_support / len(values)) * 100
        
        return {
            'count': len(values),
            'mean_minutes': round(mean_minutes, 1),
            'max_minutes': round(max_minutes, 1),
            'target_support_percent': round(target_support_percent, 1)
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if not self.analysis_latencies and not self.chat_latencies:
            return 0.0
        
        scores = []
        
        # Analysis latency score (40% weight)
        if self.analysis_latencies:
            analysis_stats = self._get_latency_stats(self.analysis_latencies, 'analysis')
            scores.append(analysis_stats['target_met_percent'] * 0.4)
        
        # Chat latency score (30% weight)
        if self.chat_latencies:
            chat_stats = self._get_latency_stats(self.chat_latencies, 'chat')
            scores.append(chat_stats['target_met_percent'] * 0.3)
        
        # FPS score (20% weight)
        if self.fps_measurements:
            fps_stats = self._get_fps_stats()
            scores.append(fps_stats['target_met_percent'] * 0.2)
        
        # Frame processing score (10% weight)
        if self.frame_processing_latencies:
            frame_stats = self._get_latency_stats(self.frame_processing_latencies, 'frame_processing')
            scores.append(frame_stats['target_met_percent'] * 0.1)
        
        return round(sum(scores), 1) if scores else 0.0
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get prioritized optimization recommendations"""
        recommendations = []
        
        # Check analysis latency
        if self.analysis_latencies:
            analysis_stats = self._get_latency_stats(self.analysis_latencies, 'analysis')
            if analysis_stats['status'] in ['fair', 'poor']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'analysis_latency',
                    'issue': f"Analysis latency is {analysis_stats['status']} (target met: {analysis_stats['target_met_percent']}%)",
                    'suggestions': [
                        "Optimize GPU memory allocation",
                        "Enable model quantization (INT8)",
                        "Increase batch size for parallel processing",
                        "Use Flash Attention 2 for faster inference"
                    ]
                })
        
        # Check FPS performance
        if self.fps_measurements:
            fps_stats = self._get_fps_stats()
            if fps_stats['target_met_percent'] < 80:
                recommendations.append({
                    'priority': 'high',
                    'category': 'fps_performance',
                    'issue': f"FPS performance below target (target met: {fps_stats['target_met_percent']}%)",
                    'suggestions': [
                        "Optimize DeepStream pipeline",
                        "Reduce frame processing overhead",
                        "Enable GPU memory pooling",
                        "Use TensorRT optimization"
                    ]
                })
        
        # Check memory usage
        if self.analysis_latencies:
            recent_latencies = list(self.analysis_latencies)[-10:]
            if len(recent_latencies) >= 5:
                recent_avg = statistics.mean(recent_latencies)
                if recent_avg > self.latency_target * 1.2:  # 20% above target
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'memory_optimization',
                        'issue': f"Recent performance degradation detected (avg: {recent_avg:.2f}ms)",
                        'suggestions': [
                            "Clear GPU memory cache",
                            "Restart GPU service",
                            "Check for memory leaks",
                            "Optimize model loading"
                        ]
                    })
        
        return recommendations
    
    def export_performance_report(self) -> str:
        """Export performance report as formatted text"""
        summary = self.get_performance_summary()
        recommendations = self.get_optimization_recommendations()
        
        report = f"""
# AI Video Detective - Performance Report (Round 2)

## Performance Summary
- **Overall Score**: {summary['overall_performance_score']}/100
- **Analysis Latency**: {summary['latency_metrics']['analysis']['status'].upper()} ({summary['latency_metrics']['analysis']['target_met_percent']}% target compliance)
- **Chat Latency**: {summary['latency_metrics']['chat']['status'].upper()} ({summary['latency_metrics']['chat']['target_met_percent']}% target compliance)
- **FPS Performance**: {summary['throughput_metrics']['fps']['target_met_percent']}% target compliance

## Detailed Metrics

### Latency Performance
- **Video Analysis**: {summary['latency_metrics']['analysis']['mean']}ms avg (target: {self.latency_target}ms)
- **Chat Response**: {summary['latency_metrics']['chat']['mean']}ms avg (target: 500ms)
- **Frame Processing**: {summary['latency_metrics']['frame_processing']['mean']}ms avg (target: 11ms for 90fps)

### Throughput Performance
- **Video Processing**: {summary['throughput_metrics']['fps']['mean']} FPS avg (target: {self.fps_target} FPS)
- **Video Duration Support**: {summary['throughput_metrics']['video_duration']['target_support_percent']}% of videos within target duration

## Optimization Recommendations
"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"""
### {i}. {rec['category'].replace('_', ' ').title()} ({rec['priority'].upper()} priority)
**Issue**: {rec['issue']}

**Suggestions**:
"""
            for suggestion in rec['suggestions']:
                report += f"- {suggestion}\n"
        
        report += f"""
## Performance Targets
- **Latency Target**: <{self.latency_target}ms
- **FPS Target**: {self.fps_target} FPS
- **Video Duration**: Up to {Config.PERFORMANCE_TARGETS['max_video_duration'] / 60:.0f} minutes
- **Concurrent Sessions**: {Config.PERFORMANCE_TARGETS['concurrent_sessions']}

---
*Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.analysis_latencies.clear()
        self.chat_latencies.clear()
        self.frame_processing_latencies.clear()
        self.fps_measurements.clear()
        self.video_durations.clear()
        self.performance_history.clear()
        self.optimization_suggestions.clear()
        print("✅ Performance metrics reset") 