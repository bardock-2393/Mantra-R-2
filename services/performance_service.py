"""
Enhanced Performance Service for GPU Optimization
Handles torch.compile, model quantization, and memory management for long video processing
"""

import os
import time
import torch
import psutil
import threading
from typing import Dict, List, Optional, Tuple
import gc

class PerformanceMonitor:
    """Enhanced performance monitoring and optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gpu_config = config.get('GPU_CONFIG', {})
        self.performance_targets = config.get('PERFORMANCE_TARGETS', {})
        self.monitoring_active = False
        self.monitor_thread = None
        self.performance_history = []
        self.max_history_size = 1000
        
        # Performance optimization settings
        self.torch_compile_mode = config.get('torch_compile_mode', 'max-autotune')
        self.quantization_enabled = config.get('quantization_enabled', True)
        self.memory_cleanup_threshold = config.get('memory_cleanup_threshold', 0.8)  # 80% GPU memory
        
        # Initialize monitoring
        self._init_monitoring()
    
    def _init_monitoring(self):
        """Initialize performance monitoring"""
        try:
            if self.gpu_config.get('enabled', False) and torch.cuda.is_available():
                self.monitoring_active = True
                self.start_monitoring()
                print("🚀 Performance monitoring initialized")
            else:
                print("⚠️ GPU monitoring disabled - CUDA not available")
        except Exception as e:
            print(f"❌ Failed to initialize performance monitoring: {e}")
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_metrics()
                    time.sleep(5)  # Collect metrics every 5 seconds
                except Exception as e:
                    print(f"⚠️ Performance monitoring error: {e}")
                    time.sleep(10)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Performance monitoring stopped")
    
    def _collect_metrics(self):
        """Collect current performance metrics"""
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_metrics': self._get_gpu_metrics()
            }
            
            self.performance_history.append(metrics)
            
            # Limit history size
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]
            
            # Check for memory cleanup threshold
            if metrics['gpu_metrics'].get('memory_percent', 0) > self.memory_cleanup_threshold * 100:
                self._trigger_memory_cleanup()
                
        except Exception as e:
            print(f"❌ Failed to collect metrics: {e}")
    
    def _get_gpu_metrics(self) -> Dict:
        """Get GPU performance metrics"""
        try:
            if not torch.cuda.is_available():
                return {'error': 'CUDA not available'}
            
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            
            return {
                'device': device,
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'total_gb': round(total, 2),
                'memory_percent': round((allocated / total) * 100, 1),
                'utilization': self._get_gpu_utilization()
            }
        except Exception as e:
            return {'error': f'GPU metrics failed: {str(e)}'}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            # This is a simplified approach - in production you might use nvidia-ml-py
            # For now, we'll estimate based on memory usage patterns
            if torch.cuda.is_available():
                # Simple heuristic: if memory is actively changing, GPU is likely busy
                return min(100.0, torch.cuda.memory_allocated() / 1024**3 * 10)  # Rough estimate
            return 0.0
        except:
            return 0.0
    
    def _trigger_memory_cleanup(self):
        """Trigger GPU memory cleanup when threshold is exceeded"""
        try:
            print("🧹 Memory cleanup threshold exceeded, cleaning up GPU memory...")
            self.cleanup_gpu_memory()
        except Exception as e:
            print(f"❌ Memory cleanup failed: {e}")
    
    def cleanup_gpu_memory(self, aggressive: bool = False):
        """Clean up GPU memory"""
        try:
            if torch.cuda.is_available():
                # Empty cache
                torch.cuda.empty_cache()
                
                # Synchronize
                torch.cuda.synchronize()
                
                # Garbage collection
                gc.collect()
                
                if aggressive:
                    # More aggressive cleanup
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                
                print("🧹 GPU memory cleaned up successfully")
                
                # Get memory after cleanup
                metrics = self._get_gpu_metrics()
                print(f"📊 Memory after cleanup: {metrics.get('allocated_gb', 0)}GB allocated")
                
        except Exception as e:
            print(f"❌ GPU memory cleanup failed: {e}")
    
    def optimize_model(self, model, model_name: str = "unknown") -> torch.nn.Module:
        """Apply performance optimizations to a model"""
        try:
            print(f"🚀 Optimizing model: {model_name}")
            
            # Enable torch.compile if available
            if hasattr(torch, 'compile') and self.torch_compile_mode:
                try:
                    print(f"🔧 Applying torch.compile with mode: {self.torch_compile_mode}")
                    model = torch.compile(model, mode=self.torch_compile_mode)
                    print("✅ torch.compile optimization applied")
                except Exception as e:
                    print(f"⚠️ torch.compile failed: {e}")
            
            # Apply quantization if enabled
            if self.quantization_enabled:
                try:
                    model = self._apply_quantization(model)
                    print("✅ Model quantization applied")
                except Exception as e:
                    print(f"⚠️ Quantization failed: {e}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                device = torch.device(self.gpu_config.get('device', 'cuda:0'))
                model = model.to(device)
                print(f"🚀 Model moved to GPU: {device}")
            
            return model
            
        except Exception as e:
            print(f"❌ Model optimization failed: {e}")
            return model
    
    def _apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model quantization for memory efficiency"""
        try:
            # Try dynamic quantization first (more compatible)
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
                print("✅ Dynamic quantization applied (INT8)")
                return quantized_model
            except Exception as e:
                print(f"⚠️ Dynamic quantization failed: {e}")
            
            # Fallback to FP16 if quantization fails
            try:
                if hasattr(model, 'half'):
                    model = model.half()
                    print("✅ FP16 precision applied")
                return model
            except Exception as e:
                print(f"⚠️ FP16 conversion failed: {e}")
            
            return model
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return model
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        try:
            if not self.performance_history:
                return {'error': 'No performance data available'}
            
            recent_metrics = self.performance_history[-10:]  # Last 10 measurements
            
            # Calculate averages
            cpu_avg = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
            memory_avg = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
            
            # Get latest GPU metrics
            latest_gpu = recent_metrics[-1]['gpu_metrics'] if recent_metrics else {}
            
            return {
                'cpu_percent_avg': round(cpu_avg, 1),
                'memory_percent_avg': round(memory_avg, 1),
                'gpu_metrics': latest_gpu,
                'monitoring_active': self.monitoring_active,
                'history_size': len(self.performance_history)
            }
            
        except Exception as e:
            return {'error': f'Performance summary failed: {str(e)}'}
    
    def check_performance_targets(self) -> Dict:
        """Check if performance targets are being met"""
        try:
            current_metrics = self.get_performance_summary()
            
            if 'error' in current_metrics:
                return {'error': current_metrics['error']}
            
            targets = self.performance_targets
            results = {}
            
            # Check CPU usage
            if 'cpu_target' in targets:
                cpu_target = targets['cpu_target']
                cpu_current = current_metrics['cpu_percent_avg']
                results['cpu'] = {
                    'target': cpu_target,
                    'current': cpu_current,
                    'meeting_target': cpu_current <= cpu_target
                }
            
            # Check memory usage
            if 'memory_target' in targets:
                memory_target = targets['memory_target']
                memory_current = current_metrics['memory_percent_avg']
                results['memory'] = {
                    'target': memory_target,
                    'current': memory_current,
                    'meeting_target': memory_current <= memory_target
                }
            
            # Check GPU memory
            gpu_metrics = current_metrics.get('gpu_metrics', {})
            if 'memory_percent' in gpu_metrics:
                gpu_memory_target = 80  # 80% GPU memory threshold
                gpu_memory_current = gpu_metrics['memory_percent']
                results['gpu_memory'] = {
                    'target': gpu_memory_target,
                    'current': gpu_memory_current,
                    'meeting_target': gpu_memory_current <= gpu_memory_target
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Performance target check failed: {str(e)}'}
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        try:
            recommendations = []
            performance_check = self.check_performance_targets()
            
            if 'error' in performance_check:
                return [f"Unable to generate recommendations: {performance_check['error']}"]
            
            # CPU recommendations
            if 'cpu' in performance_check:
                cpu_result = performance_check['cpu']
                if not cpu_result['meeting_target']:
                    recommendations.append(f"CPU usage ({cpu_result['current']}%) exceeds target ({cpu_result['target']}%). Consider reducing batch size or optimizing preprocessing.")
            
            # Memory recommendations
            if 'memory' in performance_check:
                memory_result = performance_check['memory']
                if not memory_result['meeting_target']:
                    recommendations.append(f"System memory usage ({memory_result['current']}%) exceeds target ({memory_result['target']}%). Consider closing other applications or reducing video resolution.")
            
            # GPU recommendations
            if 'gpu_memory' in performance_check:
                gpu_result = performance_check['gpu_memory']
                if not gpu_result['meeting_target']:
                    recommendations.append(f"GPU memory usage ({gpu_result['current']}%) exceeds target ({gpu_result['target']}%). Consider reducing frame count or enabling aggressive memory cleanup.")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Performance targets are being met. No immediate optimizations needed.")
            else:
                recommendations.append("Consider enabling torch.compile with 'max-autotune' mode for better performance.")
                recommendations.append("Enable model quantization (INT8/FP16) for memory efficiency.")
                recommendations.append("Use decord instead of OpenCV for faster video processing.")
            
            return recommendations
            
        except Exception as e:
            return [f"Failed to generate recommendations: {str(e)}"]
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except:
            pass 