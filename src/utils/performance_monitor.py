"""
Performance monitoring and clean logging for voice assistant
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.logger import logger

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def stop(self) -> float:
        """Stop timing and return duration"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration

class PerformanceMonitor:
    """Monitor and log performance metrics cleanly"""
    
    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self.metrics: List[PerformanceMetric] = []
        self.active_timers: Dict[str, PerformanceMetric] = {}
        self.session_start = time.time()
        
    def start_timer(self, name: str, metadata: Dict = None) -> str:
        """Start a performance timer"""
        metric = PerformanceMetric(
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.active_timers[name] = metric
        
        if self.enable_detailed_logging:
            logger.debug(f"⏱️  Started: {name}")
            
        return name
        
    def stop_timer(self, name: str, log_result: bool = True) -> Optional[float]:
        """Stop a timer and optionally log the result"""
        if name not in self.active_timers:
            logger.warning(f"Timer '{name}' not found")
            return None
            
        metric = self.active_timers.pop(name)
        duration = metric.stop()
        self.metrics.append(metric)
        
        if log_result:
            self._log_performance(metric)
            
        return duration
        
    def _log_performance(self, metric: PerformanceMetric):
        """Log performance metric with appropriate detail level"""
        duration = metric.duration
        name = metric.name
        
        # Categorize performance levels
        if duration < 0.1:
            level = "debug"  # Very fast
        elif duration < 0.5:
            level = "info"   # Normal
        elif duration < 2.0:
            level = "warning"  # Slow
        else:
            level = "error"   # Very slow
            
        # Format duration nicely
        if duration < 1.0:
            duration_str = f"{duration*1000:.0f}ms"
        else:
            duration_str = f"{duration:.2f}s"
            
        # Log with appropriate level
        message = f"⏱️  {name}: {duration_str}"
        
        if level == "debug" and self.enable_detailed_logging:
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
            
    def get_session_summary(self) -> Dict:
        """Get performance summary for the session"""
        total_session_time = time.time() - self.session_start
        
        # Group metrics by category
        categories = {}
        for metric in self.metrics:
            category = metric.name.split('_')[0]  # First word as category
            if category not in categories:
                categories[category] = []
            categories[category].append(metric.duration)
            
        # Calculate averages
        summary = {
            'session_duration': total_session_time,
            'total_operations': len(self.metrics),
            'categories': {}
        }
        
        for category, durations in categories.items():
            summary['categories'][category] = {
                'count': len(durations),
                'total': sum(durations),
                'average': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations)
            }
            
        return summary
        
    def log_session_summary(self):
        """Log a clean session summary"""
        summary = self.get_session_summary()
        
        logger.info("📊 Performance Summary:")
        logger.info(f"   Session: {summary['session_duration']:.1f}s, Operations: {summary['total_operations']}")
        
        for category, stats in summary['categories'].items():
            avg = stats['average']
            if avg < 1.0:
                avg_str = f"{avg*1000:.0f}ms"
            else:
                avg_str = f"{avg:.2f}s"
                
            logger.info(f"   {category.title()}: {stats['count']} ops, avg {avg_str}")

# Global performance monitor
perf_monitor = PerformanceMonitor(enable_detailed_logging=False)

def time_operation(name: str, metadata: Dict = None):
    """Decorator to time operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = f"{name}_{func.__name__}"
            perf_monitor.start_timer(timer_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                perf_monitor.stop_timer(timer_name)
                return result
            except Exception as e:
                perf_monitor.stop_timer(timer_name, log_result=False)
                logger.error(f"❌ {timer_name} failed: {e}")
                raise
                
        return wrapper
    return decorator

class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, name: str, metadata: Dict = None, log_result: bool = True):
        self.name = name
        self.metadata = metadata
        self.log_result = log_result
        
    def __enter__(self):
        perf_monitor.start_timer(self.name, self.metadata)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        perf_monitor.stop_timer(self.name, self.log_result)
        if exc_type:
            logger.error(f"❌ {self.name} failed: {exc_val}")