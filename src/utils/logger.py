"""
Enhanced logging utilities with timestamps and performance tracking
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any
from colorama import Fore, Style, init

# Initialize colorama
init()

class TimestampedLogger:
    """Logger with timestamps and performance tracking"""
    
    def __init__(self, enable_timestamps: bool = True, enable_colors: bool = True):
        self.enable_timestamps = enable_timestamps
        self.enable_colors = enable_colors
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Color mapping for different log types
        self.colors = {
            "info": Fore.BLUE,
            "success": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "speech": Fore.MAGENTA,
            "debug": Fore.CYAN,
            "vad": Fore.LIGHTBLUE_EX,
            "synthesis": Fore.LIGHTGREEN_EX,
            "interrupt": Fore.LIGHTYELLOW_EX,
            "streaming": Fore.LIGHTCYAN_EX
        }
        
    def get_timestamp(self) -> str:
        """Get formatted timestamp with elapsed time"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        delta = current_time - self.last_log_time
        self.last_log_time = current_time
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        return f"[{timestamp} | +{elapsed:6.2f}s | Δ{delta:5.3f}s]"
        
    def log(self, message: str, log_type: str = "info", prefix: str = None):
        """Log a message with timestamp and color"""
        parts = []
        
        # Add timestamp
        if self.enable_timestamps:
            parts.append(self.get_timestamp())
            
        # Add custom prefix or default type prefix
        if prefix:
            parts.append(f"[{prefix}]")
        else:
            parts.append(f"[{log_type.upper()}]")
            
        # Add message
        parts.append(message)
        
        # Build final message
        full_message = " ".join(parts)
        
        # Apply color if enabled
        if self.enable_colors and log_type in self.colors:
            full_message = f"{self.colors[log_type]}{full_message}{Style.RESET_ALL}"
            
        print(full_message)
        
    def info(self, message: str, prefix: str = None):
        """Log info message"""
        self.log(message, "info", prefix)
        
    def success(self, message: str, prefix: str = None):
        """Log success message"""
        self.log(message, "success", prefix)
        
    def warning(self, message: str, prefix: str = None):
        """Log warning message"""
        self.log(message, "warning", prefix)
        
    def error(self, message: str, prefix: str = None):
        """Log error message"""
        self.log(message, "error", prefix)
        
    def debug(self, message: str, prefix: str = None):
        """Log debug message"""
        self.log(message, "debug", prefix)
        
    def vad(self, message: str):
        """Log VAD-related message"""
        self.log(message, "vad", "VAD")
        
    def synthesis(self, message: str):
        """Log synthesis-related message"""
        self.log(message, "synthesis", "TTS")
        
    def interrupt(self, message: str):
        """Log interrupt-related message"""
        self.log(message, "interrupt", "INTERRUPT")
        
    def streaming(self, message: str):
        """Log streaming-related message"""
        self.log(message, "streaming", "STREAM")
        
    def reset_timer(self):
        """Reset the start time for elapsed time tracking"""
        self.start_time = time.time()
        self.last_log_time = self.start_time


# Global logger instance with default settings
logger = TimestampedLogger(enable_timestamps=True, enable_colors=True)


def log_performance(operation_name: str):
    """Decorator to log performance of operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {operation_name}...")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.success(f"{operation_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{operation_name} failed after {elapsed:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator