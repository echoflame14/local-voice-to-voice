#!/usr/bin/env python3
"""
Test the timestamped logging system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import logger, log_performance
import time

def test_logging():
    """Test the logging functionality"""
    logger.info("Testing timestamped logging system")
    
    time.sleep(0.1)
    logger.debug("Debug message with timestamps")
    
    time.sleep(0.05)
    logger.success("Success message")
    
    time.sleep(0.03)
    logger.warning("Warning message")
    
    time.sleep(0.02)
    logger.error("Error message")
    
    time.sleep(0.01)
    logger.vad("VAD-specific message")
    logger.synthesis("Synthesis-specific message")
    logger.interrupt("Interrupt-specific message")
    logger.streaming("Streaming-specific message")
    
    # Test performance decorator
    @log_performance("Test Operation")
    def slow_operation():
        time.sleep(1.0)
        return "Operation complete"
    
    result = slow_operation()
    logger.info(f"Operation result: {result}")

if __name__ == "__main__":
    test_logging()