# Logging configuration
# File: src/config/logging_config.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
