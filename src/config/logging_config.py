
# =============================================================================
# File: src/config/logging_config.py  
# =============================================================================

import logging
import logging.config
from pathlib import Path
import os

def setup_logging(log_level: str = "INFO", log_dir: str = "logs", log_file: str = "training.log"):
    """
    Setup comprehensive logging configuration for the ML pipeline
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Full path to log file
    log_file_path = log_path / log_file
    
    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(log_file_path),
                'mode': 'a',
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': str(log_path / 'errors.log'),
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'matplotlib': {
                'level': 'WARNING'
            },
            'plotly': {
                'level': 'WARNING'
            },
            'urllib3': {
                'level': 'WARNING'
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Create logger and log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("🚀 M5 Demand Forecasting Pipeline - Logging Initialized")
    logger.info(f"📁 Log file: {log_file_path}")
    logger.info(f"📊 Log level: {log_level}")
    logger.info("=" * 60)
    
    return logger

if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging()
    
    logger.info("Testing logging configuration...")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("Logging configuration test completed!")