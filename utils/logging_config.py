"""
Logging Configuration and Utilities

Provides structured logging with different levels and monitoring utilities
for system health across the ManaKnight AI E-Commerce Recommendation System.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from config import get_config


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ApplicationLogger:
    """Centralized logging manager for the application."""
    
    def __init__(self, config=None):
        """Initialize the logging system."""
        self.config = config or get_config()
        self._loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up the logging configuration."""
        # Create logs directory
        log_dir = Path(self.config.logging.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(self.config.logging.format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.logging.file_path,
            maxBytes=self.config.logging.max_file_size,
            backupCount=self.config.logging.backup_count
        )
        file_handler.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        # Use structured formatter for file logs
        if self.config.logging.level.upper() == 'DEBUG':
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(self.config.logging.format)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        logger = self.get_logger('performance')
        extra_fields = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_metric': True,
            **kwargs
        }
        
        # Add extra fields to the log record
        logger.info(
            f"Performance: {operation} completed in {duration:.3f}s",
            extra={'extra_fields': extra_fields}
        )
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       duration: float, user_agent: Optional[str] = None):
        """Log API request details."""
        logger = self.get_logger('api')
        extra_fields = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration_seconds': duration,
            'user_agent': user_agent,
            'api_request': True
        }
        
        logger.info(
            f"API {method} {endpoint} - {status_code} ({duration:.3f}s)",
            extra={'extra_fields': extra_fields}
        )
    
    def log_model_prediction(self, model_name: str, input_type: str, 
                           confidence: float, prediction: str, **kwargs):
        """Log model prediction details."""
        logger = self.get_logger('ml_models')
        extra_fields = {
            'model_name': model_name,
            'input_type': input_type,
            'confidence': confidence,
            'prediction': prediction,
            'model_prediction': True,
            **kwargs
        }
        
        logger.info(
            f"Model {model_name}: {prediction} (confidence: {confidence:.3f})",
            extra={'extra_fields': extra_fields}
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        logger = self.get_logger('errors')
        extra_fields = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'error_log': True
        }
        
        logger.error(
            f"Error: {type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={'extra_fields': extra_fields}
        )


# Global logger instance
app_logger = ApplicationLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return app_logger.get_logger(name)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    app_logger.log_performance(operation, duration, **kwargs)


def log_api_request(endpoint: str, method: str, status_code: int, 
                   duration: float, user_agent: Optional[str] = None):
    """Log API request details."""
    app_logger.log_api_request(endpoint, method, status_code, duration, user_agent)


def log_model_prediction(model_name: str, input_type: str, 
                        confidence: float, prediction: str, **kwargs):
    """Log model prediction details."""
    app_logger.log_model_prediction(model_name, input_type, confidence, prediction, **kwargs)


def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """Log error with additional context."""
    app_logger.log_error_with_context(error, context)


# Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                log_performance(operation_name, duration, success=False, error=str(e))
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test logging
    logger = get_logger('test')
    logger.info("Logging system initialized successfully")
    
    # Test performance logging
    log_performance('test_operation', 0.123, test_param='value')
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error_with_context(e, {'test_context': 'test_value'})
