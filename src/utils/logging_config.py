"""
Comprehensive logging configuration for the dashboard application.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
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
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
    
    def start_operation(self, operation_id: str, operation_type: str, **kwargs):
        """Start timing an operation."""
        self.metrics[operation_id] = {
            'start_time': datetime.utcnow(),
            'operation_type': operation_type,
            **kwargs
        }
    
    def end_operation(self, operation_id: str, success: bool = True, **kwargs):
        """End timing an operation and log the result."""
        if operation_id not in self.metrics:
            self.logger.warning(f"Operation {operation_id} not found in metrics")
            return
        
        start_data = self.metrics.pop(operation_id)
        duration = (datetime.utcnow() - start_data['start_time']).total_seconds()
        
        self.logger.info(
            f"Operation completed",
            extra={
                'operation_id': operation_id,
                'operation_type': start_data['operation_type'],
                'duration_seconds': duration,
                'success': success,
                **start_data,
                **kwargs
            }
        )


class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self, logger_name: str = 'security'):
        self.logger = logging.getLogger(logger_name)
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = None):
        """Log login attempts."""
        self.logger.info(
            f"Login attempt",
            extra={
                'event_type': 'login_attempt',
                'username': username,
                'success': success,
                'ip_address': ip_address
            }
        )
    
    def log_access_denied(self, username: str, resource: str, ip_address: str = None):
        """Log access denied events."""
        self.logger.warning(
            f"Access denied",
            extra={
                'event_type': 'access_denied',
                'username': username,
                'resource': resource,
                'ip_address': ip_address
            }
        )
    
    def log_suspicious_activity(self, description: str, username: str = None, ip_address: str = None):
        """Log suspicious activities."""
        self.logger.warning(
            f"Suspicious activity: {description}",
            extra={
                'event_type': 'suspicious_activity',
                'description': description,
                'username': username,
                'ip_address': ip_address
            }
        )


def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = 'logs',
    enable_json_logging: bool = True,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json_logging: Whether to use JSON formatting
        enable_file_logging: Whether to log to files
        enable_console_logging: Whether to log to console
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    if enable_file_logging:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Set up formatters
    if enable_json_logging:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handlers = []
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handlers
    if enable_file_logging:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'dashboard.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setFormatter(formatter)
        handlers.append(app_handler)
        
        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'errors.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        handlers.append(error_handler)
        
        # Performance log
        perf_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'performance.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        perf_handler.setFormatter(formatter)
        
        # Security log
        security_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'security.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        security_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Configure specific loggers
    loggers = {}
    
    # Application logger
    app_logger = logging.getLogger('dashboard')
    loggers['app'] = app_logger
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    if enable_file_logging:
        perf_logger.addHandler(perf_handler)
    loggers['performance'] = PerformanceLogger('performance')
    
    # Security logger
    security_logger = logging.getLogger('security')
    if enable_file_logging:
        security_logger.addHandler(security_handler)
    loggers['security'] = SecurityLogger('security')
    
    # Database logger
    db_logger = logging.getLogger('database')
    loggers['database'] = db_logger
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Global logger instances
loggers = setup_logging(
    log_level=os.environ.get('LOG_LEVEL', 'INFO'),
    log_dir=os.environ.get('LOG_DIR', 'logs'),
    enable_json_logging=os.environ.get('JSON_LOGGING', 'true').lower() == 'true'
)

# Export commonly used loggers
app_logger = loggers['app']
performance_logger = loggers['performance']
security_logger = loggers['security']
database_logger = loggers['database']