"""
ROCm Bridge - Logging Configuration
===================================
Centralized logging setup for the entire application.
Ensures consistent log formatting and output destinations.

Production Features:
- Structured JSON logging option
- File and console handlers
- Log level configuration via environment
- Log rotation for long-running daemons
- Thread-safe logging (NO global context modification)

BUG FIXES APPLIED:
- Removed LogContext class (caused cross-thread contamination)
- Added proper handling for non-serializable objects in JSON formatter
- Added timezone-aware timestamps
"""

import logging
import sys
import json
from pathlib import Path
from typing import Optional, Any
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "structured",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure application-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        log_format: 'structured' for JSON, 'standard' for human-readable
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console Handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_path}")
    
    # Suppress noisy third-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized at level: {log_level}")

# ============================================================================
# REPLACE StructuredFormatter CLASS
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with recursion protection."""
    
    def _serialize_value(self, value: Any, depth: int = 0) -> Any:
        """Convert value to JSON-serializable format safely."""
        # FIX #8: Depth limit to prevent infinite recursion
        if depth > 3:
            return f"<Max Depth Reached: {type(value).__name__}>"
            
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v, depth + 1) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v, depth + 1) for k, v in value.items()}
        elif hasattr(value, '__dict__'):
            # Ignore internal python hidden variables
            return {
                k: self._serialize_value(v, depth + 1) 
                for k, v in value.__dict__.items() if not k.startswith('_')
            }
        else:
            return str(value)
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ["name", "msg", "args", "created", "filename",
                              "funcName", "levelname", "levelno", "lineno",
                              "module", "msecs", "pathname", "process",
                              "processName", "relativeCreated", "stack_info",
                              "exc_info", "exc_text", "thread", "threadName",
                              "message"]:
                    log_data[key] = self._serialize_value(value)
        
        try:
            return json.dumps(log_data)
        except (TypeError, ValueError) as e:
            return f"{{\"error\": \"JSON serialization failed: {e}\", \"message\": \"{record.getMessage()}\"}}"
        
# Thread-safe logging helper functions
def log_with_context(logger: logging.Logger, level: int, message: str, **context):
    """
    Log a message with additional context (thread-safe alternative to LogContext).
    
    Usage:
        log_with_context(logger, logging.INFO, "Processing file", file_path="kernel.cu")
    """
    logger.log(level, message, extra=context)