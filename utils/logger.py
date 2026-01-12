# ============================================================================
# LOGGER - Centralized Logging System
# ============================================================================
# This module provides structured logging with console and file output.
#
# USAGE:
#   from utils.logger import get_logger
#   
#   logger = get_logger(__name__)
#   logger.success("Configuration loaded")
#   logger.error("File not found")
#   logger.info("Starting optimization")
#
# FEATURES:
#   - Colored console output (using utils.colors)
#   - Optional file logging
#   - Multiple log levels (DEBUG, INFO, WARNING, ERROR, SUCCESS)
#   - Section headers and progress indicators
#
# Last updated: 2025-12-31
# ============================================================================

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from utils.colors import (
    Colors, print_colored, print_header, print_section,
    print_success, print_error, print_warning, print_info, print_debug
)


# ============================================================================
# CUSTOM LOG LEVELS
# ============================================================================

# Add custom SUCCESS log level (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


# ============================================================================
# LOGGER CLASS
# ============================================================================

class Logger:
    """
    Custom logger with colored console output and optional file logging.
    
    Args:
        name: Logger name (usually module name)
        log_file: Path to log file (optional, for file logging)
        console_level: Console log level (default: INFO)
        file_level: File log level (default: DEBUG)
        verbose: If True, show more detailed logs
    """
    
    def __init__(self,
                 name: str = "HPO",
                 log_file: Optional[str] = None,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 verbose: bool = True):
        
        self.name = name
        self.verbose = verbose
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Remove existing handlers (prevent duplicates)
        self.logger.handlers = []
        
        # Console handler (with colors, no formatting - we handle that)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)
        
        # File handler (optional, plain text with timestamps)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(file_level)
            
            # File format includes timestamp and level
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    # ========================================================================
    # LOG METHODS
    # ========================================================================
    
    def success(self, message: str, prefix: str = "[SUCCESS]"):
        """Log success message (green)."""
        print_success(message, prefix)
        self.logger.log(SUCCESS_LEVEL, message)
    
    def error(self, message: str, prefix: str = "[ERROR]"):
        """Log error message (red, bold)."""
        print_error(message, prefix)
        self.logger.error(message)
    
    def warning(self, message: str, prefix: str = "[WARNING]"):
        """Log warning message (yellow)."""
        print_warning(message, prefix)
        self.logger.warning(message)
    
    def info(self, message: str, prefix: str = "[INFO]"):

        """Log info message (cyan)."""
        formatted = f"[INFO] {message}"
        if self.verbose:
            print(formatted)
    
    def debug(self, message: str, prefix:  str = "[DEBUG]"):
        """Log debug message (magenta)."""
        if self.verbose:
            print_debug(message, prefix)
        self.logger.debug(message)
    
    # ========================================================================
    # FORMATTING METHODS
    # ========================================================================
    
    def header(self, text: str, char: str = '=', width: int = 70):
        """Print formatted header."""
        print_header(text, char, width)
        self.logger.info(f"{'=' * width}")
        self.logger.info(text)
        self.logger.info(f"{'=' * width}")
    
    def section(self, text: str, char: str = '-', width: int = 70):
        """Print formatted section."""
        print_section(text, char, width)
        self.logger.info(f"{'-' * width}")
        self.logger.info(text)
    
    def progress(self, current: int, total: int, text: str = "Progress"):
        """Print progress indicator."""
        from utils.colors import print_progress
        print_progress(current, total, text)
        
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"{text}: {current}/{total} ({percentage:.1f}%)")
    
    def separator(self, char: str = '-', width: int = 70):
        """Print separator line."""
        sep = char * width
        print_colored(sep, Colors.DIM)
        self.logger.info(sep)
    
    def blank(self):
        """Print blank line."""
        print()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

# Global logger instance
_global_logger:  Optional[Logger] = None


def get_logger(name: str = "HPO",
               log_file: Optional[str] = None,
               verbose: bool = True) -> Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        verbose: If True, show debug messages
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.success("Ready!")
    """
    global _global_logger
    
    # Return existing logger if already created
    if _global_logger is not None:
        return _global_logger
    
    # Create new logger
    _global_logger = Logger(name=name, log_file=log_file, verbose=verbose)
    return _global_logger


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == '__main__':
    # Test logger
    logger = Logger(name="TEST", verbose=True)
    
    logger.header("LOGGER SYSTEM DEMO")
    
    logger.section("Log Levels")
    logger.success("This is a success message")
    logger.error("This is an error message")
    logger.warning("This is a warning message")
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    
    logger.blank()
    logger.section("Progress Indicators")
    logger.progress(1, 10, "Processing")
    logger.progress(5, 10, "Processing")
    logger.progress(10, 10, "Processing")
    
    logger.blank()
    logger.section("Separators")
    logger.separator()
    logger.separator('=')
    logger.separator('-', width=50)
    
    logger.blank()
    logger.header("DEMO COMPLETE")
    
    # Test with file logging
    print("\n" + "="*70)
    print("Testing file logging...")
    
    file_logger = Logger(name="FILE_TEST", log_file="test_log.txt", verbose=True)
    file_logger.info("This message goes to both console and file")
    file_logger.success("Log file created:  test_log.txt")
    
    print("\nCheck 'test_log.txt' to see file output")