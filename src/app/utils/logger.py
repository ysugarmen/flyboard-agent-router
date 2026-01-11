import logging
import sys
from typing import Optional, Union

from logging.handlers import TimedRotatingFileHandler

from src.app.core.config import get_settings

settings = get_settings()


def _resolve_log_level(log_level: Optional[Union[int, str]]) -> int:
    """Translate env/config log levels to logging ints."""

    value = log_level if log_level is not None else settings.LOG_LEVEL
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = getattr(logging, value.upper(), None)
        if isinstance(candidate, int):
            return candidate
        if value.isdigit():
            return int(value)
    raise ValueError(f"Invalid log level: {value}")


def get_logger(name: str, log_level: Optional[Union[int, str]] = None, log_file: str = "") -> logging.Logger:
    """Simple logger setup for single-process services honoring env LOG_LEVEL."""

    logger = logging.getLogger(name)
    level = _resolve_log_level(log_level)
    logger.setLevel(level)
    logger.propagate = False  # Prevents messages from being passed to parent loggers

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file:
            file_handler = TimedRotatingFileHandler(log_file, when='D', interval=2, backupCount=5)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger