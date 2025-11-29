"""Logging utilities for RelevantAI."""

import logging
import time
from typing import Optional


class RAGFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m"
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, json_output: bool = False):
        super().__init__()
        self.use_colors = use_colors
        self.json_output = json_output

    def format(self, record: logging.LogRecord) -> str:
        if self.json_output:
            import json
            return json.dumps({
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage()
            })
        
        msg = f"[{self.formatTime(record)}] {record.levelname:8} {record.name}: {record.getMessage()}"
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"
        return msg


def get_logger(name: str, level: int = logging.INFO, use_colors: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(RAGFormatter(use_colors=use_colors))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class LogTimer:
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.DEBUG):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start: Optional[float] = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.logger.log(self.level, f"{self.operation} completed in {elapsed:.3f}s")
