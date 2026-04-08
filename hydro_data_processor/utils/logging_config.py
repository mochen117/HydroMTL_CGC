"""
Custom logging configuration for Hydro Data Processor.
"""

import logging
import sys
from typing import Optional


class PipelineFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return f"{record.getMessage()}"
        elif record.levelno == logging.WARNING:
            return f"WARNING: {record.getMessage()}"
        elif record.levelno == logging.ERROR:
            return f"ERROR: {record.getMessage()}"
        elif record.levelno == logging.DEBUG:
            return f"[DEBUG] {record.getMessage()}"
        else:
            return super().format(record)


class WarningFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        unwanted_messages = [
            "Missing expected subdirectories",
            "camels_name.txt not found",
            "No HUC2 mapping found for gauge",
            "camels_name.txt does not contain huc_02 column",
            "camels_name.txt not found in any expected location"
        ]
        for unwanted in unwanted_messages:
            if unwanted in message:
                return False
        return True


def setup_logging(verbose: bool = False, log_file: Optional[str] = None, level: Optional[str] = None):
    """
    Configure logging for the pipeline.

    Args:
        verbose: Enable verbose (DEBUG) logging (overrides level if True)
        log_file: Optional log file path
        level: Explicit logging level (DEBUG, INFO, WARNING, ERROR) - overrides verbose if provided
    """
    if level is not None:
        log_level = getattr(logging, level.upper(), logging.INFO)
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(PipelineFormatter())
    console_handler.addFilter(WarningFilter())
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    return root_logger


def log_section(title: str, logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 60}")