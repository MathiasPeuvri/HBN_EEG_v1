"""
Simplified HBN-EEG Data Loader

This package provides a minimal, efficient loader for HBN-EEG datasets
focusing on core functionality.
"""

from .simple_loader import SimpleHBNLoader, SimpleConfig, normalize_task_name

# Keep backward compatibility with EnhancedHBNLoader
from .config import DatabaseLoaderConfig, Subject, TaskConfig, create_default_config

__version__ = "0.2.0"
__author__ = "HBN-EEG Analysis Contributors"

# Expose main classes and functions
__all__ = [
    # Simplified loader (recommended)
    "SimpleHBNLoader",
    "SimpleConfig",
    "normalize_task_name",
    # Legacy enhanced loader (for backward compatibility)
    "EnhancedHBNLoader",
    "DatabaseLoaderConfig",
    "Subject",
    "TaskConfig",
    "create_default_config",
]
