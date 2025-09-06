"""
HBN EEG Analysis Package

Advanced analysis tools for the Healthy Brain Network EEG database.
This project is designed to work as an experimental upgrade to EFG2025_HBN-EEG_databse.

Main modules:
- loader: Enhanced EEG data loader with BIDS and HED support
- visualization: Plotting and visualization utilities
"""

from .loader import SimpleConfig, create_default_config

__version__ = "0.1.0"
__author__ = "HBN-EEG Analysis Contributors"

__all__ = [
    "SimpleConfig",
    "create_default_config",
    ]
