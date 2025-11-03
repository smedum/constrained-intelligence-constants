
"""
Constrained Intelligence Constants
A foundational framework for discovering and applying mathematical constants 
in bounded intelligent systems.
"""

__version__ = "1.0.0"
__author__ = "Constrained Intelligence Research Team"

from .core import ConstantsMeasurement, OptimizationEngine, BoundedSystemAnalyzer
from .discovery import ConstantDiscovery, DiscoveryMethods
from .constants import *

__all__ = [
    "ConstantsMeasurement",
    "OptimizationEngine", 
    "BoundedSystemAnalyzer",
    "ConstantDiscovery",
    "DiscoveryMethods",
    # Constants
    "GOLDEN_RATIO",
    "EULER_NUMBER",
    "PI",
    "NATURAL_LOG_BASE",
    "OPTIMAL_RESOURCE_SPLIT",
    "CONVERGENCE_THRESHOLD_FACTOR",
    "LEARNING_RATE_BOUNDARY",
    "MAX_EFFICIENCY_RATIO",
    "MINIMAL_COMPLEXITY_CONSTANT",
    "INFORMATION_DENSITY_LIMIT",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "VALIDATION_SIGNIFICANCE_LEVEL",
]
