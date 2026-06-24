"""pattern_finder — constrained-environment pattern detection library."""

from .api import PatternFinder
from .strategies.base import DetectionResult, PatternFinderStrategy

__all__ = ["PatternFinder", "PatternFinderStrategy", "DetectionResult"]
