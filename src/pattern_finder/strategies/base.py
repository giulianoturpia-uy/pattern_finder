from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a pattern-finding run.

    corner_count: number of geometric vertices of the detected pattern.
    vertices:     pixel coordinates (x, y) of each detected vertex,
                  ordered consistently.
    """
    corner_count: int
    vertices: Tuple[Tuple[int, int], ...] = ()


class PatternFinderStrategy(abc.ABC):
    """Abstract strategy for locating a ".bmp" pattern inside a ".tif" image.

    Implementations receive raw pixel arrays (numpy uint8 grayscale) so
    that image I/O is handled once at the API layer, keeping strategies
    free of file-format concerns.
    """

    @abc.abstractmethod
    def find(
        self,
        pattern: "np.ndarray",
        image: "np.ndarray",
    ) -> DetectionResult:
        """Locate *pattern* in *image* and return the corner geometry.

        Parameters
        ----------
        pattern:
            Grayscale uint8 array of the template image (from .bmp).
        image:
            Grayscale uint8 array of the full image image (from .tif).

        Returns
        -------
        DetectionResult
            Always populated; strategies must guarantee corner_count >= 0.
        """
