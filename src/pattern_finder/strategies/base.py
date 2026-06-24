from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import abc
import numpy as np

@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a pattern-finding run.

    found:        success code — True if the pattern was located, else False.
                  When False, vertices is empty.
    vertices:     pixel coordinates (x, y) of each detected extreme, ordered
                  consistently. For the quarter-circle target this holds the
                  3 extremes: the apex followed by the two arc endpoints.
    corner_count: number of detected vertices (len(vertices)); kept for
                  convenience and to satisfy the corner-counting use case.
    """
    found: bool
    vertices: Tuple[Tuple[int, int], ...] = ()

    @property
    def corner_count(self) -> int:
        return len(self.vertices)


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
            Grayscale uint8 array of the search image (from .tif).

        Returns
        -------
        DetectionResult
            found=True with the 3 extremes on success; found=False with an
            empty vertices tuple when the pattern is not located.
        """
