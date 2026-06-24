"""OpenCV detection strategy: longest-edges extraction.

Locates the quarter-circle target with the shared two-stage contour pipeline
(see :class:`TwoStageContourStrategy`), then extracts the 3 extremes from the
contour's polygon approximation: the two longest edges of the approximation are
the straight radii of the sector, so their shared vertex is the apex and their
far ends are the arc endpoints. Rotation-invariant by construction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ._contour_base import Point, TwoStageContourStrategy


class OpenCVStrategy(TwoStageContourStrategy):
    """Detect the quarter-circle via Canny contour + longest-edge analysis.

    Adds to the shared parameters (see :class:`TwoStageContourStrategy`):

    approx_eps_frac:
        ``approxPolyDP`` epsilon as a fraction of the contour perimeter. Small
        enough to keep the arc as a polyline while straightening the radii.
    """

    def __init__(
        self,
        work_dim: int = 1000,
        crop_margin: int = 40,
        canny_low: int = 50,
        canny_high: int = 150,
        approx_eps_frac: float = 0.02,
        min_area: float = 1000.0,
        radii_ratio_min: float = 0.75,
        apex_angle_range: Tuple[float, float] = (80.0, 100.0),
    ) -> None:
        super().__init__(
            work_dim=work_dim,
            crop_margin=crop_margin,
            canny_low=canny_low,
            canny_high=canny_high,
            min_area=min_area,
            radii_ratio_min=radii_ratio_min,
            apex_angle_range=apex_angle_range,
        )
        self.approx_eps_frac = approx_eps_frac

    def _extract_extremes(
        self, contour: np.ndarray
    ) -> Optional[Tuple[Point, Point, Point]]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_eps_frac * peri, True)
        pts = approx.reshape(-1, 2)
        n = len(pts)
        if n < 3:
            return None

        # Lengths of every (closed) polygon edge, as (length, i, j).
        edges = []
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            edges.append(
                (float(np.hypot(b[0] - a[0], b[1] - a[1])), i, (i + 1) % n)
            )
        edges.sort(reverse=True)

        # The two longest edges are the radii; find their shared vertex.
        set1 = {edges[0][1], edges[0][2]}
        set2 = {edges[1][1], edges[1][2]}
        shared = set1 & set2
        if not shared:
            return None  # radii do not meet at a vertex; not a clean sector
        apex_idx = shared.pop()
        end1_idx = (set1 - {apex_idx}).pop()
        end2_idx = (set2 - {apex_idx}).pop()

        return self._finalize(pts[apex_idx], pts[end1_idx], pts[end2_idx])
