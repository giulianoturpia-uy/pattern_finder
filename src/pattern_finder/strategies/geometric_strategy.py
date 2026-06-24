"""Geometric detection strategy: maximum-area inscribed triangle.

Locates the quarter-circle target with the shared two-stage contour,
then extracts the 3 extremes as the vertices of the **maximum-area triangle**
inscribed in the contour's convex hull.

For a 90 deg circular sector the largest triangle you can
inscribe has its corners exactly at the apex and the two arc endpoints (any
other choice has smaller area). Its angles are 90 / 45 / 45 deg, so the apex is
simply the vertex whose angle is closest to 90 deg.
Rotation-invariant by construction.
"""

from __future__ import annotations

import itertools
from typing import Optional, Tuple

import cv2
import numpy as np

from ._contour_base import Point, TwoStageContourStrategy


class GeometricStrategy(TwoStageContourStrategy):
    """Detect the quarter-circle via convex hull + max-area triangle.

    Adds to the shared parameters (see :class:`TwoStageContourStrategy`):

    hull_eps_frac:
        ``approxPolyDP`` epsilon (as a fraction of the hull perimeter) used to
        simplify the convex hull before the triangle search. Keeps the vertex
        count small so the search is cheap, while Douglas-Peucker preserves the
        sharp corners (apex and arc endpoints) we need.
    """

    def __init__(
        self,
        work_dim: int = 1000,
        crop_margin: int = 40,
        canny_low: int = 50,
        canny_high: int = 150,
        hull_eps_frac: float = 0.01,
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
        self.hull_eps_frac = hull_eps_frac

    def _extract_extremes(
        self, contour: np.ndarray
    ) -> Optional[Tuple[Point, Point, Point]]:
        # Convex hull, then simplify so the triangle search runs on few points.
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        hull = cv2.approxPolyDP(hull, self.hull_eps_frac * peri, True)
        pts = hull.reshape(-1, 2)
        n = len(pts)
        if n < 3:
            return None

        # Maximum-area triangle among the hull vertices.
        best_area = 0.0
        best = None
        for i, j, k in itertools.combinations(range(n), 3):
            area = self._triangle_area(pts[i], pts[j], pts[k])
            if area > best_area:
                best_area = area
                best = (pts[i], pts[j], pts[k])
        if best is None:
            return None

        # The 3 vertices are the extremes. We do not label which is the apex;
        # validate (order-independently) that they form a quarter-circle
        # triangle, then return them as-is.
        if not self.is_quarter_triangle(*best):
            return None
        a, b, c = best
        return (
            (int(a[0]), int(a[1])),
            (int(b[0]), int(b[1])),
            (int(c[0]), int(c[1])),
        )

    @staticmethod
    def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Area of triangle (a, b, c) via the cross-product (shoelace)."""
        return abs(
            (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        ) / 2.0
