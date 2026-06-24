"""Shared two-stage contour for the OpenCV-based strategies.

Both detection strategies locate the target the same way — downscale to find
it, then refine on a full-resolution crop — and validate the result the same
way (radii roughly equal, apex angle near 90 deg). They differ only in how the
3 extremes are extracted from the refined contour, which subclasses provide via
:meth:`_extract_extremes`. Template Method pattern.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import DetectionResult, PatternFinderStrategy

# A detected vertex in (x, y) pixel coordinates.
Point = Tuple[int, int]


class TwoStageContourStrategy(PatternFinderStrategy):
    """Locate the target via a downscale + full-res-crop refine pipeline.

    Shared parameters
    -----------------
    work_dim:
        Longest-side size (px) the locate stage downscales to.
    crop_margin:
        Padding (px, full-res) around the located box before the refine crop.
    canny_low, canny_high:
        Canny hysteresis thresholds.
    min_area:
        Minimum contour area in **full-resolution** px^2 (the downscaled area
        is normalised back to full resolution first, so it is independent of
        ``work_dim``).
    radii_ratio_min:
        Validation gate: shorter radius / longer radius must be at least this.
    apex_angle_range:
        Validation gate: accepted apex angle (deg); the quarter circle is 90.
    """

    def __init__(
        self,
        work_dim: int = 1000,
        crop_margin: int = 40,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: float = 1000.0,
        radii_ratio_min: float = 0.75,
        apex_angle_range: Tuple[float, float] = (80.0, 100.0),
    ) -> None:
        self.work_dim = work_dim
        self.crop_margin = crop_margin
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.radii_ratio_min = radii_ratio_min
        self.apex_angle_range = apex_angle_range

    # Invariant Method

    def find(self, pattern: np.ndarray, image: np.ndarray) -> DetectionResult:
        # The pattern fixes the shape we look for; detection is geometric.
        del pattern

        height, width = image.shape[:2]

        # Stage 1: locate on a downscaled copy.
        scale = min(1.0, float(self.work_dim) / max(height, width))
        if scale < 1.0:
            small = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        located = self._largest_contour(small)
        if located is None:
            return DetectionResult(found=False)
        # Normalise area to full-res px so min_area is work_dim-independent.
        if cv2.contourArea(located) / (scale * scale) < self.min_area:
            return DetectionResult(found=False)

        # Map the located bounding box back to full resolution, with margin.
        x, y, w, h = cv2.boundingRect(located)
        x0 = max(0, int(x / scale) - self.crop_margin)
        y0 = max(0, int(y / scale) - self.crop_margin)
        x1 = min(width, int((x + w) / scale) + self.crop_margin)
        y1 = min(height, int((y + h) / scale) + self.crop_margin)

        # Stage 2: refine on the full-resolution crop.
        crop = image[y0:y1, x0:x1]
        refined = self._largest_contour(crop)
        if refined is None:
            return DetectionResult(found=False)

        extremes = self._extract_extremes(refined)  # subclass-specific
        if extremes is None:
            return DetectionResult(found=False)

        # Shift crop-local coordinates back into the full image.
        apex, end1, end2 = ((px + x0, py + y0) for (px, py) in extremes)
        return DetectionResult(found=True, vertices=(apex, end1, end2))

    # -- the one varying step --------------------------------------------

    @abc.abstractmethod
    def _extract_extremes(
        self, contour: np.ndarray
    ) -> Optional[Tuple[Point, Point, Point]]:
        """Return (apex, arc_end1, arc_end2) from a contour, or None.

        Implementations should locate the three points then hand them to
        :meth:`_finalize`, which validates the geometry, orders the endpoints
        and casts to int tuples.
        """

    # Shared Helpers

    def _largest_contour(self, gray: np.ndarray):
        """Canny edge detection + contour, returning the largest contour.

        Returns None if no contour is found.
        """
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        # Dilate to close 1-px gaps so the boundary is a single closed contour.
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _finalize(
        self, apex: np.ndarray, end1: np.ndarray, end2: np.ndarray
    ) -> Optional[Tuple[Point, Point, Point]]:
        """Validate the sector, order the endpoints, return int-tuple points."""
        if not self._is_valid_sector(apex, end1, end2):
            return None
        end1, end2 = self._order_endpoints(apex, end1, end2)
        return (
            (int(apex[0]), int(apex[1])),
            (int(end1[0]), int(end1[1])),
            (int(end2[0]), int(end2[1])),
        )

    def _is_valid_sector(
        self, apex: np.ndarray, end1: np.ndarray, end2: np.ndarray
    ) -> bool:
        """Check the detected triangle looks like a circular sector."""
        v1 = np.asarray(end1, float) - np.asarray(apex, float)
        v2 = np.asarray(end2, float) - np.asarray(apex, float)
        r1 = float(np.hypot(*v1))
        r2 = float(np.hypot(*v2))
        if r1 == 0.0 or r2 == 0.0:
            return False
        if min(r1, r2) / max(r1, r2) < self.radii_ratio_min:
            return False
        cos_a = float(np.dot(v1, v2)) / (r1 * r2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = np.degrees(np.arccos(cos_a))
        lo, hi = self.apex_angle_range
        return lo <= angle <= hi

    def is_quarter_triangle(
        self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> bool:
        """True if the triangle matches a quarter circle, order-independent.

        Checks the widest angle is near 90 deg and the two sides meeting there
        are roughly equal (the right-isosceles 90/45/45 signature). Used when
        the caller does not need to know which vertex is the apex.
        """
        tri = (p0, p1, p2)
        widest = max(
            range(3),
            key=lambda i: self.angle_at(tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]),
        )
        return self._is_valid_sector(
            tri[widest], tri[(widest + 1) % 3], tri[(widest + 2) % 3]
        )

    @staticmethod
    def angle_at(vertex: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        """Interior angle (deg) at *vertex* of the triangle (vertex, p, q)."""
        v1 = np.asarray(p, float) - np.asarray(vertex, float)
        v2 = np.asarray(q, float) - np.asarray(vertex, float)
        n1 = float(np.hypot(*v1))
        n2 = float(np.hypot(*v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2)))
        return float(np.degrees(np.arccos(cos_a)))

    @staticmethod
    def _order_endpoints(
        apex: np.ndarray, end1: np.ndarray, end2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the two endpoints ordered by angle about the apex (CCW)."""
        a1 = np.arctan2(end1[1] - apex[1], end1[0] - apex[0])
        a2 = np.arctan2(end2[1] - apex[1], end2[0] - apex[0])
        return (end1, end2) if a1 <= a2 else (end2, end1)
