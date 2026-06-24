"""OpenCV-based detection strategy.

Uses OpenCV's edge-detection and contour tools to locate the quarter-circle
target and report its 3 extremes (apex + 2 arc endpoints).

Design notes
------------
The search image can be very large while the target is
small, so a single full-resolution pass would obliterate timing constraints. The
strategy therefore works in two stages:

1. **Locate** — downscale the image to ``work_dim`` and find the target's
   bounding box quickly.
2. **Refine** — re-run detection on a small full-resolution crop around that
   box, so the reported coordinates keep full-resolution precision.

The 3 extremes are extracted geometrically:
- On the contour's polygon approximation the two longest edges are the straight radii of the sector.
- Their shared vertex is the apex and their far ends are the arc endpoints.

This is rotation-invariant by construction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import DetectionResult, PatternFinderStrategy

# A detected vertex in (x, y) pixel coordinates.
Point = Tuple[int, int]


class OpenCVStrategy(PatternFinderStrategy):
    """Detect the quarter-circle via Canny edges + contour analysis.

    Parameters
    ----------
    work_dim:
        Longest-side size (px) the locate stage downscales to. Smaller is
        faster but coarser.
    crop_margin:
        Padding (px, full-res) added around the located bounding box before
        the refine stage, so the whole shape is comfortably inside the crop.
    canny_low, canny_high:
        Hysteresis thresholds for Canny.
    approx_eps_frac:
        ``approxPolyDP`` epsilon as a fraction of the contour perimeter.
    min_area:
        Minimum contour area in **full-resolution** px^2 to be considered a
        real target rather than noise. The downscaled locate-stage area is
        normalised back to full resolution before this check, so the value is
        independent of ``work_dim``.
    radii_ratio_min:
        Geometric gate: the shorter radius must be at least this fraction of
        the longer one for the shape to count as a circular sector.
    apex_angle_range:
        Geometric gate: accepted apex angle in degrees (the quarter circle is
        90 deg; a tolerant window allows rotations and discretisation to be considered).
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
        apex_angle_range: Tuple[float, float] = (50.0, 130.0),
    ) -> None:
        self.work_dim = work_dim
        self.crop_margin = crop_margin
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.approx_eps_frac = approx_eps_frac
        self.min_area = min_area
        self.radii_ratio_min = radii_ratio_min
        self.apex_angle_range = apex_angle_range

    # Public API

    def find(self, pattern: np.ndarray, image: np.ndarray) -> DetectionResult:
        # The pattern fixes the shape we are looking for (a quarter circle,
        # 3 extremes); detection itself is shape-geometric on the image.
        del pattern

        height, width = image.shape[:2]

        # Stage 1: Locate on a downscaled image. Find the bounding box
        scale = min(1.0, float(self.work_dim) / max(height, width))
        if scale < 1.0:
            small = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        # Stage 2: Locate the largest contour in the downscaled image.
        located = self._largest_contour(small)
        if located is None:
            return DetectionResult(found=False)
        # Normalise the area back to full-resolution pixels so the min_area
        # threshold is expressed in real image units, independent of dimension.
        area_full = cv2.contourArea(located) / (scale * scale)
        if area_full < self.min_area:
            return DetectionResult(found=False)

        # Stage 3: Map the located bounding box back to full resolution, with margin.
        x, y, w, h = cv2.boundingRect(located)
        x0 = max(0, int(x / scale) - self.crop_margin)
        y0 = max(0, int(y / scale) - self.crop_margin)
        x1 = min(width, int((x + w) / scale) + self.crop_margin)
        y1 = min(height, int((y + h) / scale) + self.crop_margin)

        # Stage 4: Refine the image (full resolution) crop to get the 3 extremes precisely.
        crop = image[y0:y1, x0:x1]
        refined = self._largest_contour(crop)
        if refined is None:
            return DetectionResult(found=False)

        # Stage 5: Extract the 3 extremes from the refined contour.
        extremes = self._extract_extremes(refined)
        if extremes is None:
            return DetectionResult(found=False)

        # Stage 6: Shift local coordinates back into the full image.
        apex, end1, end2 = (
            (px + x0, py + y0) for (px, py) in extremes
        )
        return DetectionResult(found=True, vertices=(apex, end1, end2))

    # -- Internals --------------------------------------------------------

    def _largest_contour(self, gray: np.ndarray):
        """Canny edge detection + contour, returning the largest contour.

        Returns None if no contour is found.
        """
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        # Dilate to close 1-px gaps in the edge so the boundary is a single
        # closed contour rather than broken arcs.
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _extract_extremes(
        self, contour: np.ndarray
    ) -> Optional[Tuple[Point, Point, Point]]:
        """Return (apex, arc_end1, arc_end2) or None if not a valid sector.

        The two longest edges of the polygon approximation are the straight
        radii; their shared vertex is the apex and their far ends are the arc
        endpoints. The result is validated geometrically (radii roughly equal,
        apex angle near 90 deg) before being accepted.
        """
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
            edges.append((float(np.hypot(b[0] - a[0], b[1] - a[1])), i, (i + 1) % n))
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

        apex = pts[apex_idx]
        end1 = pts[end1_idx]
        end2 = pts[end2_idx]

        if not self._is_valid_sector(apex, end1, end2):
            return None

        # Order the two arc endpoints deterministically (by angle about apex).
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
        v1 = end1 - apex
        v2 = end2 - apex
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

    @staticmethod
    def _order_endpoints(
        apex: np.ndarray, end1: np.ndarray, end2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the two endpoints ordered by angle about the apex (CCW)."""
        a1 = np.arctan2(end1[1] - apex[1], end1[0] - apex[0])
        a2 = np.arctan2(end2[1] - apex[1], end2[0] - apex[0])
        return (end1, end2) if a1 <= a2 else (end2, end1)
