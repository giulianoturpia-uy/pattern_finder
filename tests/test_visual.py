"""Visual tests: run each strategy and save an annotated image to eyeball.

Each test also asserts the outcome, so they double as regression tests.
After running, inspect the PNGs under tests/output/.
"""

import os

import cv2
import numpy as np
import pytest

from pattern_finder.strategies.geometric_strategy import GeometricStrategy
from pattern_finder.strategies.opencv_strategy import OpenCVStrategy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
STRATEGIES = {"opencv": OpenCVStrategy(), "geometric": GeometricStrategy()}


def make_wedge(angle=0):
    """A white quarter circle on black, rotated by `angle` degrees."""
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(img, (200, 200), (150, 150), angle, 0, 90, 255, -1)
    return img


def save_annotated(name, gray, vertices):
    """Save `gray` with the detected extremes marked, under tests/output/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y in vertices:
        cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), vis)


@pytest.mark.parametrize("name", STRATEGIES)
def test_finds_rotated_wedge(name):
    wedge = make_wedge(angle=30)

    result = STRATEGIES[name].find(wedge, wedge)

    assert result.found
    assert len(result.vertices) == 3
    save_annotated("wedge_{}.png".format(name), wedge, result.vertices)


@pytest.mark.parametrize("name", STRATEGIES)
def test_rejects_full_circle(name):
    circle = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(circle, (200, 200), 150, 255, -1)

    assert not STRATEGIES[name].find(circle, circle).found
