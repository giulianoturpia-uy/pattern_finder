"""Resource tests: detection must fit the low-end device budget.

- Memory: peak RSS (real physical memory used) stays well under 4 GB.
- Compute: under 100 ms. This is only a regression guard - the x86 dev host is
  faster than the ARMv7 target, so passing here does not prove the device meets
  the budget, but failing here certainly would.
"""

import os
import resource

import pytest

from pattern_finder import PatternFinder

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
PATTERN = os.path.join(IMAGES, "patron.bmp")
IMAGE = os.path.join(IMAGES, "imagen.tif")

# Skip the whole module if the sample images are not present.
pytestmark = pytest.mark.skipif(
    not (os.path.exists(PATTERN) and os.path.exists(IMAGE)),
    reason="sample images not present",
)


@pytest.mark.parametrize("strategy", ["opencv", "geometric"])
def test_memory_under_budget(strategy):
    finder = PatternFinder(strategy)
    finder.find(PATTERN, IMAGE)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert peak_mb < 1024  # measured ~300 MB; 4 GB device has plenty of room


@pytest.mark.parametrize("strategy", ["opencv", "geometric"])
def test_compute_under_budget(strategy):
    finder = PatternFinder(strategy)
    finder.find(PATTERN, IMAGE)

    assert finder.last_compute_ms < 100  # measured ~20 ms
