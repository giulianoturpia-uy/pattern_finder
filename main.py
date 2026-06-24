#!/usr/bin/env python3
"""Command-line demo for pattern_finder.

Usage
-----
    python main.py PATTERN IMAGE [STRATEGY]

    PATTERN    path to the pattern image (.bmp)
    IMAGE      path to the search image (.tif)
    STRATEGY   optional: 'opencv' or 'geometric' (default: geometric)

Exit codes
----------
    0   pattern found        (coordinates printed)
    1   pattern not found
    2   usage / I/O error
"""

from __future__ import annotations

import sys

from pattern_finder import PatternFinder
from pattern_finder.utils.image_io import ImageLoadError

EXIT_FOUND = 0
EXIT_NOT_FOUND = 1
EXIT_ERROR = 2

# Hard real-time budget from the specification (applies to compute, not I/O).
BUDGET_MS = 100.0

USAGE = "usage: python main.py PATTERN IMAGE [STRATEGY]"


def _print_result(result):
    """Display the detection outcome: status code + the 3 extremes."""
    if not result.found:
        print("status: NOT_FOUND")
        return
    print("status: FOUND")
    labels = ("apex", "arc_start", "arc_end")
    for i, (x, y) in enumerate(result.vertices):
        label = labels[i] if i < len(labels) else "vertex_{}".format(i)
        print("  {:<10} x={:<6} y={:<6}".format(label + ":", x, y))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    # The only thing this script validates is the argument count.
    if len(argv) not in (2, 3):
        print(USAGE, file=sys.stderr)
        return EXIT_ERROR

    pattern_path = argv[0]
    image_path = argv[1]
    strategy_name = argv[2] if len(argv) == 3 else None

    # PatternFinder decides which strategy to build.
    try:
        finder = PatternFinder(strategy_name)
    except (ValueError, NotImplementedError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    # The API loads the files and runs detection, timing each part separately.
    try:
        result = finder.find(pattern_path, image_path)
    except ImageLoadError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    _print_result(result)

    # Check Timings
    within = "OK" if finder.last_compute_ms <= BUDGET_MS else "OVER BUDGET"
    print("load:    {:.1f} ms".format(finder.last_load_ms))
    print("compute: {:.1f} ms / {:.0f} ms budget ({})".format(
        finder.last_compute_ms, BUDGET_MS, within))

    return EXIT_FOUND if result.found else EXIT_NOT_FOUND

if __name__ == "__main__":
    sys.exit(main())
