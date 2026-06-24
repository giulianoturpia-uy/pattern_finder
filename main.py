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
from pattern_finder.utils.image_io import ImageLoader, ImageLoadError
from pattern_finder.utils.timing import timed_call

EXIT_FOUND = 0
EXIT_NOT_FOUND = 1
EXIT_ERROR = 2

USAGE = "usage: python main.py PATTERN IMAGE [STRATEGY]"

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    # The only thing this script validates is the argument count.
    if len(argv) not in (2, 3):
        print(USAGE, file=sys.stderr)
        return EXIT_ERROR

    pattern_path = argv[0]
    image_path = argv[1]
    strategy_name = argv[2] if len(argv) == 3 else None

    # PatternFinder decides which strategy to build
    try:
        finder = PatternFinder(strategy_name)
    except (ValueError, NotImplementedError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    # Wrap the time measurement around the actual "find" of the pattern in the image.
    try:
        result = timed_call(finder.find, pattern_path, image_path)
    except ImageLoadError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    print("Pattern detected in {}:".format(image_path))

if __name__ == "__main__":
    sys.exit(main())
