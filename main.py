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

    # Cheap header-only check to confirm both files are readable, without
    # decoding the (potentially huge) pixel data twice — find() does the
    # real load below.
    loader = ImageLoader()
    try:
        pw, ph = loader.image_dimensions(pattern_path)
        iw, ih = loader.image_dimensions(image_path)
    except ImageLoadError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR
    print("pattern {}: {}x{} (OK)".format(pattern_path, pw, ph))
    print("image   {}: {}x{} (OK)".format(image_path, iw, ih))

    # The context decides which strategy to build; we just report any problem.
    try:
        finder = PatternFinder(strategy_name, loader=loader)
    except (ValueError, NotImplementedError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    # The API takes the file paths and does the loading/conversion itself.
    try:
        result = finder.find(pattern_path, image_path)
    except ImageLoadError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_ERROR

    print("Pattern detected in {}:".format(image_path))

if __name__ == "__main__":
    sys.exit(main())
