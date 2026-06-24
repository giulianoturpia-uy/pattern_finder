from __future__ import annotations

from typing import Tuple
from PIL import Image

import os
import numpy as np

# Guard against decoding an image so large it would exhaust RAM on the
# device. 64 megapixels of grayscale uint8 is ~64 MB; well within
# budget while still rejecting very large images.
DEFAULT_MAX_PIXELS = 64 * 1024 * 1024

class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or fails validation."""

class ImageLoader:
    """Loads image files as grayscale uint8 numpy arrays.

    The memory guard (max_pixels) is configured once at construction, so
    every load through this instance shares the same limit. The instance
    is also callable: ``loader(path)`` is shorthand for
    ``loader.load_grayscale(path)``.
    """

    def __init__(self, max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
        self.max_pixels = max_pixels

    def __call__(self, path: str) -> np.ndarray:
        return self.load_grayscale(path)

    @staticmethod
    def _validate_path(path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ImageLoadError("Image path must be a non-empty string.")
        if not os.path.isfile(path):
            raise ImageLoadError("Image file does not exist: {}".format(path))

    def image_dimensions(self, path: str) -> Tuple[int, int]:
        """Return (width, height) from the image header without decoding pixels.

        Useful for pre-flight checks on the device before committing memory.
        """
        self._validate_path(path)
        try:
            with Image.open(path) as img:
                return img.size
        except Exception as exc:
            raise ImageLoadError(
                "Failed to read image header {}: {}".format(path, exc)
            ) from exc

    def load_grayscale(self, path: str) -> np.ndarray:
        """Load an image file as a grayscale uint8 numpy array.

        Uses Pillow to decode the image, then converts to 8-bit grayscale and
        returns a 2-D numpy array.

        Parameters
        ----------
        path:
            Filesystem path to the image (.bmp pattern or .tif search image).

        Returns
        -------
        np.ndarray
            2-D array of shape (height, width), dtype uint8.

        Raises
        ------
        ImageLoadError
            If the path is invalid, the file exceeds self.max_pixels, or it
            cannot be decoded as an image.
        """
        # Header-only pre-flight check: validates the path and reads the
        # dimensions without decoding pixels, so an oversized image is
        # rejected before any large allocation.
        width, height = self.image_dimensions(path)
        if width * height > self.max_pixels:
            raise ImageLoadError(
                "Image {} is too large: {}x{} exceeds the {}-pixel "
                "limit.".format(path, width, height, self.max_pixels)
            )

        try:
            with Image.open(path) as img:
                # "L" = 8-bit grayscale. Converts here so strategies never have
                # to reason about colour channels or bit depth.
                gray = img.convert("L")
                return np.asarray(gray, dtype=np.uint8)
        except Exception as exc:  # Pillow raises a variety of error types.
            raise ImageLoadError(
                "Failed to load image {}: {}".format(path, exc)
            ) from exc


