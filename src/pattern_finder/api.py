"""Public API for pattern_finder.

`PatternFinder` is the *context* of the Strategy pattern: it owns a detection
strategy and an image loader — receive the target pattern and the search image,
return a success code plus the x,y coordinates of the 3 extremes.

Typical use
-----------
    from pattern_finder import PatternFinder
    from pattern_finder.strategies.geometric_strategy import GeometricStrategy

    finder = PatternFinder(GeometricStrategy())
    result = finder.find("patron.bmp", "imagen.tif")
    if result.found:
        print(result.vertices)   # ((ax, ay), (sx, sy), (ex, ey))
"""

from __future__ import annotations

from typing import Optional, Union

from .strategies.base import DetectionResult, PatternFinderStrategy
from .utils.image_io import ImageLoader

# Accepted strategy selector: a name to resolve, a ready instance, or None
# (which selects the default strategy).
StrategySelector = Optional[Union[str, PatternFinderStrategy]]

# Strategy names the context knows how to build.
STRATEGY_OPENCV = "opencv"
STRATEGY_GEOMETRIC = "geometric"

class PatternFinder:
    """Locate a quarter-circle pattern in an image and report its 3 extremes.

    The context owns strategy selection: pass it a strategy *name* and it
    builds the matching algorithm, or pass a ready strategy instance directly.

    Parameters
    ----------
    strategy:
        Either a strategy name ('opencv' or 'geometric') or a concrete
        :class:`PatternFinderStrategy` instance. Defaults to 'geometric',
        the minimal-dependency algorithm. Swappable at runtime via
        :meth:`set_strategy`.
    loader:
        Optional image loader. Defaults to a standard ``ImageLoader`` with the
        built-in memory guard; pass a custom one to change the size limit.
    """

    #: Strategy names the context can build. Exposed so callers (e.g. the CLI)
    #: can present the valid choices without hard-coding them.
    STRATEGIES = (STRATEGY_OPENCV, STRATEGY_GEOMETRIC)
    DEFAULT_STRATEGY = STRATEGY_GEOMETRIC

    def __init__(
        self,
        strategy: StrategySelector = None,
        loader: Optional[ImageLoader] = None,
    ) -> None:
        self._strategy = self._resolve_strategy(strategy)
        self._loader = loader if loader is not None else ImageLoader()

    @classmethod
    def _resolve_strategy(
        cls, strategy: StrategySelector
    ) -> PatternFinderStrategy:
        """Turn a name (or instance, or None) into a concrete strategy.

        Raises
        ------
        ValueError
            If the name is not one of ``cls.STRATEGIES``.
        NotImplementedError
            If the strategy is recognised but not yet implemented.
        """
        # An already-built strategy passes straight through.
        if isinstance(strategy, PatternFinderStrategy):
            return strategy

        name = cls.DEFAULT_STRATEGY if strategy is None else strategy
        if name not in cls.STRATEGIES:
            raise ValueError(
                "unknown strategy '{}' (choose from {})".format(
                    name, ", ".join(cls.STRATEGIES)
                )
            )

        if name == STRATEGY_OPENCV:
            raise NotImplementedError(
                "The 'opencv' strategy is not implemented yet."
            )
        # name == STRATEGY_GEOMETRIC

        raise NotImplementedError(
            "The 'geometric' strategy is not implemented yet."
        )

    def set_strategy(self, strategy: StrategySelector) -> None:
        """Replace the detection strategy (Strategy-pattern swap).

        Accepts a name or a concrete instance, same as the constructor.
        """
        self._strategy = self._resolve_strategy(strategy)

    @property
    def strategy(self) -> PatternFinderStrategy:
        return self._strategy

    def find(self, pattern_path: str, image_path: str) -> DetectionResult:
        """Find the pattern inside the image and return the detection result.

        Both arguments are filesystem paths to image files (.bmp / .tif). The
        context loads and converts them to grayscale via its configured
        :class:`ImageLoader`; the caller passes the files as-is.

        Parameters
        ----------
        pattern_path:
            Path to the pattern image (e.g. ``patron.bmp``).
        image_path:
            Path to the search image (e.g. ``imagen.tif``).

        Returns
        -------
        DetectionResult
            ``found`` is the success code; ``vertices`` holds the 3 extremes
            (apex, arc_start, arc_end) when found, and is empty otherwise.

        Raises
        ------
        ImageLoadError
            If either file cannot be loaded.
        """
        pattern_arr = self._loader(pattern_path)
        image_arr = self._loader(image_path)
        return self._strategy.find(pattern_arr, image_arr)
