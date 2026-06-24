"""Simple timing helper: measure a call"""

import time


def timed_call(func, *args, **kwargs):
    """Call *func*, always print how long it took (ms), and return its result."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print("detection: {:.2f} ms".format(elapsed_ms))
    return result
