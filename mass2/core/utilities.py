"""
Various utility functions and classes:

* MouseClickReader: a class to use as a callback for reading mouse click
    locations in matplotlib plots.
* InlineUpdater: a class that loops over a generator and prints a message to
    the terminal each time it yields.
"""

from collections.abc import Callable
from typing import Any
import functools
import time
import sys
import logging


class InlineUpdater:
    """A class to print progress updates to the terminal."""

    def __init__(self, baseString: str):
        self.fracDone = 0.0
        self.minElapseTimeForCalc = 1.0
        self.startTime = time.time()
        self.baseString = baseString
        self.logger = logging.getLogger("mass")

    def update(self, fracDone: float) -> None:
        """Update the progress to the given fraction done."""
        if self.logger.getEffectiveLevel() >= logging.WARNING:
            return
        self.fracDone = fracDone
        sys.stdout.write(f"\r{self.baseString} {self.fracDone * 100.0:.1f}% done, estimated {self.timeRemainingStr} left")
        sys.stdout.flush()
        if fracDone >= 1:
            sys.stdout.write(f"\n{self.baseString} finished in {self.elapsedTimeStr}\n")

    @property
    def timeRemaining(self) -> float:
        """Estimate of time remaining in seconds, or -1 if not enough information yet."""
        if self.elapsedTimeSec > self.minElapseTimeForCalc and self.fracDone > 0:
            fracRemaining = 1 - self.fracDone
            rate = self.fracDone / self.elapsedTimeSec
            try:
                return fracRemaining / rate
            except ZeroDivisionError:
                return -1
        else:
            return -1

    @property
    def timeRemainingStr(self) -> str:
        """String version of time-remaining estimate."""
        timeRemaining = self.timeRemaining
        if timeRemaining == -1:
            return "?"
        else:
            return "%.1f min" % (timeRemaining / 60.0)

    @property
    def elapsedTimeSec(self) -> float:
        """Elapsed time in seconds since the creation of this object."""
        return time.time() - self.startTime

    @property
    def elapsedTimeStr(self) -> str:
        """String version of elapsed time."""
        return "%.1f min" % (self.elapsedTimeSec / 60.0)


class NullUpdater:
    """A do-nothing updater class with the same API as InlineUpdater."""

    def update(self, f: float) -> None:
        """Do nothing."""
        pass


def show_progress(name: str) -> Callable:
    """A decorator to show progress updates for another function."""

    def decorator(func: Callable) -> Callable:
        """A decorator to show progress updates for another function."""

        @functools.wraps(func)
        def work(self: Any, *args: Any, **kwargs: Any) -> None:
            """Update the progress of the wrapped function."""
            try:
                if "sphinx" in sys.modules:  # supress output during doctests
                    print_updater = NullUpdater()
                else:
                    print_updater = self.updater(name)
            except TypeError:
                print_updater = NullUpdater()

            for d in func(self, *args, **kwargs):
                print_updater.update(d)

        return work

    return decorator
