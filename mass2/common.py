"""
common.py

Stand-along functions that can be used throughout MASS.
"""

import six
from typing import Any


def isstr(s: Any) -> bool:
    """Is s a string or a bytes type? Make python 2 and 3 compatible."""
    return isinstance(s, (six.string_types, bytes))


def tostr(s: bytes | str) -> str:
    """Return a string version of `s` whether it's already a string or bytes"""
    if isinstance(s, bytes):
        return s.decode()
    return str(s)
