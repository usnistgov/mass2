"""
Configuration file for pytest
"""

# import pytest
import warnings
import logging
import matplotlib

# Suppress matplotlib warnings during tests. See
# https://stackoverflow.com/questions/55109716/c-argument-looks-like-a-single-numeric-rgb-or-rgba-sequence
# from matplotlib.axes._axes import _log as matplotlib_axes_logger
# matplotlib_axes_logger.setLevel('ERROR')

matplotlib.use("svg")  # set to common backend so will run ci with fewer dependencies
warnings.filterwarnings("ignore")  # not sure what this does

# Raise the logging threshold, to reduce extraneous output during tests
LOG = logging.getLogger("mass")
LOG.setLevel(logging.ERROR)

def windows_monkey_patch_to_use_utf8_by_default_for_pathlib_read_text_otherwise_we_get_errors_in_test_mkdocs():
    import pathlib

    _original_read_text = pathlib.Path.read_text

    def _utf8_read_text(self, encoding=None, errors=None, newline=None):
        return _original_read_text(self, encoding=encoding or "utf-8", errors=errors, newline=newline)

    pathlib.Path.read_text = _utf8_read_text

windows_monkey_patch_to_use_utf8_by_default_for_pathlib_read_text_otherwise_we_get_errors_in_test_mkdocs()