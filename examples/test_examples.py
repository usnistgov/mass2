import pytest
import subprocess


# def test_example_marimo():
#     import moss
#     import moss.example_marimo
#     moss.example_marimo.app.run()


def test_off_ebit():
    from . import off_ebit as notebook
    notebook.app.run()

def test_broken_notebook():
    from . import broken_notebook as notebook
    with pytest.raises(Exception):
        notebook.app.run()