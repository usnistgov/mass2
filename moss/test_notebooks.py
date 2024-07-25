import pytest
import subprocess


# def test_example_marimo():
#     from . import example_marimo as notebook
#     notebook.app.run()

# def test_example_marimo_ebit_off():
#     from . import example_marimo_ebit_of as notebook
#     notebook.app.run()

def test_broken_notebook():
    from . import broken_notebook as notebook
    with pytest.raises(Exception):
        notebook.app.run()