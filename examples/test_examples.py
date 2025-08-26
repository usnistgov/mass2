import pytest


def test_ljh_mnkalpha():
    # enforce the order by doing it in the same function
    # parquet_after_ljh_mnkalpha needs to open files written by ljh_mnkalpha
    from . import ljh_mnkalpha as notebook
    notebook.app.run()
    from . import parquet_after_ljh_mnkalpha as notebook2
    notebook2.app.run()


def test_broken_notebook():
    from . import broken_notebook as notebook
    with pytest.raises(Exception):
        notebook.app.run()


def test_bessy():
    from . import bessy_20240727 as notebook
    notebook.app.run()

def test_truebq():
    from . import truqbq_from_parquet_202508 as notebook
    notebook.app.run()

def test_gamma():
    from . import gamma_20241005 as notebook
    notebook.app.run()

# currently fails due to raising a warning on an unclosed file
# @pytest.mark.filterwarnings("ignore:pytest.PytestUnraisableExceptionWarning")
# def test_ebit_july2024_mass_off():
#     from . import ebit_july2024_mass_off as notebook
#     notebook.app.run()
