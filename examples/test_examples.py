import pytest


@pytest.mark.filterwarnings("UserWarning: Using UFloat objects with std_dev==0 may give unexpected results.")
def test_ljh_mnkalpha():
    # enforce the order by doing it in the same function
    # parquet_after_ljh_mnkalpha needs to open files written by ljh_mnkalpha
    from . import ljh_mnkalpha as notebook
    notebook.app.run()
    from . import parquet_after_ljh_mnkalpha as notebook2
    notebook2.app.run()


@pytest.mark.filterwarnings("UserWarning: Using UFloat objects with std_dev==0 may give unexpected results.")
def test_off_ebit():
    from . import off_ebit as notebook
    notebook.app.run()


def test_broken_notebook():
    from . import broken_notebook as notebook
    with pytest.raises(Exception):
        notebook.app.run()


@pytest.mark.filterwarnings("UserWarning: Using UFloat objects with std_dev==0 may give unexpected results.")
def test_ebit_july2024_from_off():
    from . import ebit_july2024_from_off as notebook
    notebook.app.run()

# currently fails due to raising a warning on an unclosed file
# @pytest.mark.filterwarnings("ignore:pytest.PytestUnraisableExceptionWarning")
# def test_ebit_july2024_mass_off():
#     from . import ebit_july2024_mass_off as notebook
#     notebook.app.run()
