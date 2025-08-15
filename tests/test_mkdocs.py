import pathlib
import pytest

from mktestdocs import check_md_file

"""test_mkdocs

    Run tests in which we search for every markdown file in and under the `docs/` directory,
    and check it for runnable python, and be sure that mktestdocs is happy.

    See https://github.com/koaning/mktestdocs for where this script comes from.

    Note the use of `ids=str`, makes for pretty output
"""


doc_paths = pathlib.Path("docs").glob("**/*.md")


@pytest.mark.parametrize("fpath", doc_paths, ids=str)
def test_files_good(fpath):
    print(f"Testing {fpath}")
    check_md_file(fpath=fpath, memory=True)
