# Unit testing

If you want to run the unit tests for `mass2` go to your `mass2` directory and do `pytest .`

If you want to add tests to `mass2`, just use simple, modern tests in the `pytest`. Put any new tests in a file somewhere inside `tests` with a name like `test_myfeature.py`, it must match the pattern `test_*.py` to be found by pytest. Use assertions, `numpy.allclose`, and similar tools to test that outcomes match expectations.

On each commit to develop, the tests will be run automatically by GitHub Actions. See [results of recent tests](https://github.com/usnistgov/mass2/actions).


## Documentation

We could use a lot of help with documentation. As of September 2, 2025, we used GitHub's AI chatbot to generate all 450+ missing doc strings. We think most of them look pretty reasonable, and many are excellent. But few use the desired numpy style for docstrings. Please consider upgrading any docstrings that touch on your work as you develop Mass2.
