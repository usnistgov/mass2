import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    assert 4 + 4 == 8
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
