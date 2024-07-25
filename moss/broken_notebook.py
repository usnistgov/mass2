import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __():
    raise Exception()
    return


if __name__ == "__main__":
    app.run()
