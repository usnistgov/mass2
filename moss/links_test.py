import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __():
    a=4
    return a,


@app.cell
def __():
    a=4
    return a,


@app.cell
def __():
    b=5
    raise Exception()
    return b,


@app.cell
def __(a, b):
    c=a+b
    return c,


@app.cell
def __(c):
    d=c+1
    return d,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
