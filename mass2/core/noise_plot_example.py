import marimo

__generated_with = "0.7.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from mass2 import moss
    import pylab as plt

    return mo, moss, plt


@app.cell
def __(mo, moss, plt):
    noise_path = r"C:\Users\oneilg\Desktop\python\src\mass\tests\ljh_files\20230626\0000\20230626_run0000_chan4102.ljh"
    noise_ch = moss.NoiseChannel.from_ljh(noise_path)
    noise_ch.spectrum().plot()
    plt.title("warning, may have factor of 2 error in absolute scaling")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return noise_ch, noise_path


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
